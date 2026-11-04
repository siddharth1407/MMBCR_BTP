from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from parameters import R, G, M_CH4, M_H2, M_AR, KineticParameters
from properties import Le2023Properties
from thermodynamics import Le2023Thermo
from reaction_only_solver import solve_reaction_only

# ---------------- benchmark ----------------
D=0.03; H=1.15; P_IN=2e5; T=1040+273.15
J_CH4_IN=1.2212e-4; J_H2_IN=0.0; J_AR_IN=7.6035e-5
KIN=KineticParameters(); props=Le2023Properties(); thermo=Le2023Thermo()
rhoL=float(props.rho_liquid(T)); muL=float(props.mu_liquid(T)); sigma=float(props.sigma_liquid(T))

# inlet mass fractions
JT_IN=J_CH4_IN+J_AR_IN
XIN=np.array([J_CH4_IN/JT_IN,0.0,J_AR_IN/JT_IN])

DTYPE=torch.float64
torch.set_default_dtype(DTYPE)

class Net(nn.Module):
    def __init__(self,width=64,depth=3):
        super().__init__()
        layers=[nn.Linear(1,width),nn.Tanh()]
        for _ in range(depth-1): layers += [nn.Linear(width,width),nn.Tanh()]
        layers += [nn.Linear(width,4)]  # log Jt, 3 composition logits, pressure correction
        self.net=nn.Sequential(*layers)
    def forward(self,x):
        o=self.net(x)
        logjt=np.log(JT_IN)+o[:,0:1]
        jt=torch.exp(logjt)
        probs=torch.softmax(o[:,1:4],dim=1)
        P=P_IN*torch.exp(o[:,4:5]) if o.shape[1]>4 else P_IN*torch.exp(o[:,3:4])
        # There are only 4 outputs: logJt + 3 logits. Use an additional
        # pressure channel by keeping output width 5.
        return jt, probs, P

# redefine network with 5 outputs
class PINN(nn.Module):
    def __init__(self,width=64,depth=3):
        super().__init__()
        seq=[nn.Linear(1,width),nn.Tanh()]
        for _ in range(depth-1): seq += [nn.Linear(width,width),nn.Tanh()]
        seq += [nn.Linear(width,5)]
        self.net=nn.Sequential(*seq)
    def forward(self,xi):
        o=self.net(xi)
        jt=(JT_IN*torch.exp(o[:,0])).reshape(-1)
        x=torch.softmax(o[:,1:4],dim=1)
        P=(P_IN*torch.exp(o[:,4])).reshape(-1)
        J=jt[:,None]*x
        return J,P,x,jt

def residuals(model,xi):
    xi=xi.requires_grad_(True)
    J,P,x,jt=model(xi)
    h=H*xi
    # Torch scalar constants
    TT=torch.tensor(T); rhoL_t=torch.tensor(rhoL); muL_t=torch.tensor(muL); sig_t=torch.tensor(sigma); G_t=torch.tensor(G); D_t=torch.tensor(D)
    # Ideal gas / hydrodynamics
    Mmix=1.0/(x[:,0]/M_CH4+x[:,1]/M_H2+x[:,2]/M_AR)
    rhoG=P*Mmix/(R*TT)
    ug=jt/rhoG
    # drift-flux closure (churn-turbulent)
    Uchar=(sig_t*G_t*(rhoL_t-rhoG)/rhoL_t**2).clamp_min(1e-20)**0.25
    uplus=ug/Uchar
    Nmu=muL_t/torch.sqrt(rhoL_t*sig_t*torch.sqrt(sig_t/(G_t*(rhoL_t-rhoG).clamp_min(1e-20))))
    DH=D_t/torch.sqrt(sig_t/(G_t*(rhoL_t-rhoG).clamp_min(1e-20)))
    rr=rhoL_t/rhoG
    v1a=0.92*rr**(-0.157)*Nmu**(-0.562)
    v1b=0.0019*DH**0.809*rr**(-0.157)*Nmu**(-0.562)
    v1=torch.where(DH>=30,v1a,v1b)
    v2=torch.sqrt(torch.tensor(2.0,dtype=DTYPE))
    e=torch.exp(-1.39*uplus)
    vp=v2*e+v1*(1-e)
    C0=1.2-0.2*torch.sqrt((rhoG/rhoL_t).clamp_min(1e-20))
    a=uplus/(uplus*C0+vp)
    NBo=G_t*D_t**2*rhoL_t/sig_t
    nuL=muL_t/rhoL_t
    NGa=G_t*D_t**3/nuL**2
    NFr=ug/torch.sqrt(G_t*D_t)
    db=26*NBo**(-0.50)*NGa**(-0.12)*NFr**(-0.12)*D_t
    a_s=6/db
    # kinetics
    Cch4=rhoG*x[:,0]/M_CH4
    Ch2=rhoG*x[:,1]/M_H2
    Kdim=torch.exp(torch.tensor(13.2714)-torch.tensor(91204.6)/(R*TT))
    Kc=Kdim*1e5/(R*TT)
    drive=1.0-(Ch2**2)/(Cch4.clamp_min(1e-12)*Kc)
    rn=KIN.k_non_0*torch.exp(-KIN.Ea_non/(R*TT))*Cch4**KIN.n*drive
    rc=KIN.k_cat_0*torch.exp(-KIN.Ea_cat/(R*TT))*Cch4*drive*a_s
    ra=rn+rc
    src=torch.stack([-a*ra* M_CH4, 2*a*ra*M_H2, torch.zeros_like(ra)],dim=1)
    # derivatives wrt physical h
    dJ=[]
    for k in range(3):
        dJ.append(torch.autograd.grad(J[:,k].sum(),xi,create_graph=True)[0][:,0]/H)
    dP=torch.autograd.grad(P.sum(),xi,create_graph=True)[0][:,0]/H
    dJ=torch.stack(dJ,dim=1)
    dP_model=-(rhoL_t*(1-a)+rhoG*a)*G_t
    # scaled residuals
    rJ=(dJ-src)/(JT_IN/H)
    rP=(dP-dP_model)/(P_IN/H)
    return rJ,rP,dict(J=J,P=P,x=x,jt=jt,rhoG=rhoG,ug=ug,a=a,db=db,a_s=a_s,r_A=ra)

def train(seed=7,epochs=800,nc=64):
    torch.manual_seed(seed); np.random.seed(seed)
    model=PINN(32,2).to(dtype=DTYPE)
    opt=torch.optim.Adam(model.parameters(),lr=2e-3)
    # fixed collocation points + extra inlet/outlet-focused points
    u=np.linspace(0,1,nc)[:,None]
    xi=torch.tensor(u,dtype=DTYPE)
    history=[]
    for ep in range(epochs):
        opt.zero_grad()
        rJ,rP,meta=residuals(model,xi)
        loss_phys=(rJ**2).mean()+rP.pow(2).mean()
        # inlet BCs
        xin=torch.zeros((1,1),dtype=DTYPE)
        Jin,Pin,xin_mass,jtin=model(xin)
        targetJ=torch.tensor([[J_CH4_IN,J_H2_IN,J_AR_IN]],dtype=DTYPE)
        bcJ=((Jin-targetJ)/JT_IN).pow(2).mean()
        bcP=((Pin-P_IN)/P_IN).pow(2).mean()
        loss=loss_phys+10*bcJ+10*bcP
        # mild positivity/simplex are already enforced by softmax/exp
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
        if ep%100==0 or ep==epochs-1:
            history.append({'epoch':ep,'loss':float(loss),'physics':float(loss_phys),'bc':float(bcJ+bcP)})
    # L-BFGS polishing
    lbfgs=torch.optim.LBFGS(model.parameters(),lr=0.4,max_iter=40,tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    def closure():
        lbfgs.zero_grad(); rJ,rP,_=residuals(model,xi); xin=torch.zeros((1,1),dtype=DTYPE); Jin,Pin,_,_=model(xin); targetJ=torch.tensor([[J_CH4_IN,J_H2_IN,J_AR_IN]],dtype=DTYPE); loss=(rJ**2).mean()+rP.pow(2).mean()+10*((Jin-targetJ)/JT_IN).pow(2).mean()+10*((Pin-P_IN)/P_IN).pow(2).mean(); loss.backward(); return loss
    lbfgs.step(closure)
    return model,history

if __name__=='__main__':
    out=Path('results/stage8_pinn_reaction_only'); out.mkdir(parents=True,exist_ok=True)
    model,history=train()
    grid=np.linspace(0,1,300)[:,None]; xi=torch.tensor(grid,dtype=DTYPE)
    with torch.no_grad(): J,P,x,jt=model(xi)
    X=100*(1-J[:,0].numpy()/J_CH4_IN)
    data=np.column_stack([grid[:,0]*H,J[:,0].numpy(),J[:,1].numpy(),J[:,2].numpy(),P.numpy(),X,x.numpy()])
    np.savetxt(out/'pinn_profile.csv',data,delimiter=',',header='h_m,J_CH4,J_H2,J_Ar,P_Pa,X_CH4_pct,x_CH4,x_H2,x_Ar',comments='')
    ref=solve_reaction_only(type('C',(),dict(D=D,H=H,P_in=P_IN,T_L_in=T,j_CH4_in=J_CH4_IN,j_H2_in=0.,j_Ar_in=J_AR_IN))(),props,thermo,n_eval=500)
    from scipy.interpolate import interp1d
    Xref=interp1d(ref.h,ref.X_ch4)(grid[:,0]*H)
    err=np.abs(X-Xref)
    np.savetxt(out/'pinn_vs_reference.csv',np.column_stack([grid[:,0]*H,X,Xref,err]),delimiter=',',header='h_m,X_PINN_pct,X_reference_pct,abs_error_pct',comments='')
    np.savez(out/'model_state.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    (out/'training_history.json').write_text(json.dumps(history,indent=2))
    print('PINN X_out %',X[-1],'reference X_out %',ref.X_out,'MAE pp',err.mean(),'max pp',err.max(),'final loss',history[-1]['loss'])
