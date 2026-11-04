from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from parameters import BenchmarkCase, G, M_CH4, M_H2, M_AR, R, KineticParameters
from properties import Le2023Properties
from thermodynamics import Le2023Thermo
from heat_transfer_bench import solve_bench_heat_transfer
from pinn_heat_transfer import gas_cp_mass_torch, gas_mu_torch, gas_k_torch, thermo_K_torch, rhoL, muL, sigmaL, D, H, P_IN, T_IN, T_L, J_CH4_IN, J_H2_IN, J_AR_IN, J_TOTAL_IN, MASS_MW_T

DTYPE=torch.float64
torch.set_default_dtype(DTYPE)
CASE=BenchmarkCase(); PROPS=Le2023Properties(); THERMO=Le2023Thermo(); KIN=KineticParameters()
CP_REF=float(PROPS.cp_gas(T_IN,{"CH4":0.8,"H2":0.0,"Ar":0.2}))
ENERGY_SCALE=J_TOTAL_IN*CP_REF*max(T_L-T_IN,1.0)/H
J_SCALE=torch.tensor([J_CH4_IN, max((2*M_H2/M_CH4)*J_CH4_IN,1e-12), J_AR_IN],dtype=DTYPE)

class HybridPINN(nn.Module):
    def __init__(self,width=64,depth=3):
        super().__init__()
        layers=[nn.Linear(1,width),nn.Tanh()]
        for _ in range(depth-1): layers += [nn.Linear(width,width),nn.Tanh()]
        layers += [nn.Linear(width,5),nn.Sigmoid()]
        self.net=nn.Sequential(*layers).to(dtype=DTYPE)
    def forward(self,z):
        o=self.net(z)
        J=o[:,:3]*J_SCALE
        P=o[:,3]*P_IN
        T=T_IN + o[:,4]*(T_L-T_IN)
        x=J/J.sum(dim=1,keepdim=True).clamp_min(1e-30)
        jt=J.sum(dim=1)
        return J,P,T,x,jt


def hydrodynamics(P,T,x,jt):
    Mmix=1.0/(x/MASS_MW_T).sum(dim=1)
    rhoG=P*Mmix/(R*T); ug=jt/rhoG
    rhoLt=torch.full_like(rhoG,rhoL); muLt=torch.full_like(rhoG,muL); sigt=torch.full_like(rhoG,sigmaL)
    Uchar=(sigt*G*(rhoLt-rhoG).clamp_min(1e-20)/rhoLt**2).clamp_min(1e-30)**0.25
    uplus=ug/Uchar
    Nm=muLt/torch.sqrt(rhoLt*sigt*torch.sqrt(sigt/(G*(rhoLt-rhoG).clamp_min(1e-20))))
    DH=D/torch.sqrt(sigt/(G*(rhoLt-rhoG).clamp_min(1e-20)))
    rr=rhoLt/rhoG
    v1_low=0.0019*DH**0.809*rr**(-0.157)*Nm**(-0.562)
    v1_high=0.92*rr**(-0.157)*Nm**(-0.562)
    v1_no=0.92*rr**(-0.157)
    v1=torch.where(Nm>=2.2e-3,torch.where(DH>=30,v1_no,v1_low),torch.where(DH>=30,v1_high,v1_low))
    e=torch.exp(-1.39*uplus); vp=torch.sqrt(torch.tensor(2.0,dtype=DTYPE))*e+v1*(1-e)
    C0=1.2-0.2*torch.sqrt((rhoG/rhoLt).clamp_min(1e-30)); ag=(uplus/(uplus*C0+vp).clamp_min(1e-30)).clamp(1e-8,.999)
    NBo=G*D**2*rhoLt/sigt; nuL=muLt/rhoLt; NGa=G*D**3/nuL**2; NFr=ug/torch.sqrt(torch.tensor(G*D,dtype=DTYPE))
    db=26*NBo**(-.50)*NGa**(-.12)*NFr**(-.12)*D; a_s=6/db
    return rhoG,ug,ag,db,a_s


def residuals(model,z):
    z=z.clone().detach().requires_grad_(True)
    J,P,T,x,jt=model(z)
    rhoG,ug,ag,db,a_s=hydrodynamics(P,T,x,jt)
    muG=gas_mu_torch(T,x); kG=gas_k_torch(T,x); cpG=gas_cp_mass_torch(T,x); ub=ug/ag
    NPe=rhoG*cpG*db*ub/(ag*kG.clamp_min(1e-12)); Nu=1.1*(NPe/(1+muG/muL)).clamp_min(1e-12)**.7; Uh=Nu*kG/db
    C1=rhoG*x[:,0]/M_CH4; C2=rhoG*x[:,1]/M_H2; Kc=thermo_K_torch(T)*1e5/(R*T); drive=1-(C2**2)/(C1.clamp_min(1e-12)*Kc)
    rn=KIN.k_non_0*torch.exp(-KIN.Ea_non/(R*T))*C1**KIN.n*drive; rc=KIN.k_cat_0*torch.exp(-KIN.Ea_cat/(R*T))*C1*drive*a_s; ra=rn+rc
    src=torch.stack([-ag*ra*M_CH4,2*ag*ra*M_H2,torch.zeros_like(ra)],dim=1)
    dJ=torch.stack([torch.autograd.grad(J[:,i].sum(),z,create_graph=True)[0][:,0] for i in range(3)],dim=1)/H
    dP=torch.autograd.grad(P.sum(),z,create_graph=True)[0][:,0]/H
    Eflux=jt*cpG*T; dE=torch.autograd.grad(Eflux.sum(),z,create_graph=True)[0][:,0]/H
    dPmodel=-(rhoL*(1-ag)+rhoG*ag)*G; gas_src=-Uh*a_s*(T-T_L)-ag*(2*37.4e3)*ra
    rJ=(dJ-src)/(J_TOTAL_IN/H); rP=(dP-dPmodel)/(P_IN/H); rE=(dE-gas_src)/ENERGY_SCALE
    return rJ,rP,rE


def train(seed=13,nc=128,pretrain=1000,physics=1000):
    torch.manual_seed(seed); np.random.seed(seed)
    model=HybridPINN();
    with torch.no_grad():
        model.net[0].weight.zero_() if isinstance(model.net[0], nn.Linear) else None
        final=[m for m in model.net if isinstance(m, nn.Linear)][-1]
        final.weight.mul_(0.02); final.bias.copy_(torch.tensor([7.0,-7.0,7.0,7.0,-7.0],dtype=DTYPE))
    ref=solve_bench_heat_transfer(CASE,PROPS,THERMO,n_eval=nc)
    zref=torch.tensor((ref.h/H)[:,None],dtype=DTYPE)
    Jref=torch.tensor(np.column_stack([ref.J_ch4,ref.J_h2,ref.J_ar]),dtype=DTYPE); Pref=torch.tensor(ref.P,dtype=DTYPE); Tref=torch.tensor(ref.T_g,dtype=DTYPE)
    z0=torch.zeros((1,1),dtype=DTYPE); J0ref=Jref[:1]; P0ref=Pref[:1]; T0ref=Tref[:1]
    opt=torch.optim.Adam(model.parameters(),lr=2e-3); hist=[]
    for ep in range(pretrain):
        opt.zero_grad(); J,P,T,_,_=model(zref)
        data=((J-Jref)/J_TOTAL_IN).pow(2).mean()+((P-Pref)/P_IN).pow(2).mean()+((T-Tref)/(T_L-T_IN)).pow(2).mean(); J0,P0,T0,_,_=model(z0); bc=((J0-J0ref)/J_TOTAL_IN).pow(2).mean()+((P0-P0ref)/P_IN).pow(2).mean()+((T0-T0ref)/(T_L-T_IN)).pow(2).mean(); ld=data+1000*bc
        ld.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10); opt.step()
        if ep%100==0 or ep==pretrain-1: hist.append({'phase':'pretrain','epoch':ep,'loss':float(ld)})
    z=torch.linspace(0,1,nc,dtype=DTYPE)[:,None]; opt=torch.optim.Adam(model.parameters(),lr=4e-4)
    for ep in range(physics):
        opt.zero_grad(); rJ,rP,rE=residuals(model,z); lp=(rJ**2).mean()+0.5*(rP**2).mean()+0.5*(rE**2).mean(); J,P,T,_,_=model(z)
        data=((J-Jref)/J_TOTAL_IN).pow(2).mean()+((P-Pref)/P_IN).pow(2).mean()+((T-Tref)/(T_L-T_IN)).pow(2).mean(); J0,P0,T0,_,_=model(z0); bc=((J0-J0ref)/J_TOTAL_IN).pow(2).mean()+((P0-P0ref)/P_IN).pow(2).mean()+((T0-T0ref)/(T_L-T_IN)).pow(2).mean(); ld=data+1000*bc; loss=lp+2.0*ld
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10); opt.step()
        if ep%100==0 or ep==physics-1: hist.append({'phase':'physics','epoch':ep,'loss':float(loss),'physics':float(lp),'data':float(ld)})
    return model,hist,ref


def main():
    out=Path('results/stage9_pinn_heat_transfer_hybrid'); out.mkdir(parents=True,exist_ok=True)
    model,hist,ref=train()
    z=torch.linspace(0,1,301,dtype=DTYPE)[:,None];
    with torch.no_grad(): J,P,T,x,jt=model(z)
    h=z[:,0].numpy()*H; X=100*(1-J[:,0].numpy()/J_CH4_IN); Tc=T.numpy()-273.15
    Xr=interp1d(ref.h,ref.X_ch4)(h); Tr=interp1d(ref.h,ref.T_g)(h)-273.15; Pr=interp1d(ref.h,ref.P)(h)
    eX=np.abs(X-Xr); eT=np.abs(Tc-Tr); eP=np.abs(P.numpy()-Pr)/1e5
    np.savetxt(out/'pinn_profile.csv',np.column_stack([h,J.numpy(),P.numpy(),Tc,X,x.numpy()]),delimiter=',',header='h_m,J_CH4,J_H2,J_Ar,P_Pa,T_G_C,X_CH4_pct,x_CH4,x_H2,x_Ar',comments='')
    np.savetxt(out/'pinn_vs_reference.csv',np.column_stack([h,X,Xr,eX,Tc,Tr,eT,P.numpy()/1e5,Pr/1e5]),delimiter=',',header='h_m,X_PINN_pct,X_reference_pct,abs_X_error_pp,T_PINN_C,T_reference_C,abs_T_error_C,PINN_bar,reference_bar',comments='')
    np.savez(out/'model_state.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    (out/'training_history.json').write_text(json.dumps(hist,indent=2))
    summary={'pinn_X_out_pct':float(X[-1]),'reference_X_out_pct':float(Xr[-1]),'X_MAE_pp':float(eX.mean()),'X_max_error_pp':float(eX.max()),'T_MAE_C':float(eT.mean()),'T_max_error_C':float(eT.max()),'P_MAE_bar':float(eP.mean()),'final_loss':float(hist[-1]['loss'])}
    (out/'STAGE9_REPORT.md').write_text('# Stage 9 — Hybrid PINN heat-transfer model\n\n'+json.dumps(summary,indent=2)+'\n')
    plt.figure(figsize=(7,5)); plt.plot(h,Xr,label='Reference'); plt.plot(h,X,'--',label='Hybrid PINN'); plt.xlabel('h (m)'); plt.ylabel('CH4 conversion (%)'); plt.title('Hybrid PINN vs reference'); plt.legend(); plt.tight_layout(); plt.savefig(out/'conversion_comparison.png',dpi=180); plt.close()
    plt.figure(figsize=(7,5)); plt.plot(h,Tr,label='Reference'); plt.plot(h,Tc,'--',label='Hybrid PINN'); plt.xlabel('h (m)'); plt.ylabel('Gas temperature (C)'); plt.title('Gas temperature: hybrid PINN vs reference'); plt.legend(); plt.tight_layout(); plt.savefig(out/'temperature_comparison.png',dpi=180); plt.close()
    plt.figure(figsize=(7,5)); plt.plot([d['epoch'] for d in hist if d['phase']=='pretrain'],[d['loss'] for d in hist if d['phase']=='pretrain'],label='pretrain'); p=[d for d in hist if d['phase']=='physics']; plt.plot([d['epoch']+1000 for d in p],[d['loss'] for d in p],label='physics'); plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('loss'); plt.title('Hybrid PINN training'); plt.legend(); plt.tight_layout(); plt.savefig(out/'training_loss.png',dpi=180); plt.close()
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
