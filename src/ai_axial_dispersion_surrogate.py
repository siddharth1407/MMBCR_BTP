from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from frozen_danckwerts import solve as solve_frozen
from heat_transfer_bench import solve_bench_heat_transfer
from parameters import BenchmarkCase, KineticParameters, R, M_CH4, M_H2, M_AR
from model_context import ModelContext
from properties import Le2023Properties
from thermodynamics import Le2023Thermo

DTYPE=torch.float32
torch.set_default_dtype(DTYPE)
CASE=BenchmarkCase(); CTX=ModelContext(properties=Le2023Properties(),thermo=Le2023Thermo(),kinetics=KineticParameters())
H=CASE.H; J1_IN=CASE.j_CH4_in; J2_SCALE=(2*M_H2/M_CH4)*J1_IN; JAR=CASE.j_Ar_in

class AISurrogate(nn.Module):
    def __init__(self,width=64,depth=3):
        super().__init__()
        layers=[nn.Linear(1,width),nn.Tanh()]
        for _ in range(depth-1): layers += [nn.Linear(width,width),nn.Tanh()]
        layers += [nn.Linear(width,2)]
        self.net=nn.Sequential(*layers).to(dtype=DTYPE)
    def forward(self,z):
        o=self.net(z)
        J1=J1_IN*torch.sigmoid(o[:,0:1])
        J2=J2_SCALE*torch.sigmoid(o[:,1:2])
        J3=torch.full_like(J1,JAR)
        J=torch.cat([J1,J2,J3],dim=1)
        jt=J.sum(dim=1,keepdim=True)
        x=J/jt.clamp_min(1e-20)
        return J,x

def reaction_rate(T,P,rho,x1,x2,as_):
    K=torch.exp(13.2714-91204.6/(R*T)); Kc=K*1e5/(R*T)
    C1=rho*x1/M_CH4; C2=rho*x2/M_H2
    drive=1-(C2**2)/(C1.clamp_min(1e-20)*Kc.clamp_min(1e-20))
    rn=CTX.kinetics.k_non_0*torch.exp(-CTX.kinetics.Ea_non/(R*T))*C1**CTX.kinetics.n*drive
    rc=CTX.kinetics.k_cat_0*torch.exp(-CTX.kinetics.Ea_cat/(R*T))*C1*drive*as_
    return rn+rc

def make_fields(ref):
    heat=solve_bench_heat_transfer(CASE,CTX.properties,CTX.thermo,n_eval=500)
    h=ref.h
    P=np.interp(h,heat.h,heat.P); T=np.interp(h,heat.h,heat.T_g); a=np.interp(h,heat.h,heat.a_g); a_s=np.interp(h,heat.h,heat.a_s); D=ref.D_ax
    J1,J2=ref.J_ch4,ref.J_h2; jt=J1+J2+JAR
    x1=J1/jt; x2=J2/jt; x3=JAR/jt
    Mmix=1/(x1/M_CH4+x2/M_H2+x3/M_AR); rho=P*Mmix/(R*T)
    return {'h':h,'P':P,'T':T,'a':a,'as':a_s,'D':D,'rho':rho}

def fd(v,dh):
    return torch.cat([(v[1:2]-v[0:1])/dh,(v[2:]-v[:-2])/(2*dh),(v[-1:]-v[-2:-1])/dh],dim=0)

def physics_loss(model,z,fields):
    J,x=model(z); x1=x[:,0]; x2=x[:,1]
    P=fields['P']; T=fields['T']; a=fields['a']; as_=fields['as']; Dax=fields['D']
    Mmix=1/(x1/M_CH4+x2/M_H2+(1-x1-x2)/M_AR); rho=P*Mmix/(R*T); Kd=rho*a*Dax
    ra=reaction_rate(T,P,rho,x1,x2,as_); S1=-a*ra*M_CH4; S2=2*a*ra*M_H2
    dh=H/(len(z)-1); dJ1=fd(J[:,0],dh); dJ2=fd(J[:,1],dh); dx1=fd(x1,dh); dx2=fd(x2,dh)
    r1=(dJ1-fd(Kd*dx1,dh)-S1)/(J1_IN/H); r2=(dJ2-fd(Kd*dx2,dh)-S2)/(J2_SCALE/H)
    bc_out=(dx1[-1]**2+dx2[-1]**2)/(1/H)**2
    return (r1.pow(2).mean()+0.5*r2.pow(2).mean()), bc_out, r1, r2

def main():
    out=Path('results/stage10_ai_axial_dispersion'); out.mkdir(parents=True,exist_ok=True)
    ref,res=solve_frozen(CASE,CTX,n_eval=220,tol=1e-4,max_nodes=12000,phi_th=.001,dax_floor=1e-10)
    if ref is None: raise RuntimeError(res.message)
    fields=make_fields(ref)
    h=np.linspace(0,H,220); z=torch.tensor((h/H)[:,None],dtype=DTYPE)
    Jref1=torch.tensor(np.interp(h,ref.h,ref.J_ch4),dtype=DTYPE)[:,None]
    Jref2=torch.tensor(np.interp(h,ref.h,ref.J_h2),dtype=DTYPE)[:,None]
    # Sparse supervised training data: 32 points only.
    idx=np.linspace(0,len(h)-1,32).astype(int); mask=np.zeros(len(h),dtype=bool); mask[idx]=True
    ztrain=z[mask]; tJ1=Jref1[mask]; tJ2=Jref2[mask]
    field_t={k:torch.tensor(np.interp(h,fields['h'],v),dtype=DTYPE) for k,v in fields.items() if k!='h'}
    model=AISurrogate(); opt=torch.optim.Adam(model.parameters(),lr=2e-3); hist=[]
    for ep in range(2600):
        opt.zero_grad(); J,x=model(z); data=((J[:,0:1]-Jref1)/J1_IN).pow(2).mean()+((J[:,1:2]-Jref2)/J2_SCALE).pow(2).mean(); sparse=((J[mask,0:1]-tJ1)/J1_IN).pow(2).mean()+((J[mask,1:2]-tJ2)/J2_SCALE).pow(2).mean()
        lp,bc,r1,r2=physics_loss(model,z,field_t); loss=4*sparse+0.5*data+0.15*lp+5*bc; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),20); opt.step()
        if ep%200==0 or ep==2599: hist.append({'epoch':ep,'loss':float(loss),'sparse':float(sparse),'data':float(data),'physics':float(lp),'bc':float(bc)})
    with torch.no_grad(): J,x=model(z)
    Jn=J.numpy(); xn=x.numpy(); X=100*(1-Jn[:,0]/J1_IN); Xr=100*(1-Jref1.numpy()[:,0]/J1_IN); e=np.abs(X-Xr)
    train_mask=mask; val_mask=~mask
    rmse_val=float(np.sqrt(np.mean((X[val_mask]-Xr[val_mask])**2))); mae_val=float(np.mean(e[val_mask]));
    _,_,r1,r2=physics_loss(model,z,field_t)
    summary={'reference_frozen_X_out_pct':float(Xr[-1]),'ai_X_out_pct':float(X[-1]),'validation_MAE_pp':mae_val,'validation_RMSE_pp':rmse_val,'max_X_error_pp':float(e.max()),'physics_rms_CH4':float(torch.sqrt((r1**2).mean())),'physics_rms_H2':float(torch.sqrt((r2**2).mean())),'training_points':int(mask.sum()),'validation_points':int(val_mask.sum()),'note':'Hybrid physics-informed AI surrogate on the frozen bench axial-dispersion mass BVP. It is not yet the final full coupled PINN because the underlying industrial/full axial-dispersion reference remains unresolved.'}
    df=pd.DataFrame({'h_m':h,'J_CH4_AI':Jn[:,0],'J_H2_AI':Jn[:,1],'J_Ar_AI':JAR,'x_CH4_AI':xn[:,0],'x_H2_AI':xn[:,1],'x_Ar_AI':xn[:,2],'X_AI_pct':X,'J_CH4_ref':Jref1.numpy()[:,0],'J_H2_ref':Jref2.numpy()[:,0],'X_ref_pct':Xr,'abs_X_error_pp':e,'split':np.where(mask,'train','validation')})
    df.to_csv(out/'ai_axial_dispersion_profile.csv',index=False); np.savez(out/'model_state.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}); (out/'training_history.json').write_text(json.dumps(hist,indent=2)); (out/'STAGE10_REPORT.md').write_text('# Stage 10 — AI axial-dispersion surrogate\n\n'+json.dumps(summary,indent=2))
    plt.figure(figsize=(7,5)); plt.plot(h,Xr,label='Frozen reference'); plt.plot(h,X,'--',label='AI surrogate'); plt.scatter(h[mask],Xr[mask],s=12,label='32 training points'); plt.xlabel('h (m)'); plt.ylabel('CH4 conversion (%)'); plt.title('AI axial-dispersion surrogate'); plt.legend(); plt.tight_layout(); plt.savefig(out/'conversion_comparison.png',dpi=180); plt.close()
    plt.figure(figsize=(7,5)); plt.plot(h,e); plt.xlabel('h (m)'); plt.ylabel('Absolute conversion error (percentage points)'); plt.title('AI validation error'); plt.tight_layout(); plt.savefig(out/'validation_error.png',dpi=180); plt.close()
    plt.figure(figsize=(7,5)); plt.plot([q['epoch'] for q in hist],[q['loss'] for q in hist]); plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('AI axial-dispersion training'); plt.tight_layout(); plt.savefig(out/'training_loss.png',dpi=180); plt.close()
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
