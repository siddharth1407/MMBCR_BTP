from __future__ import annotations
import json, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from parameters import BenchmarkCase, KineticParameters, R, M_CH4, M_H2, M_AR
from properties import Le2023Properties
from thermodynamics import Le2023Thermo
from model_context import ModelContext
from heat_transfer_bench import solve_bench_heat_transfer
from pinn_heat_transfer_hybrid import HybridPINN as HeatNet, residuals as heat_residuals, CASE as HCASE, PROPS as HPROPS, THERMO as HTHERMO, H as HH, J_TOTAL_IN as HJT, P_IN as HPIN, T_L as HTL, T_IN as HTIN
from frozen_danckwerts import solve as solve_frozen

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'stage14_ai_final'
OUT.mkdir(parents=True,exist_ok=True)
DTYPE=torch.float64
torch.set_default_dtype(DTYPE)

# ---------- Common helpers ----------
def set_seeds(seed:int):
    torch.manual_seed(seed); np.random.seed(seed)

def metrics(pred, ref):
    e=np.abs(pred-ref)
    return {'mae':float(e.mean()),'rmse':float(np.sqrt(np.mean((pred-ref)**2))),'max':float(e.max()),'out':float(pred[-1]),'ref_out':float(ref[-1])}

# ---------- Stage 9: leak-free hybrid heat-transfer AI ----------
def train_heat(seed=202, n_ref=401, n_train=80, physics_epochs=2600):
    set_seeds(seed)
    model=HeatNet()
    ref=solve_bench_heat_transfer(HCASE,HPROPS,HTHERMO,n_eval=n_ref)
    zref=torch.tensor((ref.h/HH)[:,None],dtype=DTYPE)
    Jref=torch.tensor(np.c_[ref.J_ch4,ref.J_h2,ref.J_ar],dtype=DTYPE)
    Pref=torch.tensor(ref.P,dtype=DTYPE); Tref=torch.tensor(ref.T_g,dtype=DTYPE)
    # deterministic stratified training subset including both boundaries
    idx=np.unique(np.linspace(0,n_ref-1,n_train).round().astype(int)); train_mask=np.zeros(n_ref,dtype=bool); train_mask[idx]=True
    ztr=zref[train_mask]; Jtr=Jref[train_mask]; Ptr=Pref[train_mask]; Ttr=Tref[train_mask]
    z0=torch.zeros((1,1),dtype=DTYPE)
    opt=torch.optim.Adam(model.parameters(),lr=2e-3)
    hist=[]
    for ep in range(physics_epochs):
        opt.zero_grad()
        J,P,T,_,_=model(ztr)
        data=((J-Jtr)/HJT).pow(2).mean()+((P-Ptr)/HPIN).pow(2).mean()+((T-Ttr)/(HTL-HTIN)).pow(2).mean()
        # physics collocation across full domain; residual audited in bulk later
        zc=torch.linspace(0,1,128,dtype=DTYPE)[:,None]
        rJ,rP,rE=heat_residuals(model,zc)
        lp=(rJ**2).mean()+0.5*(rP**2).mean()+0.5*(rE**2).mean()
        J0,P0,T0,_,_=model(z0)
        bc=((J0-Jref[:1])/HJT).pow(2).mean()+((P0-Pref[:1])/HPIN).pow(2).mean()+((T0-Tref[:1])/(HTL-HTIN)).pow(2).mean()
        loss=2.0*data+lp+1000.0*bc
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
        if ep%200==0 or ep==physics_epochs-1:
            hist.append({'epoch':ep,'loss':float(loss.detach()),'data':float(data.detach()),'physics':float(lp.detach()),'bc':float(bc.detach())})
    # evaluate on all spatial points, but train/validation is honest: only train points entered supervised loss
    zall=torch.tensor((ref.h/HH)[:,None],dtype=DTYPE)
    with torch.no_grad(): J,P,T,x,jt=model(zall)
    X=100*(1-J[:,0].numpy()/ref.J_ch4[0]); Xr=ref.X_ch4
    Tc=T.numpy()-273.15; Tr=ref.T_g-273.15; Pb=P.numpy()/1e5; Pr=ref.P/1e5
    val=~train_mask
    # residual audit in bulk only (exclude sharp inlet thermal layer)
    zphys=torch.cat([torch.linspace(0.02,0.20,96,dtype=DTYPE),torch.linspace(0.20,1.0,96,dtype=DTYPE)])[:,None]
    rJ,rP,rE=heat_residuals(model,zphys)
    summary={
      'model':'Leak-free hybrid heat-transfer neural surrogate',
      'supervised_train_points':int(train_mask.sum()),'held_out_validation_points':int(val.sum()),
      'reference_X_out_pct':float(Xr[-1]),'ai_X_out_pct':float(X[-1]),
      'validation_conversion':metrics(X[val],Xr[val]),
      'all_domain_conversion':metrics(X,Xr),
      'validation_temperature_C':metrics(Tc[val],Tr[val]),
      'validation_pressure_bar':metrics(Pb[val],Pr[val]),
      'bulk_physics_rms':{'mass':float(torch.sqrt((rJ**2).mean())),'pressure':float(torch.sqrt((rP**2).mean())),'energy':float(torch.sqrt((rE**2).mean()))},
      'note':'Reference data are used only at the supervised training locations; the remaining spatial points are held out for evaluation. Governing-equation residuals are imposed over the full collocation domain.'
    }
    d=pd.DataFrame({
      'h_m':ref.h,'split':np.where(train_mask,'train','validation'),
      'X_AI_pct':X,'X_reference_pct':Xr,'abs_X_error_pp':np.abs(X-Xr),
      'T_AI_C':Tc,'T_reference_C':Tr,'abs_T_error_C':np.abs(Tc-Tr),
      'P_AI_bar':Pb,'P_reference_bar':Pr,'abs_P_error_bar':np.abs(Pb-Pr)
    })
    d.to_csv(OUT/'heat_transfer_ai_validation.csv',index=False)
    np.savez(OUT/'heat_transfer_model.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    (OUT/'heat_transfer_history.json').write_text(json.dumps(hist,indent=2))
    (OUT/'heat_transfer_summary.json').write_text(json.dumps(summary,indent=2))
    return summary

# ---------- Stage 10: leak-free axial-dispersion AI ----------
class AxialNet(nn.Module):
    def __init__(self,width=64,depth=3):
        super().__init__(); layers=[nn.Linear(1,width),nn.Tanh()]
        for _ in range(depth-1): layers += [nn.Linear(width,width),nn.Tanh()]
        layers += [nn.Linear(width,2)]; self.net=nn.Sequential(*layers).to(dtype=DTYPE)
    def forward(self,z):
        o=self.net(z)
        J1=J1_IN*torch.sigmoid(o[:,0:1]); J2=J2_SCALE*torch.sigmoid(o[:,1:2]); J3=torch.full_like(J1,JAR)
        J=torch.cat([J1,J2,J3],dim=1); jt=J.sum(dim=1,keepdim=True); x=J/jt.clamp_min(1e-20)
        return J,x

CASE=BenchmarkCase(); CTX=ModelContext(properties=Le2023Properties(),thermo=Le2023Thermo(),kinetics=KineticParameters())
H=CASE.H; J1_IN=CASE.j_CH4_in; J2_SCALE=(2*M_H2/M_CH4)*J1_IN; JAR=CASE.j_Ar_in

def reaction_rate(T,rho,x1,x2,as_):
    K=torch.exp(13.2714-91204.6/(R*T)); Kc=K*1e5/(R*T)
    C1=rho*x1/M_CH4; C2=rho*x2/M_H2
    drive=1-(C2**2)/(C1.clamp_min(1e-20)*Kc.clamp_min(1e-20))
    rn=CTX.kinetics.k_non_0*torch.exp(-CTX.kinetics.Ea_non/(R*T))*C1**CTX.kinetics.n*drive
    rc=CTX.kinetics.k_cat_0*torch.exp(-CTX.kinetics.Ea_cat/(R*T))*C1*drive*as_
    return rn+rc

def fd(v,dh):
    return torch.cat([(v[1:2]-v[0:1])/dh,(v[2:]-v[:-2])/(2*dh),(v[-1:]-v[-2:-1])/dh],dim=0)

def train_axial(seed=303,n_eval=220,n_train=32,epochs=3200):
    set_seeds(seed)
    ref,res=solve_frozen(CASE,CTX,n_eval=n_eval,tol=1e-4,max_nodes=12000,phi_th=.001,dax_floor=1e-10)
    if ref is None: raise RuntimeError(f'Frozen axial reference failed: {res.message}')
    heat=solve_bench_heat_transfer(CASE,CTX.properties,CTX.thermo,n_eval=500)
    h=ref.h; P=np.interp(h,heat.h,heat.P); T=np.interp(h,heat.h,heat.T_g); a=np.interp(h,heat.h,heat.a_g); ass=np.interp(h,heat.h,heat.a_s); Dax=ref.D_ax
    fields={k:torch.tensor(v,dtype=DTYPE) for k,v in {'P':P,'T':T,'a':a,'as':ass,'D':Dax}.items()}
    z=torch.tensor((h/H)[:,None],dtype=DTYPE)
    Jref1=torch.tensor(ref.J_ch4,dtype=DTYPE)[:,None]; Jref2=torch.tensor(ref.J_h2,dtype=DTYPE)[:,None]
    idx=np.unique(np.linspace(0,n_eval-1,n_train).round().astype(int)); train_mask=np.zeros(n_eval,dtype=bool); train_mask[idx]=True
    model=AxialNet(); opt=torch.optim.Adam(model.parameters(),lr=2e-3); hist=[]
    def physics():
        J,x=model(z); x1=x[:,0]; x2=x[:,1]
        Mmix=1/(x1/M_CH4+x2/M_H2+(1-x1-x2)/M_AR); rho=fields['P']*Mmix/(R*fields['T']); Kd=rho*fields['a']*fields['D']
        ra=reaction_rate(fields['T'],rho,x1,x2,fields['as']); S1=-fields['a']*ra*M_CH4; S2=2*fields['a']*ra*M_H2
        dh=H/(n_eval-1); dJ1=fd(J[:,0],dh); dJ2=fd(J[:,1],dh); dx1=fd(x1,dh); dx2=fd(x2,dh)
        r1=(dJ1-fd(Kd*dx1,dh)-S1)/(J1_IN/H); r2=(dJ2-fd(Kd*dx2,dh)-S2)/(J2_SCALE/H)
        bc=(dx1[-1]**2+dx2[-1]**2)/(1/H)**2
        return r1,r2,bc
    for ep in range(epochs):
        opt.zero_grad(); J,x=model(z)
        sparse=((J[train_mask,0:1]-Jref1[train_mask])/J1_IN).pow(2).mean()+((J[train_mask,1:2]-Jref2[train_mask])/J2_SCALE).pow(2).mean()
        r1,r2,bc=physics(); lp=(r1**2).mean()+0.5*(r2**2).mean(); loss=4*sparse+0.25*lp+5*bc
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),20); opt.step()
        if ep%250==0 or ep==epochs-1: hist.append({'epoch':ep,'loss':float(loss.detach()),'sparse':float(sparse.detach()),'physics':float(lp.detach()),'bc':float(bc.detach())})
    with torch.no_grad(): J,x=model(z)
    Jn=J.numpy(); X=100*(1-Jn[:,0]/J1_IN); Xr=ref.X_ch4; err=np.abs(X-Xr); val=~train_mask
    r1,r2,bc=physics()
    summary={
      'model':'Leak-free hybrid physics-informed axial-dispersion surrogate',
      'supervised_train_points':int(train_mask.sum()),'held_out_validation_points':int(val.sum()),
      'reference_X_out_pct':float(Xr[-1]),'ai_X_out_pct':float(X[-1]),
      'validation_conversion':metrics(X[val],Xr[val]),'all_domain_conversion':metrics(X,Xr),
      'bulk_physics_rms':{'CH4':float(torch.sqrt((r1**2).mean())),'H2':float(torch.sqrt((r2**2).mean()))},
      'note':'Only sparse training locations are used in supervised loss. Remaining spatial points are held out. Physics residual and outlet-gradient boundary condition act across the full domain.'
    }
    d=pd.DataFrame({'h_m':h,'split':np.where(train_mask,'train','validation'),'X_AI_pct':X,'X_reference_pct':Xr,'abs_X_error_pp':err,'J_CH4_AI':Jn[:,0],'J_H2_AI':Jn[:,1],'J_CH4_ref':Jref1.numpy()[:,0],'J_H2_ref':Jref2.numpy()[:,0]})
    d.to_csv(OUT/'axial_dispersion_ai_validation.csv',index=False)
    np.savez(OUT/'axial_model.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}); (OUT/'axial_history.json').write_text(json.dumps(hist,indent=2)); (OUT/'axial_summary.json').write_text(json.dumps(summary,indent=2))
    return summary

# ---------- Consolidate ----------
def main():
    t0=time.time()
    hs=train_heat(); print('HEAT',json.dumps(hs,indent=2))
    ax=train_axial(); print('AXIAL',json.dumps(ax,indent=2))
    pure={
      'model':'Pure physics-informed reaction-only PINN',
      'reference_X_out_pct':90.42916362683866,'ai_X_out_pct':90.10971818112878,
      'conversion_MAE_pp':0.40342467233328566,'status':'accepted_existing_benchmark',
      'training_data':'none; physics residual + inlet conditions'
    }
    rows=[
      {'model':'Pure PINN','scope':'reaction-only','reference_X_out_pct':pure['reference_X_out_pct'],'AI_X_out_pct':pure['ai_X_out_pct'],'MAE_pp':pure['conversion_MAE_pp'],'evaluation':'reference comparison','status':'accepted'},
      {'model':'Hybrid neural surrogate','scope':'heat transfer','reference_X_out_pct':hs['reference_X_out_pct'],'AI_X_out_pct':hs['ai_X_out_pct'],'MAE_pp':hs['validation_conversion']['mae'],'evaluation':'held-out spatial validation','status':'accepted'},
      {'model':'Hybrid physics-informed surrogate','scope':'axial-dispersion mass BVP','reference_X_out_pct':ax['reference_X_out_pct'],'AI_X_out_pct':ax['ai_X_out_pct'],'MAE_pp':ax['validation_conversion']['mae'],'evaluation':'held-out spatial validation','status':'accepted'},
    ]
    df=pd.DataFrame(rows); df.to_csv(OUT/'final_ai_model_comparison.csv',index=False)
    summary={'status':'AI TECHNICAL FREEZE CANDIDATE','models':rows,'total_runtime_s':time.time()-t0,'important_change':'Stage 14 removes supervised-data leakage from the Stage 9/10 hybrid surrogate evaluations by training only on designated spatial points and evaluating on held-out spatial points.'}
    (OUT/'AI_FINAL_REGISTER.md').write_text('# Stage 14 — Final AI technical layer\n\n'+json.dumps(summary,indent=2)+'\n')
    # plots
    plt.figure(figsize=(8,5)); names=[r['scope'] for r in rows]; refv=[r['reference_X_out_pct'] for r in rows]; aiv=[r['AI_X_out_pct'] for r in rows]; xx=np.arange(len(rows)); w=.36; plt.bar(xx-w/2,refv,w,label='Reference'); plt.bar(xx+w/2,aiv,w,label='AI'); plt.xticks(xx,names,rotation=15,ha='right'); plt.ylabel('CH4 conversion at outlet (%)'); plt.title('Final AI model comparison'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'final_ai_outlet_comparison.png',dpi=180); plt.close()
    plt.figure(figsize=(8,5)); plt.bar(xx,[r['MAE_pp'] for r in rows]); plt.xticks(xx,names,rotation=15,ha='right'); plt.ylabel('Conversion MAE (percentage points)'); plt.title('Final AI validation error'); plt.tight_layout(); plt.savefig(OUT/'final_ai_error_comparison.png',dpi=180); plt.close()
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
