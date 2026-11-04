from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from parameters import BenchmarkCase
from properties import Le2023Properties
from thermodynamics import Le2023Thermo
from heat_transfer_bench import solve_bench_heat_transfer
from pinn_heat_transfer_hybrid import HybridPINN, residuals, CASE, PROPS, THERMO, H, J_TOTAL_IN, P_IN, T_L, T_IN

DTYPE=torch.float64
torch.set_default_dtype(DTYPE)


def fit_reference(seed=31, n_train=401, adam_epochs=3500):
    torch.manual_seed(seed); np.random.seed(seed)
    model=HybridPINN()
    ref=solve_bench_heat_transfer(CASE,PROPS,THERMO,n_eval=n_train)
    z=torch.tensor((ref.h/H)[:,None],dtype=DTYPE)
    Jref=torch.tensor(np.c_[ref.J_ch4,ref.J_h2,ref.J_ar],dtype=DTYPE)
    Pref=torch.tensor(ref.P,dtype=DTYPE); Tref=torch.tensor(ref.T_g,dtype=DTYPE)
    z0=torch.zeros((1,1),dtype=DTYPE)
    opt=torch.optim.Adam(model.parameters(),lr=2e-3)
    hist=[]
    for ep in range(adam_epochs):
        opt.zero_grad(); J,P,T,_,_=model(z)
        data=((J-Jref)/J_TOTAL_IN).pow(2).mean()+((P-Pref)/P_IN).pow(2).mean()+((T-Tref)/(T_L-T_IN)).pow(2).mean()
        J0,P0,T0,_,_=model(z0)
        bc=((J0-Jref[:1])/J_TOTAL_IN).pow(2).mean()+((P0-Pref[:1])/P_IN).pow(2).mean()+((T0-Tref[:1])/(T_L-T_IN)).pow(2).mean()
        loss=data+100.0*bc
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
        if ep%500==0 or ep==adam_epochs-1:
            hist.append({'epoch':ep,'loss':float(loss.detach()),'data':float(data.detach()),'bc':float(bc.detach())})
    return model,ref,hist


def main():
    out=Path('/mnt/data/MMBCR_reference_solver_work/MMBCR_reference_solver/results/stage9_ai_heat_transfer'); out.mkdir(parents=True,exist_ok=True)
    model,ref,hist=fit_reference()
    z=torch.linspace(0,1,401,dtype=DTYPE)[:,None]
    with torch.no_grad(): J,P,T,x,jt=model(z)
    h=z[:,0].numpy()*H; X=100*(1-J[:,0].numpy()/ref.J_ch4[0]); Tc=T.numpy()-273.15; Xr=ref.X_ch4; Tr=ref.T_g-273.15; Pr=ref.P
    eX=np.abs(X-Xr); eT=np.abs(Tc-Tr); eP=np.abs(P.numpy()-Pr)/1e5
    # Physics residual audit only in the bulk, intentionally excluding the sharp inlet thermal boundary layer.
    zphys=torch.cat([torch.linspace(0.02,0.20,96,dtype=DTYPE),torch.linspace(0.20,1.0,96,dtype=DTYPE)])[:,None]
    rJ,rP,rE=residuals(model,zphys)
    phys={'mass_rmse':float(torch.sqrt((rJ**2).mean()).detach()),'pressure_rmse':float(torch.sqrt((rP**2).mean()).detach()),'energy_rmse':float(torch.sqrt((rE**2).mean()).detach()),'energy_residual_rms_raw_scaled':float(torch.sqrt((rE**2).mean()).detach())}
    np.savetxt(out/'ai_profile.csv',np.c_[h,J.numpy(),P.numpy(),Tc,X,x.numpy()],delimiter=',',header='h_m,J_CH4,J_H2,J_Ar,P_Pa,T_G_C,X_CH4_pct,x_CH4,x_H2,x_Ar',comments='')
    np.savetxt(out/'ai_vs_reference.csv',np.c_[h,X,Xr,eX,Tc,Tr,eT,P.numpy()/1e5,Pr/1e5],delimiter=',',header='h_m,X_AI_pct,X_reference_pct,abs_X_error_pp,T_AI_C,T_reference_C,abs_T_error_C,P_AI_bar,P_reference_bar',comments='')
    np.savez(out/'model_state.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    (out/'training_history.json').write_text(json.dumps(hist,indent=2))
    summary={'model_type':'data-assisted neural surrogate with governing-equation residual audit','ai_X_out_pct':float(X[-1]),'reference_X_out_pct':float(Xr[-1]),'X_MAE_pp':float(eX.mean()),'X_max_error_pp':float(eX.max()),'X_out_error_pp':float(eX[-1]),'T_MAE_C':float(eT.mean()),'T_max_error_C':float(eT.max()),'T_out_error_C':float(eT[-1]),'P_MAE_bar':float(eP.mean()),**phys,'final_training_loss':float(hist[-1]['loss'])}
    (out/'STAGE9_REPORT.md').write_text('# Stage 9 — AI heat-transfer surrogate\n\nThis stage trains a neural surrogate against the validated conventional heat-transfer solver and then audits the learned field against the governing-equation residuals in the bulk reactor. The sharp inlet gas-heating boundary layer is excluded from the residual audit because the reference solution itself is stiff there. This is a deliberate AI-surrogate stage; it is not being presented as a pure PINN solution from governing equations alone.\n\n'+json.dumps(summary,indent=2)+'\n')
    plt.figure(figsize=(7,5)); plt.plot(h,Xr,label='Reference'); plt.plot(h,X,'--',label='AI surrogate'); plt.xlabel('Reactor height h (m)'); plt.ylabel('CH4 conversion (%)'); plt.title('AI surrogate vs reference'); plt.legend(); plt.tight_layout(); plt.savefig(out/'conversion_comparison.png',dpi=180); plt.close()
    plt.figure(figsize=(7,5)); plt.plot(h,Tr,label='Reference'); plt.plot(h,Tc,'--',label='AI surrogate'); plt.xlabel('Reactor height h (m)'); plt.ylabel('Gas temperature (C)'); plt.title('Gas temperature comparison'); plt.legend(); plt.tight_layout(); plt.savefig(out/'temperature_comparison.png',dpi=180); plt.close()
    plt.figure(figsize=(7,5)); plt.plot(h,Pr/1e5,label='Reference'); plt.plot(h,P.numpy()/1e5,'--',label='AI surrogate'); plt.xlabel('Reactor height h (m)'); plt.ylabel('Pressure (bar)'); plt.title('Pressure comparison'); plt.legend(); plt.tight_layout(); plt.savefig(out/'pressure_comparison.png',dpi=180); plt.close()
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
