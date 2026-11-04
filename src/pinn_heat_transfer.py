from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from parameters import BenchmarkCase, G, M_AR, M_CH4, M_H2, R, KineticParameters
from properties import Le2023Properties
from thermodynamics import Le2023Thermo
from heat_transfer_bench import solve_bench_heat_transfer

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

CASE = BenchmarkCase()
PROPS = Le2023Properties()
THERMO = Le2023Thermo()
KIN = KineticParameters()

D = CASE.D
H = CASE.H
P_IN = CASE.P_in
T_IN = CASE.T_G_in
T_L = CASE.T_L_in
J_CH4_IN = CASE.j_CH4_in
J_H2_IN = CASE.j_H2_in
J_AR_IN = CASE.j_Ar_in
J_TOTAL_IN = J_CH4_IN + J_H2_IN + J_AR_IN
DELTA_T = T_L - T_IN
X_IN = np.array([J_CH4_IN, J_H2_IN, J_AR_IN], dtype=float) / J_TOTAL_IN
LOG_X_IN = np.log(np.clip(X_IN, 1e-12, 1.0))
CP_REF = float(PROPS.cp_gas(T_IN, {"CH4": X_IN[0], "H2": X_IN[1], "Ar": X_IN[2]}))
ENERGY_SCALE = J_TOTAL_IN * CP_REF * max(DELTA_T, 1.0) / H

# Liquid properties are constant in this controlled heat-transfer model.
rhoL = float(PROPS.rho_liquid(T_L))
muL = float(PROPS.mu_liquid(T_L))
sigmaL = float(PROPS.sigma_liquid(T_L))

MASS_MW = np.array([M_CH4, M_H2, M_AR], dtype=float)
MASS_MW_T = torch.tensor(MASS_MW, dtype=DTYPE)


def _s3_coeffs(species: str, T: torch.Tensor):
    # Coefficients taken from supplied S3. Piecewise ranges match the source.
    if species == "CH4":
        low = (-0.703029, 108.4773, -42.52157, 5.862788, 0.678565)
        high = (85.81217, 11.26467, -2.114146, 0.138190, -26.42221)
        use_low = T <= 1300.0
    elif species == "H2":
        low = (33.066178, -11.363417, 11.432816, -2.772874, -0.158558)
        high = (18.563083, 12.257357, -2.859786, 1.977990, 1.977990)
        # Correct S3 H2-mid D and E are handled below; this tuple is replaced.
        high = (18.563083, 12.257357, -2.859786, 0.268238, 1.977990)
        use_low = T <= 1000.0
    elif species == "Ar":
        low = high = (20.78600, 2.825911e-7, -1.464191e-7, 1.092131e-8, -3.661371e-8)
        use_low = torch.ones_like(T, dtype=torch.bool)
    else:
        raise KeyError(species)
    a = torch.where(use_low, torch.full_like(T, low[0]), torch.full_like(T, high[0]))
    b = torch.where(use_low, torch.full_like(T, low[1]), torch.full_like(T, high[1]))
    c = torch.where(use_low, torch.full_like(T, low[2]), torch.full_like(T, high[2]))
    d = torch.where(use_low, torch.full_like(T, low[3]), torch.full_like(T, high[3]))
    e = torch.where(use_low, torch.full_like(T, low[4]), torch.full_like(T, high[4]))
    return a, b, c, d, e


def gas_cp_mass_torch(T: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    xmol = (x / MASS_MW_T) / (x / MASS_MW_T).sum(dim=1, keepdim=True)
    t = T / 1000.0
    values = []
    for i, s in enumerate(("CH4", "H2", "Ar")):
        A, B, C, Dd, E = _s3_coeffs(s, T)
        cp_molar = A + B*t + C*t**2 + Dd*t**3 + E/(t**2)
        values.append(cp_molar)
    cp_molar = torch.stack(values, dim=1)
    cp_mix = (xmol * cp_molar).sum(dim=1)
    inv_M = (x / MASS_MW_T).sum(dim=1)
    Mmix = 1.0 / inv_M
    return cp_mix / Mmix


def gas_mu_torch(T: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    xmol = (x / MASS_MW_T) / (x / MASS_MW_T).sum(dim=1, keepdim=True)
    coeff = {
        "CH4": (-0.07759e-5, 0.50484e-7, -0.43101e-10, 0.03118e-12, -0.00981e-15),
        "H2": (0.18024e-5, 0.27174e-7, -0.13395e-10, 0.00585e-12, -0.00104e-15),
        "Ar": (0.16196e-5, 0.81279e-7, -0.41263e-10, 0.01668e-12, -0.00276e-15),
    }
    mus = []
    for s in ("CH4", "H2", "Ar"):
        A,B,C,Dd,E = coeff[s]
        mus.append(A + B*T + C*T**2 + Dd*T**3 + E*T**4)
    mu = torch.stack(mus, dim=1)
    return (xmol * mu).sum(dim=1)


def gas_k_torch(T: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    xmol = (x / MASS_MW_T) / (x / MASS_MW_T).sum(dim=1, keepdim=True)
    # Correlations are transcribed from the supplied S2 source and evaluated
    # specieswise before mole-fraction mixing, matching the project property layer.
    vals = []
    for s in ("CH4", "H2", "Ar"):
        if s == "CH4":
            k = 8.154e-3 + 0.008e-3*T + 0.351e-6*T**2 - 0.339e-9*T**3 + 0.1e-12*T**4
        elif s == "H2":
            k = 0.651e-3 + 0.767e-3*T - 0.687e-6*T**2 + 0.506e-9*T**3 - 0.139e-12*T**4
        else:
            k = -0.908e-3 + 0.112e-3*T - 0.084e-6*T**2 + 0.569e-9*T**3 - 0.0156e-12*T**4
        vals.append(k)
    kg = torch.stack(vals, dim=1)
    return (xmol * kg).sum(dim=1)


def thermo_K_torch(T: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.tensor(13.2714, dtype=DTYPE) - torch.tensor(91204.6, dtype=DTYPE)/(R*T))


def network(width=64, depth=3):
    layers = [nn.Linear(1, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, 6)]
    return nn.Sequential(*layers).to(dtype=DTYPE)


def decode(model: nn.Module, z: torch.Tensor):
    o = model(z)
    s = torch.sigmoid(o)
    # Hard-code inlet boundary conditions and physically directed evolution.
    j_ch4 = J_CH4_IN * (1.0 - z[:,0]*s[:,0])
    j_h2 = (2.0*M_H2/M_CH4) * J_CH4_IN * z[:,0]*s[:,1]
    j_ar = J_AR_IN * (1.0 - 0.02*z[:,0]*s[:,2])
    J = torch.stack([j_ch4,j_h2,j_ar],dim=1)
    P = P_IN * (1.0 - 0.6*z[:,0]*s[:,4])
    T = T_IN + DELTA_T*z[:,0]*s[:,5]
    x = J / J.sum(dim=1,keepdim=True)
    jt = J.sum(dim=1)
    return J, P, T, x, jt

def drift_hydraulics(P, T, x, jt):
    Mmix = 1.0 / (x/MASS_MW_T).sum(dim=1)
    rhoG = P*Mmix/(R*T)
    ug = jt/rhoG
    rhoL_t=torch.full_like(rhoG, rhoL)
    muL_t=torch.full_like(rhoG, muL)
    sig_t=torch.full_like(rhoG, sigmaL)
    Uchar=(sig_t*G*(rhoL_t-rhoG).clamp_min(1e-20)/rhoL_t**2).clamp_min(1e-30)**0.25
    uplus=ug/Uchar
    Nm=muL_t/torch.sqrt(rhoL_t*sig_t*torch.sqrt(sig_t/(G*(rhoL_t-rhoG).clamp_min(1e-20))))
    DH=D/torch.sqrt(sig_t/(G*(rhoL_t-rhoG).clamp_min(1e-20)))
    rr=rhoL_t/rhoG
    v1_lowDH=0.0019*DH**0.809*rr**(-0.157)*Nm**(-0.562)
    v1_highDH=0.92*rr**(-0.157)*Nm**(-0.562)
    v1_no_mu=0.92*rr**(-0.157)
    # Piecewise branches from Eqs. 28-30.
    v1=torch.where(
        Nm >= 2.2e-3,
        torch.where(DH >= 30.0, v1_no_mu, v1_lowDH),
        torch.where(DH >= 30.0, v1_highDH, v1_lowDH),
    )
    v2=torch.sqrt(torch.tensor(2.0,dtype=DTYPE))
    e=torch.exp(-1.39*uplus)
    vp=v2*e+v1*(1-e)
    C0=1.2-0.2*torch.sqrt((rhoG/rhoL_t).clamp_min(1e-30))
    ag=(uplus/(uplus*C0+vp).clamp_min(1e-30)).clamp(1e-6,0.999)
    NBo=G*D**2*rhoL_t/sig_t
    nuL=muL_t/rhoL_t
    NGa=G*D**3/nuL**2
    NFr=ug/torch.sqrt(torch.tensor(G*D,dtype=DTYPE))
    db=26*NBo**(-0.50)*NGa**(-0.12)*NFr**(-0.12)*D
    a_s=6/db
    return rhoG,ug,ag,db,a_s


def residuals(model, z):
    z = z.clone().detach().requires_grad_(True)
    J,P,T,x,jt = decode(model,z)
    rhoG,ug,ag,db,a_s = drift_hydraulics(P,T,x,jt)
    muG=gas_mu_torch(T,x)
    kG=gas_k_torch(T,x)
    cpG=gas_cp_mass_torch(T,x)
    ub=ug/ag
    NPe=rhoG*cpG*db*ub/(ag*kG.clamp_min(1e-12))
    Nu=1.1*(NPe/(1+muG/muL)).clamp_min(1e-12)**0.7
    Uh=Nu*kG/db
    C_CH4=rhoG*x[:,0]/M_CH4
    C_H2=rhoG*x[:,1]/M_H2
    Kc=thermo_K_torch(T)*1e5/(R*T)
    drive=1-(C_H2**2)/(C_CH4.clamp_min(1e-12)*Kc)
    rnon=KIN.k_non_0*torch.exp(-KIN.Ea_non/(R*T))*C_CH4**KIN.n*drive
    rcat=KIN.k_cat_0*torch.exp(-KIN.Ea_cat/(R*T))*C_CH4*drive*a_s
    rA=rnon+rcat
    src=torch.stack([-ag*rA*M_CH4, 2*ag*rA*M_H2, torch.zeros_like(rA)],dim=1)
    dJ=torch.stack([torch.autograd.grad(J[:,i].sum(),z,create_graph=True)[0][:,0] for i in range(3)],dim=1)/H
    dP=torch.autograd.grad(P.sum(),z,create_graph=True)[0][:,0]/H
    Eflux=jt*cpG*T
    dE=torch.autograd.grad(Eflux.sum(),z,create_graph=True)[0][:,0]/H
    dP_model=-(rhoL*(1-ag)+rhoG*ag)*G
    gas_src=-Uh*a_s*(T-T_L)-ag*(2*37.4e3)*rA
    return {
        'mass':(dJ-src)/(J_TOTAL_IN/H),
        'pressure':(dP-dP_model)/(P_IN/H),
        'energy':(dE-gas_src)/ENERGY_SCALE,
        'meta':dict(J=J,P=P,T=T,x=x,jt=jt,rhoG=rhoG,ug=ug,ag=ag,db=db,a_s=a_s,rA=rA,Uh=Uh,NPe=NPe)
    }


def train(seed=11, pretrain_epochs=700, physics_epochs=1200, nc=128):
    torch.manual_seed(seed); np.random.seed(seed)
    model=network(64,3)
    with torch.no_grad():
        final=model[-1]
        final.weight.mul_(0.02)
        # logit-like biases for the bounded transforms near the benchmark inlet.
        final.bias.copy_(torch.tensor([0.0,0.0,-6.0,0.0,0.0,0.0],dtype=DTYPE))
    ref=solve_bench_heat_transfer(CASE,PROPS,THERMO,n_eval=nc)
    z_data=torch.tensor((ref.h/H)[:,None],dtype=DTYPE)
    J_ref=torch.tensor(np.column_stack([ref.J_ch4,ref.J_h2,ref.J_ar]),dtype=DTYPE)
    P_ref=torch.tensor(ref.P,dtype=DTYPE)
    T_ref=torch.tensor(ref.T_g,dtype=DTYPE)
    x_ref=J_ref/J_ref.sum(dim=1,keepdim=True)

    # Stage A: supervised pretraining on the trusted conventional heat-transfer solver.
    opt=torch.optim.Adam(model.parameters(),lr=2e-3)
    hist=[]
    for ep in range(pretrain_epochs):
        opt.zero_grad()
        J,P,T,x,jt=decode(model,z_data)
        lossJ=((J-J_ref)/J_TOTAL_IN).pow(2).mean()
        lossP=((P-P_ref)/P_IN).pow(2).mean()
        lossT=((T-T_ref)/max(DELTA_T,1.0)).pow(2).mean()
        loss=lossJ+lossP+lossT
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),20.0); opt.step()
        if ep%100==0 or ep==pretrain_epochs-1:
            hist.append({'phase':'pretrain','epoch':ep,'loss':float(loss.detach()),'data':float(loss.detach())})

    # Stage B: physics-informed fine-tuning, retaining a small data anchor to avoid drift.
    z=torch.linspace(0,1,nc,dtype=DTYPE).reshape(-1,1)
    opt=torch.optim.Adam(model.parameters(),lr=5e-4)
    for ep in range(physics_epochs):
        opt.zero_grad()
        rr=residuals(model,z)
        loss_phys=(rr['mass']**2).mean()+0.5*(rr['pressure']**2).mean()+0.5*(rr['energy']**2).mean()
        J,P,T,_,_=decode(model,z)
        dataJ=((J-J_ref)/J_TOTAL_IN).pow(2).mean()
        dataP=((P-P_ref)/P_IN).pow(2).mean()
        dataT=((T-T_ref)/max(DELTA_T,1.0)).pow(2).mean()
        data_loss=dataJ+dataP+dataT
        loss=loss_phys+0.2*data_loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),20.0); opt.step()
        if ep%100==0 or ep==physics_epochs-1:
            hist.append({'phase':'physics','epoch':ep,'loss':float(loss.detach()),'physics':float(loss_phys.detach()),'data':float(data_loss.detach())})

    lb= torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=50, tolerance_grad=1e-10,
                          tolerance_change=1e-12, line_search_fn='strong_wolfe')
    def closure():
        lb.zero_grad(); rr=residuals(model,z)
        J,P,T,_,_=decode(model,z)
        dataJ=((J-J_ref)/J_TOTAL_IN).pow(2).mean()
        dataP=((P-P_ref)/P_IN).pow(2).mean()
        dataT=((T-T_ref)/max(DELTA_T,1.0)).pow(2).mean()
        loss=(rr['mass']**2).mean()+0.5*(rr['pressure']**2).mean()+0.5*(rr['energy']**2).mean()+0.2*(dataJ+dataP+dataT)
        loss.backward(); return loss
    lb.step(closure)
    return model,hist,ref

def main():
    out=Path('results/stage9_pinn_heat_transfer'); out.mkdir(parents=True,exist_ok=True)
    model,hist,ref=train()
    z=torch.linspace(0,1,301,dtype=DTYPE).reshape(-1,1)
    with torch.no_grad(): J,P,T,x,jt=decode(model,z)
    from scipy.interpolate import interp1d
    ref=solve_bench_heat_transfer(CASE,PROPS,THERMO,n_eval=601)
    h=z[:,0].numpy()*H
    Xpin=100*(1-J[:,0].numpy()/J_CH4_IN)
    Tpin=T.numpy()-273.15
    aref=interp1d(ref.h,ref.a_g)(h); dbref=interp1d(ref.h,ref.d_b)(h); asref=interp1d(ref.h,ref.a_s)(h); uref=interp1d(ref.h,ref.u_g)(h); Pref=interp1d(ref.h,ref.P)(h); Tref=interp1d(ref.h,ref.T_g)(h)-273.15; Xref=interp1d(ref.h,ref.X_ch4)(h)
    errX=np.abs(Xpin-Xref); errT=np.abs(Tpin-Tref)
    data=np.column_stack([h,J[:,0].numpy(),J[:,1].numpy(),J[:,2].numpy(),P.numpy(),Tpin,Xpin,x.numpy(),jt.numpy(),aref,dbref,asref,uref])
    np.savetxt(out/'pinn_heat_profile.csv',data,delimiter=',',header='h_m,J_CH4,J_H2,J_Ar,P_Pa,T_G_C,X_CH4_pct,x_CH4,x_H2,x_Ar,J_total,a_g_ref,db_ref_m,as_ref_m2_m3,u_g_ref_m_s',comments='')
    comp=np.column_stack([h,Xpin,Xref,errX,Tpin,Tref,errT])
    np.savetxt(out/'pinn_vs_reference.csv',comp,delimiter=',',header='h_m,X_PINN_pct,X_reference_pct,abs_X_error_pp,T_PINN_C,T_reference_C,abs_T_error_C',comments='')
    np.savez(out/'model_state.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    (out/'training_history.json').write_text(json.dumps(hist,indent=2))
    summary={'pinn_X_out_pct':float(Xpin[-1]),'reference_X_out_pct':float(Xref[-1]),'X_MAE_pp':float(errX.mean()),'X_max_error_pp':float(errX.max()),'T_MAE_C':float(errT.mean()),'T_max_error_C':float(errT.max()),'final_loss':float(hist[-1]['loss'])}
    (out/'STAGE9_REPORT.md').write_text('# Stage 9 — PINN heat-transfer model\n\n'+json.dumps(summary,indent=2)+'\n')
    # Plots
    plt.figure(figsize=(7,5)); plt.plot(h,Xref,label='Reference heat-transfer'); plt.plot(h,Xpin,'--',label='PINN'); plt.xlabel('Reactor height h (m)'); plt.ylabel('CH4 conversion (%)'); plt.title('Stage 9 PINN vs reference'); plt.legend(); plt.tight_layout(); plt.savefig(out/'conversion_comparison.png',dpi=180); plt.close()
    plt.figure(figsize=(7,5)); plt.plot(h,Tref,label='Reference T_G'); plt.plot(h,Tpin,'--',label='PINN T_G'); plt.xlabel('Reactor height h (m)'); plt.ylabel('Gas temperature (°C)'); plt.title('Stage 9 gas-temperature comparison'); plt.legend(); plt.tight_layout(); plt.savefig(out/'temperature_comparison.png',dpi=180); plt.close()
    plt.figure(figsize=(7,5)); plt.semilogy([d['epoch'] for d in hist],[d['loss'] for d in hist],marker='o'); plt.xlabel('Epoch'); plt.ylabel('Training loss'); plt.title('Stage 9 PINN training history'); plt.tight_layout(); plt.savefig(out/'training_loss.png',dpi=180); plt.close()
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
