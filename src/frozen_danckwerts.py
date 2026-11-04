"""Frozen-field axial-dispersion mass BVP with exact Danckwerts BCs."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import PchipInterpolator
from parameters import M_CH4,M_H2
from gas_mixture import concentrations_from_state
from kinetics import reaction_rates
from thermodynamics import KC_from_dimensionless_K
from hydrodynamics import axial_dispersion_smoothed
from heat_transfer_bench import solve_bench_heat_transfer

@dataclass
class Result:
    h: np.ndarray
    J_ch4: np.ndarray
    J_h2: np.ndarray
    J_ar: np.ndarray
    x_ch4: np.ndarray
    x_h2: np.ndarray
    x_ar: np.ndarray
    X_ch4: np.ndarray
    D_ax: np.ndarray
    converged: bool
    message: str


def solve(params,ctx,n_eval=160,tol=1e-4,max_nodes=12000,phi_th=0.001,dax_floor=1e-10,verbose=0):
    heat=solve_bench_heat_transfer(params,ctx.properties,ctx.thermo,n_eval=max(400,n_eval))
    h0=heat.h; jt0=heat.J_ch4+heat.J_h2+heat.J_ar
    rho_g0=jt0/np.maximum(heat.u_g,1e-30)
    ub0=heat.u_g/np.maximum(heat.a_g,1e-30)
    Draw0=50*ub0**3*params.D**1.5
    D0=np.array([axial_dispersion_smoothed(float(u), float(ub0[0]), params.D, float(h), params.H, phi_th) for u,h in zip(ub0,h0)])
    D0=np.maximum(D0,dax_floor)
    K0=rho_g0*heat.a_g*D0
    K0=np.maximum(K0,1e-20)
    Pfun=PchipInterpolator(h0,heat.P); Tfun=PchipInterpolator(h0,heat.T_g)
    agfun=PchipInterpolator(h0,heat.a_g); asfun=PchipInterpolator(h0,heat.a_s)
    Kfun=PchipInterpolator(h0,K0)
    Kpfun=PchipInterpolator(h0,np.gradient(K0,h0))
    Dfun=PchipInterpolator(h0,D0)

    JAr=params.j_Ar_in
    def eval_state(h,J1,J2,z1,z2):
        P=float(Pfun(h)); T=float(Tfun(h)); ag=float(agfun(h)); ass=float(asfun(h)); K=max(float(Kfun(h)),dax_floor)
        Kp=float(Kpfun(h)); D=float(Dfun(h));
        jt=J1+J2+JAr
        gas=concentrations_from_state(P,T,max(J1,1e-14),max(J2,1e-14),JAr)
        kc=KC_from_dimensionless_K(ctx.thermo.dimensionless_K(T),T)
        rn,rc,ra=reaction_rates(T,gas['C_CH4'],gas['C_H2'],ass,kc,ctx.kinetics)
        S1=-ag*ra*M_CH4; S2=2*ag*ra*M_H2; St=S1+S2
        qp=St/jt
        J1p=jt*(gas['x_CH4']*qp+z1)
        J2p=jt*(gas['x_H2']*qp+z2)
        z1p=(J1p-Kp*z1-S1)/K
        z2p=(J2p-Kp*z2-S2)/K
        return J1p,J2p,z1p,z2p,gas,D

    def fun(h,Y):
        out=np.empty_like(Y)
        for k,hh in enumerate(h):
            out[:,k]=eval_state(float(hh),Y[0,k],Y[1,k],Y[2,k],Y[3,k])[:4]
        return out

    def bc(ya,yb):
        J1p,J2p,_,_,_,_=eval_state(0.0,ya[0],ya[1],ya[2],ya[3])
        # Danckwerts: J_i(0)=J_i,in + beta_i * J_i'(0)
        rho_g=float(jt0[0]/heat.u_g[0])
        ag=float(heat.a_g[0]); D=D0[0]; ug=float(heat.u_g[0])
        beta=ag*D/max(ug,1e-30)
        return np.array([
            ya[0]-params.j_CH4_in-beta*J1p,
            ya[1]-params.j_H2_in-beta*J2p,
            yb[2],yb[3]
        ])

    jt=jt0; x1=heat.J_ch4/jt; x2=heat.J_h2/jt
    z1=np.gradient(x1,h0); z2=np.gradient(x2,h0)
    grid=np.linspace(0,params.H,n_eval)
    yinit=np.vstack([np.interp(grid,h0,heat.J_ch4),np.interp(grid,h0,heat.J_h2),np.interp(grid,h0,z1),np.interp(grid,h0,z2)])
    res=solve_bvp(fun,bc,grid,yinit,tol=tol,max_nodes=max_nodes,verbose=verbose)
    if not res.success:
        return None,res
    rows=[]
    for hh,J1,J2 in zip(res.x,res.y[0],res.y[1]):
        jt=J1+J2+JAr
        rows.append((J1/jt,J2/jt,JAr/jt,100*(1-J1/params.j_CH4_in),float(Dfun(hh))))
    A=np.array(rows)
    return Result(res.x,res.y[0],res.y[1],np.full(res.x.size,JAr),A[:,0],A[:,1],A[:,2],A[:,3],A[:,4],True,str(res.message)),res

if __name__=='__main__':
 from parameters import BenchmarkCase,KineticParameters
 from model_context import ModelContext
 from properties import Le2023Properties
 from thermodynamics import Le2023Thermo
 p=BenchmarkCase();ctx=ModelContext(properties=Le2023Properties(),thermo=Le2023Thermo(),kinetics=KineticParameters())
 for phi in [0.05,0.01,0.001]:
  r,res=solve(p,ctx,phi_th=phi,tol=1e-4,verbose=1)
  print('phi',phi,'success',res.success,res.message,'nodes',res.x.size)
  if r: print('Xout',r.X_ch4[-1])
