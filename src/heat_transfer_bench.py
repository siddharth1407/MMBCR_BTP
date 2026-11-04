"""Bench heat-transfer model with liquid temperature fixed at T_W.

This is a controlled intermediate model.  The paper notes that the liquid
phase can remain close to the wall temperature because U_w is very large;
the full Eq. (13) liquid-energy BVP is kept separately for later validation.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from gas_mixture import concentrations_from_state
from hydrodynamics import bubble_diameter,bubble_rising_velocity,gas_holdup,gas_velocity_from_mass_flux
from kinetics import reaction_rates
from parameters import G,M_AR,M_CH4,M_H2,R
from properties import species_heat_capacity_molar,species_heat_capacity_molar_derivative
from thermodynamics import KC_from_dimensionless_K

@dataclass(frozen=True)
class HeatTransferBenchResult:
    h: np.ndarray; J_ch4: np.ndarray; J_h2: np.ndarray; J_ar: np.ndarray; P: np.ndarray; T_g: np.ndarray
    a_g: np.ndarray; u_g: np.ndarray; d_b: np.ndarray; a_s: np.ndarray; r_non: np.ndarray; r_cat: np.ndarray; r_A: np.ndarray; U_h: np.ndarray; N_Pe: np.ndarray; X_ch4: np.ndarray
    @property
    def X_out(self): return float(self.X_ch4[-1])

def solve_bench_heat_transfer(params,properties,thermo,n_eval=300,rtol=1e-7,atol=1e-10):
    T_L=params.T_L_in
    def rhs(h,y):
        J=np.asarray(y[:3]); P=float(y[3]); T=float(y[4])
        gas=concentrations_from_state(P,T,*J)
        yg={"CH4":gas["x_CH4"],"H2":gas["x_H2"],"Ar":gas["x_Ar"]}
        rho_l=properties.rho_liquid(T_L); mu_l=properties.mu_liquid(T_L); sig=properties.sigma_liquid(T_L)
        rho_g=gas["rho_g"]; Jtot=gas["J_total"]; u=gas_velocity_from_mass_flux(Jtot,rho_g)
        a=gas_holdup(u,rho_l,rho_g,sig,mu_l,params.D)
        ub=bubble_rising_velocity(u,a); db=bubble_diameter(params.D,u,rho_l,mu_l/rho_l,sig); ass=6/db
        mug=properties.mu_gas(T,yg); kg=properties.k_gas(T,yg); cpg=properties.cp_gas(T,yg)
        Kc=thermo.dimensionless_K(T)*1e5/(R*T)
        rn,rc,ra=reaction_rates(T,gas['C_CH4'],gas['C_H2'],ass,Kc)
        npe=rho_g*cpg*db*ub/(a*kg); nu=1.1*(npe/(1+mug/mu_l))**0.7; Uh=nu*kg/db
        src=np.array([-1.,2.,0.])*a*ra*np.array([M_CH4,M_H2,M_AR])
        dP=-(rho_l*(1-a)+rho_g*a)*G
        # Eq (11), with T_L fixed at the supplied liquid/wall temperature.
        gas_src=-Uh*ass*(T-T_L)-a*thermo.reaction_enthalpy(T)*ra
        names=('CH4','H2','Ar'); Ms=np.array([M_CH4,M_H2,M_AR]);
        cp=np.array([species_heat_capacity_molar(s,T) for s in names]); dcp=np.array([species_heat_capacity_molar_derivative(s,T) for s in names])
        mf=J/Ms; mf_p=src/Ms
        denom=np.dot(mf,cp)+T*np.dot(mf,dcp)
        dT=(gas_src-T*np.dot(mf_p,cp))/denom
        return np.array([src[0],src[1],src[2],dP,dT])
    y0=np.array([params.j_CH4_in,params.j_H2_in,params.j_Ar_in,params.P_in,params.T_G_in],float)
    sol=solve_ivp(rhs,(0,params.H),y0,method='BDF',rtol=rtol,atol=atol,t_eval=np.linspace(0,params.H,n_eval),max_step=0.01)
    if not sol.success: raise RuntimeError(sol.message)
    rows=[]
    for k,h in enumerate(sol.t):
        J=sol.y[:3,k];P=sol.y[3,k];T=sol.y[4,k]
        gas=concentrations_from_state(P,T,*J);yg={"CH4":gas['x_CH4'],"H2":gas['x_H2'],"Ar":gas['x_Ar']}
        rho_l=properties.rho_liquid(T_L);mu_l=properties.mu_liquid(T_L);sig=properties.sigma_liquid(T_L);rho_g=gas['rho_g'];u=gas_velocity_from_mass_flux(gas['J_total'],rho_g);a=gas_holdup(u,rho_l,rho_g,sig,mu_l,params.D);ub=bubble_rising_velocity(u,a);db=bubble_diameter(params.D,u,rho_l,mu_l/rho_l,sig);ass=6/db;mug=properties.mu_gas(T,yg);kg=properties.k_gas(T,yg);cpg=properties.cp_gas(T,yg);Kc=thermo.dimensionless_K(T)*1e5/(R*T);rn,rc,ra=reaction_rates(T,gas['C_CH4'],gas['C_H2'],ass,Kc);npe=rho_g*cpg*db*ub/(a*kg);Uh=1.1*(npe/(1+mug/mu_l))**.7*kg/db
        rows.append((a,u,db,ass,rn,rc,ra,Uh,npe,100*(1-J[0]/params.j_CH4_in)))
    A=np.asarray(rows)
    return HeatTransferBenchResult(sol.t,sol.y[0],sol.y[1],sol.y[2],sol.y[3],sol.y[4],A[:,0],A[:,1],A[:,2],A[:,3],A[:,4],A[:,5],A[:,6],A[:,7],A[:,8],A[:,9])
