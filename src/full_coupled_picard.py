"""Stage 4: fully coupled heat-transfer + axial-dispersion Picard solver.

This solver keeps the Le et al. (2023) constitutive equations and boundary
conditions, but uses outer Picard iteration to avoid a fragile monolithic
Newton solve. The iterations solve, in order:

1. gas-species axial-dispersion BVP (Danckwerts BCs),
2. pressure IVP,
3. gas-energy IVP,
4. liquid-energy BVP (Eq. 13).

At convergence, all fields are mutually consistent. This is a numerical
strategy only; it does not alter the governing equations.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.interpolate import PchipInterpolator

from parameters import BenchmarkCase, KineticParameters, G, M_AR, M_CH4, M_H2, R
from model_context import ModelContext
from properties import Le2023Properties, species_heat_capacity_molar, species_heat_capacity_molar_derivative
from thermodynamics import Le2023Thermo, KC_from_dimensionless_K
from gas_mixture import concentrations_from_state
from hydrodynamics import (
    axial_dispersion_smoothed,
    bubble_diameter,
    bubble_rising_velocity,
    gas_holdup,
    gas_velocity_from_mass_flux,
)
from kinetics import reaction_rates


@dataclass
class FullCoupledResult:
    h: np.ndarray
    J_ch4: np.ndarray
    J_h2: np.ndarray
    J_ar: np.ndarray
    P: np.ndarray
    T_g: np.ndarray
    T_l: np.ndarray
    Q_l: np.ndarray
    x_ch4: np.ndarray
    x_h2: np.ndarray
    x_ar: np.ndarray
    a_g: np.ndarray
    u_g: np.ndarray
    u_b: np.ndarray
    d_b: np.ndarray
    a_s: np.ndarray
    D_ax: np.ndarray
    r_non: np.ndarray
    r_cat: np.ndarray
    r_A: np.ndarray
    U_h: np.ndarray
    N_Pe: np.ndarray
    X_ch4: np.ndarray
    converged: bool
    iterations: int
    history: list[dict]
    message: str

    @property
    def X_out(self) -> float:
        return float(self.X_ch4[-1])


def _interp(arr, h):
    h = np.asarray(h, dtype=float)
    arr = np.asarray(arr, dtype=float)
    return lambda x: float(np.interp(float(x), h, arr))


def _constitutive_at(hh, J1, J2, P, Tg, Tl, ctx, params):
    # Collocation/interpolation can produce machine-level negative values for a species
    # whose flux is zero at the inlet. Treat only tiny negatives as roundoff.
    if J1 < -1e-14 or J2 < -1e-14:
        raise ValueError(f"Negative gas species flux in coupled solve: J1={J1:.3e}, J2={J2:.3e}")
    gas = concentrations_from_state(float(P), float(Tg), max(float(J1), 0.0), max(float(J2), 0.0), params.j_Ar_in)
    ygas = {"CH4": gas["x_CH4"], "H2": gas["x_H2"], "Ar": gas["x_Ar"]}

    rho_l = ctx.properties.rho_liquid(Tl)
    mu_l = ctx.properties.mu_liquid(Tl)
    sigma_l = ctx.properties.sigma_liquid(Tl)
    k_l = ctx.properties.k_liquid(Tl)
    cp_l = ctx.properties.cp_liquid(Tl)

    rho_g = gas["rho_g"]
    u_g = gas_velocity_from_mass_flux(gas["J_total"], rho_g)
    a_g = gas_holdup(u_g, rho_l, rho_g, sigma_l, mu_l, params.D)
    u_b = bubble_rising_velocity(u_g, a_g)
    d_b = bubble_diameter(params.D, u_g, rho_l, mu_l / rho_l, sigma_l)
    a_s = 6.0 / d_b

    mu_g = ctx.properties.mu_gas(Tg, ygas)
    k_g = ctx.properties.k_gas(Tg, ygas)
    cp_g = ctx.properties.cp_gas(Tg, ygas)
    Kdim = ctx.thermo.dimensionless_K(Tg)
    Kc = KC_from_dimensionless_K(Kdim, Tg)
    r_non, r_cat, r_A = reaction_rates(
        Tg, gas["C_CH4"], gas["C_H2"], a_s, Kc, ctx.kinetics
    )

    N_Pe = rho_g * cp_g * d_b * u_b / max(a_g * k_g, 1e-30)
    Nu_G = 1.1 * (N_Pe / (1.0 + mu_g / mu_l)) ** 0.7
    U_h = Nu_G * k_g / d_b

    return {
        "gas": gas,
        "rho_l": rho_l, "mu_l": mu_l, "sigma_l": sigma_l, "k_l": k_l, "cp_l": cp_l,
        "rho_g": rho_g, "u_g": u_g, "a_g": a_g, "u_b": u_b, "d_b": d_b, "a_s": a_s,
        "mu_g": mu_g, "k_g": k_g, "cp_g": cp_g,
        "r_non": r_non, "r_cat": r_cat, "r_A": r_A,
        "N_Pe": N_Pe, "U_h": U_h,
    }


def _compute_fields(h, J1, J2, P, Tg, Tl, ctx, params):
    h = np.asarray(h)
    vals = [_constitutive_at(float(hh), j1, j2, pp, tt, tl, ctx, params)
            for hh, j1, j2, pp, tt, tl in zip(h, J1, J2, P, Tg, Tl)]
    ub_in = vals[0]["u_b"]
    D = np.empty_like(h)
    for i, (hh, f) in enumerate(zip(h, vals)):
        D[i] = axial_dispersion_smoothed(
            f["u_b"], ub_in, params.D, float(hh), params.H, ctx.heat_transfer.phi_th
        )
    # K = rho_G * a_G * D_ax; the exact source Eq. (4) gives D_ax(0)=0.
    K = np.array([f["rho_g"] * f["a_g"] * max(D[i], 0.0) for i, f in enumerate(vals)])
    return {
        "a_g": np.array([f["a_g"] for f in vals]),
        "u_g": np.array([f["u_g"] for f in vals]),
        "u_b": np.array([f["u_b"] for f in vals]),
        "d_b": np.array([f["d_b"] for f in vals]),
        "a_s": np.array([f["a_s"] for f in vals]),
        "D_ax": D,
        "Kdisp": K,
        "U_h": np.array([f["U_h"] for f in vals]),
        "r_A": np.array([f["r_A"] for f in vals]),
    }


def _mass_bvp(params, ctx, h, J1_old, J2_old, P, Tg, Tl, fields, tol=3e-4, max_nodes=10000):
    # Lag K and a_s during this inner solve; the outer Picard restores full coupling.
    Pfun = _interp(P, h); Tfun = _interp(Tg, h)
    agfun = _interp(fields["a_g"], h); asfun = _interp(fields["a_s"], h)
    Kraw = np.maximum(fields["Kdisp"], 1e-12)
    Kfun = _interp(Kraw, h)
    Kpfun = _interp(np.gradient(Kraw, h), h)
    JAr = params.j_Ar_in

    def point(hh, J1, J2, z1, z2):
        pp = float(Pfun(hh)); tt = float(Tfun(hh)); ag = float(agfun(hh)); ass = float(asfun(hh))
        K = max(float(Kfun(hh)), 1e-12); Kp = float(Kpfun(hh))
        gas = concentrations_from_state(pp, tt, max(J1, 1e-14), max(J2, 1e-14), JAr)
        Kc = KC_from_dimensionless_K(ctx.thermo.dimensionless_K(tt), tt)
        _, _, ra = reaction_rates(tt, gas["C_CH4"], gas["C_H2"], ass, Kc, ctx.kinetics)
        S1 = -ag * ra * M_CH4
        S2 = 2.0 * ag * ra * M_H2
        St = S1 + S2
        jt = J1 + J2 + JAr
        jt_prime = St
        x1 = J1 / jt; x2 = J2 / jt
        J1p = jt_prime * x1 + jt * z1
        J2p = jt_prime * x2 + jt * z2
        z1p = (J1p - Kp * z1 - S1) / K
        z2p = (J2p - Kp * z2 - S2) / K
        return J1p, J2p, z1p, z2p

    def fun(hh, Y):
        out = np.empty_like(Y)
        for k, x in enumerate(hh):
            out[:, k] = point(float(x), Y[0, k], Y[1, k], Y[2, k], Y[3, k])
        return out

    # At h=0, the paper's Eq. (4) gives D_ax=0, so Danckwerts reduces to J_i(0)=J_i,in.
    def bc(ya, yb):
        return np.array([ya[0] - params.j_CH4_in, ya[1] - params.j_H2_in, yb[2], yb[3]])

    jt = J1_old + J2_old + JAr
    x1 = J1_old / jt; x2 = J2_old / jt
    z1 = np.gradient(x1, h); z2 = np.gradient(x2, h)
    grid = np.linspace(0.0, params.H, h.size)
    y0 = np.vstack([
        np.interp(grid, h, J1_old), np.interp(grid, h, J2_old),
        np.interp(grid, h, z1), np.interp(grid, h, z2)
    ])
    res = solve_bvp(fun, bc, grid, y0, tol=tol, max_nodes=max_nodes, verbose=0)
    if not res.success:
        raise RuntimeError(f"Mass BVP failed: {res.message}")
    return res


def _pressure_ivp(params, ctx, h, J1, J2, Tg, Tl):
    J1f = _interp(J1, h); J2f = _interp(J2, h); Tgf = _interp(Tg, h); Tlf = _interp(Tl, h)

    def rhs(hh, yy):
        P = float(yy[0]); j1 = float(J1f(hh)); j2 = float(J2f(hh)); tg = float(Tgf(hh)); tl = float(Tlf(hh))
        f = _constitutive_at(hh, j1, j2, P, tg, tl, ctx, params)
        return [-(f["rho_l"] * (1.0 - f["a_g"]) + f["rho_g"] * f["a_g"]) * G]

    sol = solve_ivp(rhs, (0.0, params.H), [params.P_in], t_eval=h, method="BDF",
                    rtol=3e-7, atol=1e-1, max_step=0.01)
    if not sol.success:
        raise RuntimeError(f"Pressure IVP failed: {sol.message}")
    return sol.y[0]


def _gas_temperature_ivp(params, ctx, h, J1, J2, z1, z2, P, Tl, Tg0=None):
    J1f = _interp(J1, h); J2f = _interp(J2, h); z1f = _interp(z1, h); z2f = _interp(z2, h)
    Pf = _interp(P, h); Tlf = _interp(Tl, h)
    JAr = params.j_Ar_in

    def rhs(hh, yy):
        tg = float(yy[0]); pp = float(Pf(hh)); tl = float(Tlf(hh)); j1 = float(J1f(hh)); j2 = float(J2f(hh))
        dz1 = float(z1f(hh)); dz2 = float(z2f(hh))
        f = _constitutive_at(hh, j1, j2, pp, tg, tl, ctx, params)
        Jt = j1 + j2 + JAr
        jt_prime = -f["a_g"] * f["r_A"] * M_CH4 + 2.0 * f["a_g"] * f["r_A"] * M_H2
        x1 = j1 / Jt; x2 = j2 / Jt
        J1p = jt_prime * x1 + Jt * dz1
        J2p = jt_prime * x2 + Jt * dz2
        J3p = 0.0

        dHr = ctx.thermo.reaction_enthalpy(tg)
        gas_src = -f["U_h"] * f["a_s"] * (tg - tl) - f["a_g"] * dHr * f["r_A"]

        Js = np.array([j1, j2, JAr]); Jsp = np.array([J1p, J2p, J3p]); Ms = np.array([M_CH4, M_H2, M_AR])
        cp = np.array([species_heat_capacity_molar(s, tg) for s in ("CH4", "H2", "Ar")])
        dcp = np.array([species_heat_capacity_molar_derivative(s, tg) for s in ("CH4", "H2", "Ar")])
        molar_flux = Js / Ms; molar_flux_prime = Jsp / Ms
        denom = float(np.dot(molar_flux, cp) + tg * np.dot(molar_flux, dcp))
        known = tg * float(np.dot(molar_flux_prime, cp))
        return [(gas_src - known) / denom]

    sol = solve_ivp(rhs, (0.0, params.H), [params.T_G_in if Tg0 is None else float(Tg0[0])],
                    t_eval=h, method="BDF", rtol=3e-7, atol=2e-5, max_step=0.005)
    if not sol.success:
        raise RuntimeError(f"Gas-temperature IVP failed: {sol.message}")
    return sol.y[0]


def _liquid_temperature_bvp(params, ctx, h, J1, J2, P, Tg, Tl_guess, Q_guess):
    """Solve Eq. (13) with hydrodynamic/transfer coefficients lagged from Tl_guess.

    This is the liquid-energy block of the outer Picard iteration. With the
    coefficients frozen, Eq. (13) is a linear two-point BVP and is much more
    robust than a monolithic nonlinear collocation solve. Repeating the outer
    Picard loop restores the full temperature/property coupling.
    """
    h = np.asarray(h, dtype=float)
    coeff=[]
    for hh,j1,j2,pp,tg,tl in zip(h,J1,J2,P,Tg,Tl_guess):
        f=_constitutive_at(float(hh),j1,j2,pp,tg,tl,ctx,params)
        cond=(1.0-f["a_g"])*f["k_l"]
        Uas=f["U_h"]*f["a_s"]
        orf=4.0*ctx.heat_transfer.Nu_orf*f["k_g"]/params.D**2
        coeff.append((max(cond,1e-12),Uas,orf))
    coeff=np.asarray(coeff)
    Kfun=_interp(coeff[:,0],h); Uasfun=_interp(coeff[:,1],h); orffun=_interp(coeff[:,2],h); Tgf=_interp(Tg,h)

    def fun(hh,Y):
        out=np.empty_like(Y)
        for k,x in enumerate(hh):
            tl=float(Y[0,k]); q=float(Y[1,k]);
            cond=float(Kfun(x)); uas=float(Uasfun(x)); orf=float(orffun(x)); tg=float(Tgf(x))
            source=uas*(tg-tl)+4.0*ctx.heat_transfer.Uw/params.D*(params.T_W-tl)+orf*(tg-tl)
            out[0,k]=q/cond
            # Eq. (13), u_L=0: dQ_l/dh + source = 0.
            out[1,k]=-source
        return out

    # Eq. (16) reduces to T_L(0)=T_L,in + Q_l(0)/u_G,in.
    gas0=concentrations_from_state(params.P_in,params.T_G_in,params.j_CH4_in,params.j_H2_in,params.j_Ar_in)
    ug0=gas_velocity_from_mass_flux(gas0["J_total"],gas0["rho_g"])
    def bc(ya,yb):
        return np.array([ya[0]-params.T_L_in-ya[1]/ug0, yb[1]])

    grid=np.linspace(0.0,params.H,h.size)
    y0=np.vstack([np.interp(grid,h,Tl_guess),np.interp(grid,h,Q_guess)])
    if np.max(np.abs(y0[1]))==0.0:
        y0[1]=10.0*(1.0-grid/params.H)
    res=solve_bvp(fun,bc,grid,y0,tol=3e-4,max_nodes=4000,verbose=0)
    if not res.success:
        raise RuntimeError(f"Liquid-energy BVP failed: {res.message}")
    return np.interp(h,res.x,res.y[0]), np.interp(h,res.x,res.y[1])


def _diagnostics(params, ctx, h, J1, J2, P, Tg, Tl, Ql):
    vals=[]
    for hh,j1,j2,pp,tg,tl,ql in zip(h,J1,J2,P,Tg,Tl,Ql):
        f=_constitutive_at(float(hh),j1,j2,pp,tg,tl,ctx,params)
        vals.append(f)
    ub_in=vals[0]["u_b"]
    D=np.array([axial_dispersion_smoothed(f["u_b"],ub_in,params.D,float(hh),params.H,ctx.heat_transfer.phi_th) for hh,f in zip(h,vals)])
    return {
        "x_ch4": np.array([f["gas"]["x_CH4"] for f in vals]),
        "x_h2": np.array([f["gas"]["x_H2"] for f in vals]),
        "x_ar": np.array([f["gas"]["x_Ar"] for f in vals]),
        "a_g": np.array([f["a_g"] for f in vals]),
        "u_g": np.array([f["u_g"] for f in vals]),
        "u_b": np.array([f["u_b"] for f in vals]),
        "d_b": np.array([f["d_b"] for f in vals]),
        "a_s": np.array([f["a_s"] for f in vals]),
        "D_ax": D,
        "r_non": np.array([f["r_non"] for f in vals]),
        "r_cat": np.array([f["r_cat"] for f in vals]),
        "r_A": np.array([f["r_A"] for f in vals]),
        "U_h": np.array([f["U_h"] for f in vals]),
        "N_Pe": np.array([f["N_Pe"] for f in vals]),
    }


def run(params=None, ctx=None, n_points=180, max_iter=18, relax=0.6, tol=2e-3, verbose=True):
    params=params or BenchmarkCase()
    ctx=ctx or ModelContext(properties=Le2023Properties(), thermo=Le2023Thermo(), kinetics=KineticParameters())

    # Start from the controlled Stage-2 solution: TL initially at the supplied wall temperature.
    from heat_transfer_bench import solve_bench_heat_transfer
    heat = solve_bench_heat_transfer(params, ctx.properties, ctx.thermo, n_eval=n_points)
    h=np.linspace(0.0, params.H, n_points)
    J1=np.interp(h,heat.h,heat.J_ch4); J2=np.interp(h,heat.h,heat.J_h2)
    P=np.interp(h,heat.h,heat.P); Tg=np.interp(h,heat.h,heat.T_g)
    Tl=np.full_like(h, params.T_L_in); Ql=np.zeros_like(h)

    history=[]
    for it in range(1,max_iter+1):
        fields=_compute_fields(h,J1,J2,P,Tg,Tl,ctx,params)
        mass=_mass_bvp(params,ctx,h,J1,J2,P,Tg,Tl,fields)
        J1m=np.interp(h,mass.x,mass.y[0]); J2m=np.interp(h,mass.x,mass.y[1]); z1=np.interp(h,mass.x,mass.y[2]); z2=np.interp(h,mass.x,mass.y[3])

        Pm=_pressure_ivp(params,ctx,h,J1m,J2m,Tg,Tl)
        Tgm=_gas_temperature_ivp(params,ctx,h,J1m,J2m,z1,z2,Pm,Tl,Tg0=Tg)
        Tlm_raw,Qlm_raw=_liquid_temperature_bvp(params,ctx,h,J1m,J2m,Pm,Tgm,Tl,Ql)
        # solve_bvp may refine its mesh; project back to the common Picard mesh.
        # The helper currently returns values on that refined mesh when needed.
        if Tlm_raw.size != h.size:
            # Reconstruct the BVP mesh is not exposed by the helper, so use uniform
            # indexing only as a fallback. This branch should be rare because the
            # helper is initialized on the common mesh.
            src=np.linspace(0.0,params.H,Tlm_raw.size)
            Tlm=np.interp(h,src,Tlm_raw); Qlm=np.interp(h,src,Qlm_raw)
        else:
            Tlm,Qlm=Tlm_raw,Qlm_raw

        J1n=relax*J1m+(1-relax)*J1; J2n=relax*J2m+(1-relax)*J2
        Pn=relax*Pm+(1-relax)*P; Tgn=relax*Tgm+(1-relax)*Tg; Tln=relax*Tlm+(1-relax)*Tl; Qln=relax*Qlm+(1-relax)*Ql
        change=max(
            np.max(np.abs((J1n-J1)/np.maximum(np.abs(J1),1e-12))),
            np.max(np.abs((J2n-J2)/np.maximum(np.abs(J2),1e-12))),
            np.max(np.abs((Pn-P)/params.P_in)),
            np.max(np.abs((Tgn-Tg)/100.0)),
            np.max(np.abs((Tln-Tl)/100.0)),
        )
        J1,J2,P,Tg,Tl,Ql=J1n,J2n,Pn,Tgn,Tln,Qln
        X=100.0*(1.0-J1[-1]/params.j_CH4_in)
        row={"iter":it,"max_change":float(change),"X_out":float(X),"P_out_bar":float(P[-1]/1e5),"Tg_out_C":float(Tg[-1]-273.15),"Tl_in_C":float(Tl[0]-273.15),"Tl_out_C":float(Tl[-1]-273.15),"mass_nodes":int(mass.x.size)}
        history.append(row)
        if verbose: print(row)
        if change<tol: break

    diag=_diagnostics(params,ctx,h,J1,J2,P,Tg,Tl,Ql)
    result=FullCoupledResult(
        h=h,J_ch4=J1,J_h2=J2,J_ar=np.full_like(h,params.j_Ar_in),P=P,T_g=Tg,T_l=Tl,Q_l=Ql,
        x_ch4=diag["x_ch4"],x_h2=diag["x_h2"],x_ar=diag["x_ar"],
        a_g=diag["a_g"],u_g=diag["u_g"],u_b=diag["u_b"],d_b=diag["d_b"],a_s=diag["a_s"],D_ax=diag["D_ax"],
        r_non=diag["r_non"],r_cat=diag["r_cat"],r_A=diag["r_A"],U_h=diag["U_h"],N_Pe=diag["N_Pe"],X_ch4=100.0*(1.0-J1/params.j_CH4_in),
        converged=history[-1]["max_change"]<tol, iterations=len(history), history=history,
        message="Converged Picard solution" if history[-1]["max_change"]<tol else "Reached maximum Picard iterations"
    )
    return result


def save_results(result: FullCoupledResult, out_dir: str):
    import csv
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    with (out/"full_coupled_profile.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["h_m","J_CH4","J_H2","J_Ar","P_Pa","T_G_K","T_L_K","Q_L","x_CH4","x_H2","x_Ar","a_G","u_G","u_b","d_b","a_s","D_ax","r_non","r_cat","r_A","U_h","N_Pe","X_CH4_pct"])
        for i in range(result.h.size):
            w.writerow([result.h[i],result.J_ch4[i],result.J_h2[i],result.J_ar[i],result.P[i],result.T_g[i],result.T_l[i],result.Q_l[i],result.x_ch4[i],result.x_h2[i],result.x_ar[i],result.a_g[i],result.u_g[i],result.u_b[i],result.d_b[i],result.a_s[i],result.D_ax[i],result.r_non[i],result.r_cat[i],result.r_A[i],result.U_h[i],result.N_Pe[i],result.X_ch4[i]])
    (out/"picard_history.json").write_text(json.dumps(result.history,indent=2))


if __name__ == "__main__":
    p=BenchmarkCase(); ctx=ModelContext(properties=Le2023Properties(),thermo=Le2023Thermo(),kinetics=KineticParameters())
    r=run(p,ctx,verbose=True)
    print(f"FINAL converged={r.converged}, iterations={r.iterations}, X={r.X_out:.6f} %, Pout={r.P[-1]/1e5:.6f} bar, Tgout={r.T_g[-1]-273.15:.6f} C, TLout={r.T_l[-1]-273.15:.6f} C")
    save_results(r, "/mnt/data/MMBCR_reference_solver_work/MMBCR_reference_solver/results/stage4_full_coupled")
