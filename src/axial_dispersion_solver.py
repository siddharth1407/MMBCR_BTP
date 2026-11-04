"""Full Le et al. (2023) heat-transfer + axial-dispersion BVP.

This solver uses a numerically robust state representation:
    q = ln(J_total), x_CH4, d(x_CH4)/dh, x_H2, d(x_H2)/dh,
    P, T_G, T_L, d(T_L)/dh
with x_Ar = 1 - x_CH4 - x_H2 and J_i = J_total*x_i.

The representation preserves positive total gas mass flux and removes the
redundant third composition variable while retaining the three inlet
Danckwerts conditions (two composition equations + total gas-flux equation).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_bvp

from gas_mixture import concentrations_from_state
from hydrodynamics import (
    axial_dispersion_raw,
    axial_dispersion_smoothed,
    bubble_diameter,
    bubble_rising_velocity,
    gas_holdup,
    gas_velocity_from_mass_flux,
)
from kinetics import reaction_rates
from parameters import G, M_AR, M_CH4, M_H2, R
from properties import species_heat_capacity_molar, species_heat_capacity_molar_derivative
from thermodynamics import KC_from_dimensionless_K


@dataclass(frozen=True)
class AxialDispersionResult:
    h: np.ndarray
    J_ch4: np.ndarray
    J_h2: np.ndarray
    J_ar: np.ndarray
    P: np.ndarray
    T_g: np.ndarray
    T_l: np.ndarray
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
    message: str

    @property
    def X_out(self) -> float:
        return float(self.X_ch4[-1])


class _Evaluator:
    """Constitutive evaluator and derivative assembler."""

    def __init__(self, ctx, params):
        self.ctx = ctx
        self.params = params
        self._inlet = None
        self._build_inlet_reference()

    def _build_inlet_reference(self):
        p = self.params
        j = np.array([p.j_CH4_in, p.j_H2_in, p.j_Ar_in], dtype=float)
        jt = float(j.sum())
        x = j / jt
        M = 1.0 / (x[0] / M_CH4 + x[1] / M_H2 + x[2] / M_AR)
        rho_g = p.P_in * M / (R * p.T_G_in)
        u_g = jt / rho_g
        rho_l = self.ctx.properties.rho_liquid(p.T_L_in)
        mu_l = self.ctx.properties.mu_liquid(p.T_L_in)
        sigma = self.ctx.properties.sigma_liquid(p.T_L_in)
        alpha = gas_holdup(u_g, rho_l, rho_g, sigma, mu_l, p.D)
        ub = bubble_rising_velocity(u_g, alpha)
        self._inlet = dict(j=j, jt=jt, x=x, rho_g=rho_g, u_g=u_g, a_g=alpha, u_b=ub)

    def decode(self, y):
        q, x1, z1, x2, z2, P, Tg, Tl, ztl = map(float, y)
        jt = float(np.exp(np.clip(q, -80.0, 20.0)))
        xa = 1.0 - x1 - x2
        J = np.array([jt * x1, jt * x2, jt * xa], dtype=float)
        return q, x1, z1, x2, z2, xa, jt, J, P, Tg, Tl, ztl

    def constitutive(self, h, y):
        q, x1, z1, x2, z2, xa, jt, J, P, Tg, Tl, ztl = self.decode(y)
        # Trial states in solve_bvp can temporarily leave the physical simplex.
        # Keep this only as a numerical guard; converged solutions are checked
        # strictly afterwards.
        if P <= 1.0 or Tg < 298.0 or Tg > 1600.0 or Tl < 1000.0 or Tl > 1500.0:
            P = max(P, 1.0)
            Tg = np.clip(Tg, 298.0001, 1600.0)
            Tl = np.clip(Tl, 1000.0001, 1499.9999)
        eps = 1.0e-12
        J_safe = np.maximum(J, eps)
        gas = concentrations_from_state(P, Tg, *J_safe)

        rho_l = self.ctx.properties.rho_liquid(Tl)
        # Numerical guard for Newton trial states: keep the gas in the
        # physically admissible density range needed by the drift-flux model.
        p_cap = 0.90 * rho_l * R * Tg / gas["M_mix"]
        if P > p_cap:
            P = p_cap
            gas = concentrations_from_state(P, Tg, *J_safe)

        yg = {"CH4": gas["x_CH4"], "H2": gas["x_H2"], "Ar": gas["x_Ar"]}

        mu_l = self.ctx.properties.mu_liquid(Tl)
        sigma_l = self.ctx.properties.sigma_liquid(Tl)
        k_l = self.ctx.properties.k_liquid(Tl)
        cp_l = self.ctx.properties.cp_liquid(Tl)
        mu_g = self.ctx.properties.mu_gas(Tg, yg)
        k_g = self.ctx.properties.k_gas(Tg, yg)
        cp_g = self.ctx.properties.cp_gas(Tg, yg)

        rho_g = gas["rho_g"]
        u_g = gas_velocity_from_mass_flux(gas["J_total"], rho_g)
        a_g = gas_holdup(u_g, rho_l, rho_g, sigma_l, mu_l, self.params.D)
        u_b = bubble_rising_velocity(u_g, a_g)
        db = bubble_diameter(self.params.D, u_g, rho_l, mu_l / rho_l, sigma_l)
        a_s = 6.0 / db

        D_ax_raw = axial_dispersion_raw(u_b, self.params.D)
        # Exact rendered Eq. (4) is centralized in hydrodynamics.py.
        D_ax = axial_dispersion_smoothed(
            u_b, self._inlet["u_b"], self.params.D, h, self.params.H,
            self.ctx.heat_transfer.phi_th,
        )
        # Numerical regularization only. At the exact inlet Eq. (4) gives 0.
        D_ax = max(D_ax, 1.0e-8)

        K_dim = self.ctx.thermo.dimensionless_K(Tg)
        Kc = KC_from_dimensionless_K(K_dim, Tg)
        rn, rc, ra = reaction_rates(Tg, gas["C_CH4"], gas["C_H2"], a_s, Kc, self.ctx.kinetics)

        NPe = rho_g * cp_g * db * u_b / (a_g * k_g)
        Nu = 1.1 * (NPe / (1.0 + mu_g / mu_l)) ** 0.7
        Uh = Nu * k_g / db

        Egas = u_g * rho_g * cp_g * Tg
        Ql = (1.0 - a_g) * k_l * ztl
        Kdisp = rho_g * a_g * D_ax

        nu = np.array([-1.0, 2.0, 0.0])
        M = np.array([M_CH4, M_H2, M_AR])
        src_species = nu * a_g * ra * M
        src_total = float(src_species.sum())

        dP = -(rho_l * (1.0 - a_g) + rho_g * a_g) * G
        dhr = self.ctx.thermo.reaction_enthalpy(Tg)
        gas_src = -Uh * a_s * (Tg - Tl) - a_g * dhr * ra
        liquid_src = (
            Uh * a_s * (Tg - Tl)
            + 4.0 * self.ctx.heat_transfer.Uw / self.params.D * (self.params.T_W - Tl)
            + 4.0 * self.ctx.heat_transfer.Nu_orf * k_g * (Tg - Tl) / self.params.D**2
        )

        # Constitutive derivative flux vector. z_Ar = -z1-z2.
        F = np.array([
            J[0], J[1], Kdisp * z1, Kdisp * z2, Egas, Ql,
        ], dtype=float)

        return {
            "jt": jt, "x": np.array([x1, x2, xa]), "J": J, "gas": gas,
            "rho_l": rho_l, "rho_g": rho_g, "u_g": u_g, "a_g": a_g,
            "u_b": u_b, "db": db, "a_s": a_s, "D_ax": D_ax,
            "D_ax_raw": D_ax_raw, "r_non": rn, "r_cat": rc, "r_A": ra,
            "U_h": Uh, "N_Pe": NPe, "Egas": Egas, "Ql": Ql,
            "Kdisp": Kdisp, "src_species": src_species, "src_total": src_total,
            "dP": dP, "gas_src": gas_src, "liquid_src": liquid_src,
            "F": F, "ztl": ztl, "Tg": Tg, "Tl": Tl,
        }

    @staticmethod
    def _finite_difference(fun, x, index, eps_rel=2e-6):
        scale = max(abs(float(x[index])), 1.0)
        step = eps_rel * scale
        xp = x.copy(); xm = x.copy()
        xp[index] += step
        xm[index] -= step
        return (fun(xp) - fun(xm)) / (2.0 * step)

    def rhs(self, h, y):
        y = np.asarray(y, dtype=float)
        f = self.constitutive(h, y)
        # q' from total gas mass balance; dispersion sums to zero because
        # x_CH4+x_H2+x_Ar=1.
        qprime = f["src_total"] / max(f["jt"], 1e-16)

        # Flux derivatives dF/dh = F_y*y' + F_h. Unknown y' has 9 entries.
        def F_of_state(yy):
            return self.constitutive(h, yy)["F"]
        n = y.size
        Fy = np.zeros((6, n), dtype=float)
        for j in range(n):
            Fy[:, j] = self._finite_difference(F_of_state, y, j)

        # Explicit h derivative.
        sh = 1.0e-6 * max(abs(h), 1.0)
        hp = min(self.params.H, h + sh)
        hm = max(0.0, h - sh)
        if hp == hm:
            hp = min(self.params.H, h + 1e-7)
            hm = max(0.0, h - 1e-7)
        Fp = self.constitutive(hp, y)["F"]
        Fm = self.constitutive(hm, y)["F"]
        Fh = (Fp - Fm) / (hp - hm)

        A = np.zeros((9, 9), dtype=float)
        b = np.zeros(9, dtype=float)
        row = 0

        # CH4/H2 dispersed species balances.
        A[row, :] = Fy[0, :] - Fy[2, :]
        b[row] = f["src_species"][0] - (Fh[0] - Fh[2])
        row += 1
        A[row, :] = Fy[1, :] - Fy[3, :]
        b[row] = f["src_species"][1] - (Fh[1] - Fh[3])
        row += 1

        # Pressure.
        A[row, 5] = 1.0
        b[row] = f["dP"]
        row += 1

        # Gas energy.
        A[row, :] = Fy[4, :]
        b[row] = f["gas_src"] - Fh[4]
        row += 1

        # Liquid conductive flux.
        A[row, :] = Fy[5, :]
        b[row] = f["liquid_src"] - Fh[5]
        row += 1

        # Definitions x1' = z1, x2' = z2, T_L' = zT.
        A[row, 1] = 1.0; b[row] = y[2]; row += 1
        A[row, 3] = 1.0; b[row] = y[4]; row += 1
        A[row, 7] = 1.0; b[row] = y[8]; row += 1

        # q'.
        A[row, 0] = 1.0; b[row] = qprime; row += 1

        assert row == 9
        # Unit scaling is essential: species balances are O(1e-4), pressure is
        # O(1e5), and the gas-energy row is O(1e2-1e4).
        rs = np.array([1.0e4, 1.0e4, 1.0e-5, 1.0e-3, 1.0, 1.0, 1.0, 1.0, 1.0])
        cs = np.array([1.0, 1.0, 1.0, 1.0, 1.0e4, 1.0e3, 1.0, 1.0, 1.0])
        As = rs[:, None] * A * cs[None, :]
        bs = rs * b
        w, *_ = np.linalg.lstsq(As, bs, rcond=1.0e-12)
        return cs * w

    def inlet_bc(self, y):
        f = self.constitutive(0.0, y)
        jt = f["jt"]; x1, x2, xa = f["x"]
        qprime = f["src_total"] / max(jt, 1e-16)
        # J_i' = J_total * (x_i*q' + z_i)
        jprime = np.array([
            jt * (x1 * qprime + y[2]),
            jt * (x2 * qprime + y[4]),
            jt * (xa * qprime - y[2] - y[4]),
        ])
        beta = f["a_g"] * f["D_ax"] / f["u_g"]
        target = self._inlet["j"]
        vals = f["J"] - target - beta * jprime

        # Eq. (16) liquid temperature BC.
        tlbc = y[7] - (
            self.params.T_L_in
            + (1.0 - f["a_g"]) * self.ctx.properties.k_liquid(y[7]) / f["u_g"] * y[8]
        )
        return vals, tlbc


def _initial_guess(params, ctx, n_points=160):
    from heat_transfer_bench import solve_bench_heat_transfer
    try:
        htf = solve_bench_heat_transfer(params, ctx.properties, ctx.thermo, n_eval=n_points)
        h = htf.h
        jt = htf.J_ch4 + htf.J_h2 + htf.J_ar
        x1 = htf.J_ch4 / jt
        x2 = htf.J_h2 / jt
        z1 = np.gradient(x1, h)
        z2 = np.gradient(x2, h)
        ztl = np.zeros_like(h)
        y = np.vstack([
            np.log(np.maximum(jt, 1e-20)), x1, z1, x2, z2,
            htf.P, htf.T_g, np.full_like(h, params.T_L_in), ztl,
        ])
        return h, y
    except Exception:
        h = np.linspace(0.0, params.H, n_points)
        x1 = np.linspace(params.j_CH4_in/(params.j_CH4_in+params.j_Ar_in), 0.05, n_points)
        x2 = 1e-3 * (1.0 - x1)
        jt = np.full_like(h, params.j_CH4_in + params.j_Ar_in)
        y = np.vstack([
            np.log(jt), x1, np.gradient(x1,h), x2, np.gradient(x2,h),
            np.linspace(params.P_in, 0.99*params.P_in,n_points),
            np.linspace(params.T_G_in, params.T_L_in,n_points),
            np.full_like(h, params.T_L_in), np.zeros_like(h)
        ])
        return h, y


def solve_axial_dispersion(params, ctx, n_points=120, max_nodes=6000, tol=2e-4, verbose=0):
    ev = _Evaluator(ctx, params)
    h, y0 = _initial_guess(params, ctx, n_points)

    def fun(hh, yy):
        return np.column_stack([ev.rhs(float(hh[k]), yy[:, k]) for k in range(len(hh))])

    def bc(ya, yb):
        vals, tlbc = ev.inlet_bc(ya)
        return np.array([
            vals[0], vals[1], vals[2],
            ya[5] - params.P_in,
            ya[6] - params.T_G_in,
            tlbc,
            yb[2], yb[4], yb[8],
        ], dtype=float)

    result = solve_bvp(fun, bc, h, y0, tol=tol, max_nodes=max_nodes, verbose=verbose)

    if not result.success:
        return AxialDispersionResult(
            h=result.x,
            J_ch4=np.array([]), J_h2=np.array([]), J_ar=np.array([]), P=np.array([]),
            T_g=np.array([]), T_l=np.array([]), x_ch4=np.array([]), x_h2=np.array([]), x_ar=np.array([]),
            a_g=np.array([]), u_g=np.array([]), u_b=np.array([]), d_b=np.array([]), a_s=np.array([]), D_ax=np.array([]),
            r_non=np.array([]), r_cat=np.array([]), r_A=np.array([]), U_h=np.array([]), N_Pe=np.array([]), X_ch4=np.array([]),
            converged=False, message=str(result.message)
        )

    rows=[]
    Js=[]
    for k, hh in enumerate(result.x):
        yy=result.y[:,k]
        f=ev.constitutive(float(hh),yy)
        Js.append(f["J"])
        rows.append((f["a_g"],f["u_g"],f["u_b"],f["db"],f["a_s"],f["D_ax"],f["r_non"],f["r_cat"],f["r_A"],f["U_h"],f["N_Pe"],100*(1-f["J"][0]/params.j_CH4_in)))
    Jarr=np.asarray(Js); A=np.asarray(rows)
    xch4=result.y[1]; xh2=result.y[3]; xar=1-xch4-xh2
    return AxialDispersionResult(
        h=result.x,
        J_ch4=Jarr[:,0], J_h2=Jarr[:,1], J_ar=Jarr[:,2], P=result.y[5],
        T_g=result.y[6], T_l=result.y[7], x_ch4=xch4, x_h2=xh2, x_ar=xar,
        a_g=A[:,0], u_g=A[:,1], u_b=A[:,2], d_b=A[:,3], a_s=A[:,4], D_ax=A[:,5],
        r_non=A[:,6], r_cat=A[:,7], r_A=A[:,8], U_h=A[:,9], N_Pe=A[:,10], X_ch4=A[:,11],
        converged=True, message=str(result.message)
    )
