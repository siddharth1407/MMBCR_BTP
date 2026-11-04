"""Reaction + momentum benchmark for the Le et al. MMBCR model.

This is the first validation layer: the reaction-only / isothermal model has
first-order ODEs and avoids the singular endpoint introduced by the axial-
dispersion smoothing. It is useful for checking kinetics, hydrodynamics,
pressure drop, and methane conversion before adding energy and dispersion.
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from gas_mixture import concentrations_from_state
from hydrodynamics import (
    bubble_diameter,
    gas_holdup,
    gas_velocity_from_mass_flux,
)
from kinetics import reaction_rates
from parameters import G, M_AR, M_CH4, M_H2, R


@dataclass(frozen=True)
class ReactionOnlyResult:
    h: np.ndarray
    J_ch4: np.ndarray
    J_h2: np.ndarray
    J_ar: np.ndarray
    P: np.ndarray
    X_ch4: np.ndarray
    a_g: np.ndarray
    u_g: np.ndarray
    d_b: np.ndarray
    a_s: np.ndarray
    r_non: np.ndarray
    r_cat: np.ndarray
    r_A: np.ndarray

    @property
    def X_out(self) -> float:
        return float(self.X_ch4[-1])


def solve_reaction_only(params, properties, thermo, kinetics=None, method="BDF", n_eval=250, rtol=2e-7, atol=1e-10):
    """Solve the reaction-only benchmark at constant gas/liquid temperature.

    The paper's Output 1 is an isothermal reaction model.  The benchmark feed
    is therefore evaluated at the supplied liquid/operating temperature.
    """
    if kinetics is None:
        from parameters import KineticParameters
        kinetics = KineticParameters()

    T = params.T_L_in
    T_l = params.T_L_in

    j0 = np.array([params.j_CH4_in, params.j_H2_in, params.j_Ar_in], dtype=float)
    n0 = np.linspace(0.0, params.H, n_eval)

    def rhs(h, y):
        J_ch4, J_h2, J_ar, P = y
        gas = concentrations_from_state(P, T, J_ch4, J_h2, J_ar)

        y_gas = {
            "CH4": gas["x_CH4"],
            "H2": gas["x_H2"],
            "Ar": gas["x_Ar"],
        }
        rho_l = properties.rho_liquid(T_l)
        mu_l = properties.mu_liquid(T_l)
        sigma_l = properties.sigma_liquid(T_l)
        rho_g = gas["rho_g"]
        u_g = gas_velocity_from_mass_flux(gas["J_total"], rho_g)
        a_g = gas_holdup(u_g, rho_l, rho_g, sigma_l, mu_l, params.D)
        d_b = bubble_diameter(
            params.D,
            u_g,
            rho_l,
            mu_l / rho_l,
            sigma_l,
        )
        a_s = 6.0 / d_b
        K = thermo.dimensionless_K(T)
        K_c = K * 1.0e5 / (R * T)
        r_non, r_cat, r_A = reaction_rates(
            T,
            gas["C_CH4"],
            gas["C_H2"],
            a_s,
            K_c,
            params=kinetics,
        )

        # Mass-conserving stoichiometric source for CH4 -> C + 2H2.
        nu = np.array([-1.0, 2.0, 0.0])
        M = np.array([M_CH4, M_H2, M_AR])
        dJ = nu * a_g * r_A * M
        dP = -(
            rho_l * (1.0 - a_g) + rho_g * a_g
        ) * G
        return np.array([dJ[0], dJ[1], dJ[2], dP])

    sol = solve_ivp(
        rhs,
        (0.0, params.H),
        np.array([*j0, params.P_in], dtype=float),
        t_eval=n0,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Reaction-only solver failed: {sol.message}")

    vals = []
    for h, Jc, Jh, Ja, P in zip(sol.t, *sol.y):
        gas = concentrations_from_state(P, T, Jc, Jh, Ja)
        rho_l = properties.rho_liquid(T_l)
        mu_l = properties.mu_liquid(T_l)
        sigma_l = properties.sigma_liquid(T_l)
        u_g = gas_velocity_from_mass_flux(gas["J_total"], gas["rho_g"])
        a_g = gas_holdup(u_g, rho_l, gas["rho_g"], sigma_l, mu_l, params.D)
        d_b = bubble_diameter(params.D, u_g, rho_l, mu_l / rho_l, sigma_l)
        a_s = 6.0 / d_b
        Kc = thermo.dimensionless_K(T) * 1.0e5 / (R * T)
        r_non, r_cat, r_A = reaction_rates(T, gas["C_CH4"], gas["C_H2"], a_s, Kc)
        vals.append((100.0 * (1.0 - Jc / params.j_CH4_in), a_g, u_g, d_b, a_s, r_non, r_cat, r_A))

    arr = np.asarray(vals)
    return ReactionOnlyResult(
        h=sol.t,
        J_ch4=sol.y[0], J_h2=sol.y[1], J_ar=sol.y[2], P=sol.y[3],
        X_ch4=arr[:, 0], a_g=arr[:, 1], u_g=arr[:, 2], d_b=arr[:, 3],
        a_s=arr[:, 4], r_non=arr[:, 5], r_cat=arr[:, 6], r_A=arr[:, 7],
    )
