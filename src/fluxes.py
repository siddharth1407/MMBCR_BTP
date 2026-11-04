
from dataclasses import dataclass
import numpy as np

from gas_mixture import concentrations_from_state
from hydrodynamics import (
    axial_dispersion_smoothed,
    bubble_diameter,
    bubble_rising_velocity,
    gas_holdup,
    gas_velocity_from_mass_flux,
    specific_interfacial_area,
)
from kinetics import reaction_rates
from parameters import M_AR, M_CH4, M_H2, R
from state import IDX
from thermodynamics import KC_from_dimensionless_K


@dataclass(frozen=True)
class FluxResult:
    species_mass_flux: np.ndarray
    gas_energy_flux: float
    liquid_conduction_flux: float

    rho_g: float
    u_g: float
    a_g: float
    u_b: float
    d_b: float
    a_s: float
    D_ax: float

    r_non: float
    r_cat: float
    r_A: float

    U_h: float
    N_Pe: float

    x_CH4: float
    x_H2: float
    x_Ar: float
    M_mix: float
    C_CH4: float
    C_H2: float
    C_Ar: float


def calculate_fluxes(h, y, ctx, params):
    """Calculate all constitutive quantities at one axial location."""
    J_ch4 = y[IDX.J_CH4]
    J_h2 = y[IDX.J_H2]
    J_ar = y[IDX.J_Ar]
    P = y[IDX.P]
    T_g = y[IDX.T_G]
    T_l = y[IDX.T_L]

    if P <= 0 or T_g <= 0 or T_l <= 0:
        raise ValueError("Nonphysical P/T encountered.")

    gas = concentrations_from_state(
        P, T_g, J_ch4, J_h2, J_ar
    )

    y_gas = {
        "CH4": gas["x_CH4"],
        "H2": gas["x_H2"],
        "Ar": gas["x_Ar"],
    }

    rho_l = ctx.properties.rho_liquid(T_l)
    mu_l = ctx.properties.mu_liquid(T_l)
    sigma_l = ctx.properties.sigma_liquid(T_l)
    k_l = ctx.properties.k_liquid(T_l)
    cp_l = ctx.properties.cp_liquid(T_l)

    mu_g = ctx.properties.mu_gas(T_g, y_gas)
    k_g = ctx.properties.k_gas(T_g, y_gas)
    cp_g = ctx.properties.cp_gas(T_g, y_gas)

    rho_g = gas["rho_g"]
    J_total = gas["J_total"]
    u_g = gas_velocity_from_mass_flux(J_total, rho_g)

    a_g = gas_holdup(
        u_g, rho_l, rho_g, sigma_l, mu_l, params.D
    )

    u_b = bubble_rising_velocity(u_g, a_g)

    nu_l = mu_l / rho_l
    d_b = bubble_diameter(
        params.D, u_g, rho_l, nu_l, sigma_l
    )
    a_s = specific_interfacial_area(d_b)

    # Eq. (4) inlet smoothing is kept in one function for easy verification.
    # Eq. (4) uses the inlet dispersion coefficient D_ax* as a fixed
    # reference quantity. The inlet bubble velocity is computed from the
    # inlet mass fluxes / boundary pressure and inlet gas temperature.
    j_in = np.array([params.j_CH4_in, params.j_H2_in, params.j_Ar_in], dtype=float)
    j_total_in = float(j_in.sum())
    y_in = j_in / j_total_in
    M_in = 1.0 / (
        y_in[0] / M_CH4 + y_in[1] / M_H2 + y_in[2] / M_AR
    )
    rho_g_in = params.P_in * M_in / (R * params.T_G_in)
    u_g_in = j_total_in / rho_g_in
    rho_l_in = ctx.properties.rho_liquid(params.T_L_in)
    mu_l_in = ctx.properties.mu_liquid(params.T_L_in)
    sigma_l_in = ctx.properties.sigma_liquid(params.T_L_in)
    a_g_in = gas_holdup(
        u_g_in, rho_l_in, rho_g_in, sigma_l_in, mu_l_in, params.D
    )
    u_b_in = bubble_rising_velocity(u_g_in, a_g_in)

    D_ax = axial_dispersion_smoothed(
        u_b,
        u_b_in,
        params.D,
        h,
        params.H,
        ctx.heat_transfer.phi_th,
    )
    # The exact sigmoid gives D_ax(0)=0, which creates a singular Jacobian
    # at the collocation endpoint when the second-order dispersed species
    # equations are rewritten as first order. The MATLAB implementation used
    # a dedicated BVP/initialization workflow; we regularize only this
    # numerical endpoint with a vanishingly small positive floor.
    D_ax = max(D_ax, 1.0e-10)

    K = ctx.thermo.dimensionless_K(T_g)
    K_c = KC_from_dimensionless_K(K, T_g)

    r_non, r_cat, r_A = reaction_rates(
        T_g,
        gas["C_CH4"],
        gas["C_H2"],
        a_s,
        K_c,
        ctx.kinetics,
    )

    N_Pe = rho_g * cp_g * d_b * u_b / (a_g * k_g)
    Nu_G = 1.1 * (N_Pe / (1.0 + mu_g / mu_l))**0.7
    U_h = Nu_G * k_g / d_b

    gas_energy_flux = u_g * rho_g * cp_g * T_g
    liquid_conduction_flux = (1.0 - a_g) * k_l * y[IDX.ZT_L]

    return FluxResult(
        species_mass_flux=np.array([J_ch4, J_h2, J_ar], dtype=float),
        gas_energy_flux=gas_energy_flux,
        liquid_conduction_flux=liquid_conduction_flux,
        rho_g=rho_g,
        u_g=u_g,
        a_g=a_g,
        u_b=u_b,
        d_b=d_b,
        a_s=a_s,
        D_ax=D_ax,
        r_non=r_non,
        r_cat=r_cat,
        r_A=r_A,
        U_h=U_h,
        N_Pe=N_Pe,
        x_CH4=gas["x_CH4"],
        x_H2=gas["x_H2"],
        x_Ar=gas["x_Ar"],
        M_mix=gas["M_mix"],
        C_CH4=gas["C_CH4"],
        C_H2=gas["C_H2"],
        C_Ar=gas["C_Ar"],
    )
