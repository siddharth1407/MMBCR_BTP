
import numpy as np

from fluxes import calculate_fluxes
from balances import model_rhs
from state import IDX


def initial_mass_fractions(params):
    """Benchmark feed mole fractions -> gas mass fractions."""
    masses = np.array([
        params.ch4_mole_fraction * 0.01604,
        params.h2_mole_fraction * 0.002016,
        params.ar_mole_fraction * 0.039948,
    ])
    x = masses / masses.sum()

    return {"CH4": x[0], "H2": x[1], "Ar": x[2]}


def bc(ya, yb, ctx, params):
    """Ten boundary conditions matching the paper's Eqs. (14)–(19)."""
    fa = calculate_fluxes(0.0, ya, ctx, params)
    dya = model_rhs(0.0, ya, ctx, params)

    j_in = np.array([
        params.j_CH4_in,
        params.j_H2_in,
        params.j_Ar_in,
    ])

    j_a = fa.species_mass_flux

    # Danckwerts Eq. (14):
    # J_i(0) = J_i,in + alpha_in*Dax_in/uG,in * J_i'(0)
    beta = fa.a_g * fa.D_ax / fa.u_g
    bc_species = j_a - j_in - beta * dya[[IDX.J_CH4, IDX.J_H2, IDX.J_Ar]]

    # Pressure Eq. (15)
    bc_P = ya[IDX.P] - params.P_in

    # Liquid-temperature Eq. (16)
    k_l_in = ctx.properties.k_liquid(ya[IDX.T_L])
    bc_TL = ya[IDX.T_L] - (
        params.T_L_in
        + (1.0 - fa.a_g) * k_l_in / fa.u_g * ya[IDX.ZT_L]
    )

    # Gas temperature Eq. (17)
    bc_TG = ya[IDX.T_G] - params.T_G_in

    # Outlet zero-gradient conditions, Eq. (18), for all three species.
    bc_out_x = np.array([
        yb[IDX.ZX_CH4],
        yb[IDX.ZX_H2],
        yb[IDX.ZX_Ar],
    ])

    # Outlet liquid-temperature gradient, Eq. (19).
    bc_out_TL = yb[IDX.ZT_L]

    return np.concatenate([
        bc_species,
        np.array([bc_P, bc_TL, bc_TG]),
        bc_out_x,
        np.array([bc_out_TL]),
    ])
