
import numpy as np

from parameters import M_AR, M_CH4, M_H2, R


def mass_fractions_from_mass_fluxes(J_ch4, J_h2, J_ar):
    """Recover gas-phase mass fractions from species mass fluxes.

    J_i = rho_G * u_G * x_i.
    Solid carbon is not part of the gas-mixture property calculation.
    """
    J = np.array([J_ch4, J_h2, J_ar], dtype=float)
    if np.any(J < -1e-10):
        raise ValueError("Species mass fluxes must be nonnegative.")
    J = np.maximum(J, 0.0)

    J_total = float(J.sum())
    if J_total <= 0:
        raise ValueError("Total gas mass flux must be positive.")

    x = J / J_total
    return {"CH4": x[0], "H2": x[1], "Ar": x[2], "J_total": J_total}


def concentrations_from_state(P, T, J_ch4, J_h2, J_ar):
    """Calculate x_i, M_mix and C_i from species mass fluxes.

    For gas-phase mass fractions:
        M_mix = 1 / sum(x_i/M_i)
        rho_G = P*M_mix/(R*T)
        C_i = rho_G*x_i/M_i
    """
    x = mass_fractions_from_mass_fluxes(J_ch4, J_h2, J_ar)

    denom = (
        x["CH4"] / M_CH4
        + x["H2"] / M_H2
        + x["Ar"] / M_AR
    )
    M_mix = 1.0 / denom

    rho_g = max(float(P), 1.0e3) * M_mix / (R * T)

    return {
        "x_CH4": x["CH4"],
        "x_H2": x["H2"],
        "x_Ar": x["Ar"],
        "J_total": x["J_total"],
        "M_mix": M_mix,
        "rho_g": rho_g,
        "C_CH4": rho_g * x["CH4"] / M_CH4,
        "C_H2": rho_g * x["H2"] / M_H2,
        "C_Ar": rho_g * x["Ar"] / M_AR,
    }
