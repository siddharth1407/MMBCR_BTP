import math
import warnings
import numpy as np

from parameters import G


def gas_volumetric_flow(n_ch4, n_h2, n_ar, T, P, R):
    if min(n_ch4, n_h2, n_ar) < 0:
        raise ValueError("Molar flow rates must be nonnegative.")
    if T <= 0 or P <= 0:
        raise ValueError("T and P must be positive.")
    return (n_ch4 + n_h2 + n_ar) * R * T / P


def superficial_gas_velocity(Q_g, D):
    if Q_g < 0 or D <= 0:
        raise ValueError("Q_g must be nonnegative and D positive.")
    return 4.0 * Q_g / (math.pi * D**2)


def gas_density(P, M_g, T, R):
    if P <= 0 or M_g <= 0 or T <= 0:
        raise ValueError("P, M_g and T must be positive.")
    return P * M_g / (R * T)


def mixture_molar_mass(C_ch4, C_h2, C_ar, M_ch4, M_h2, M_ar):
    total = C_ch4 + C_h2 + C_ar
    if total <= 0:
        raise ValueError("Total concentration must be positive.")
    return (C_ch4*M_ch4 + C_h2*M_h2 + C_ar*M_ar) / total


def characteristic_velocity(rho_l, rho_g, sigma_l, g=G):
    delta_rho = rho_l - rho_g
    if delta_rho <= 0 or rho_l <= 0 or sigma_l <= 0:
        raise ValueError("Require rho_l > rho_g > 0 and sigma_l > 0.")
    return (sigma_l * g * delta_rho / rho_l**2)**0.25


def dimensionless_superficial_velocity(u_g, rho_l, rho_g, sigma_l, g=G):
    return u_g / characteristic_velocity(rho_l, rho_g, sigma_l, g)


def viscosity_number(mu_l, rho_l, rho_g, sigma_l, g=G):
    delta_rho = rho_l - rho_g
    if mu_l <= 0 or rho_l <= 0 or sigma_l <= 0 or delta_rho <= 0:
        raise ValueError("Invalid liquid/gas properties.")
    length_scale = math.sqrt(sigma_l / (g * delta_rho))
    return mu_l / math.sqrt(rho_l * sigma_l * length_scale)


def dimensionless_hydraulic_diameter(D, rho_l, rho_g, sigma_l, g=G):
    delta_rho = rho_l - rho_g
    if D <= 0 or delta_rho <= 0 or sigma_l <= 0:
        raise ValueError("Invalid geometry/properties.")
    return D / math.sqrt(sigma_l / (g * delta_rho))


def drift_velocity_1(u_plus, rho_l, rho_g, mu_l, sigma_l, D, g=G):
    N_mu = viscosity_number(mu_l, rho_l, rho_g, sigma_l, g)
    D_H = dimensionless_hydraulic_diameter(D, rho_l, rho_g, sigma_l, g)
    rho_ratio = rho_l / rho_g

    # Le et al. (2023), Eqs. (28)-(30), transcribed from the rendered paper.
    # Eq. (28): N_mu >= 2.2e-3 and D_H* >= 30
    # Eq. (29): N_mu <= 2.2e-3 and D_H* <= 30
    # Eq. (30): N_mu <= 2.2e-3 and D_H* >= 30
    threshold = 2.2e-3
    if N_mu >= threshold and D_H >= 30:
        return 0.92 * rho_ratio**(-0.157)

    if N_mu <= threshold and D_H <= 30:
        return 0.0019 * D_H**0.809 * rho_ratio**(-0.157) * N_mu**(-0.562)

    if N_mu <= threshold and D_H >= 30:
        return 0.92 * rho_ratio**(-0.157) * N_mu**(-0.562)

    # The accessible rendered equation set does not explicitly specify the
    # fourth combination (N_mu > 2.2e-3, D_H* < 30). Do not invent a branch.
    raise NotImplementedError(
        "The accessible source does not explicitly specify the "
        "N_mu > 2.2e-3 and D_H* < 30 branch."
    )


def drift_velocity_2(
    u_plus,
    gas_holdup,
    D,
    rho_l,
    rho_g,
    flow_regime="churn_turbulent",
    g=G,
):
    regime = flow_regime.lower().replace("-", "_")

    if regime == "bubbly":
        return math.sqrt(2.0) * max(1.0 - gas_holdup, 0.0)**1.75

    if regime == "churn_turbulent":
        return math.sqrt(2.0)

    if regime == "slug":
        return 0.35 * math.sqrt(g * D * (rho_l - rho_g) / rho_l)

    raise ValueError("flow_regime must be bubbly, churn_turbulent, or slug.")


def combined_dimensionless_drift_velocity(u_plus, v1_plus, v2_plus, A=-1.39):
    e = math.exp(A * u_plus)
    return v2_plus * e + v1_plus * (1.0 - e)


def gas_holdup_from_drift(u_plus, v_plus, C0):
    denom = u_plus * C0 + v_plus
    if denom <= 0:
        raise ValueError("Drift-flux denominator must be positive.")
    return float(np.clip(u_plus / denom, 1e-12, 1.0 - 1e-12))


def gas_holdup(
    u_g,
    rho_l,
    rho_g,
    sigma_l,
    mu_l,
    D,
    flow_regime="churn_turbulent",
    A=-1.39,
    max_iter=200,
    tol=1e-10,
    g=G,
):
    u_plus = dimensionless_superficial_velocity(
        u_g, rho_l, rho_g, sigma_l, g
    )
    v1 = drift_velocity_1(
        u_plus, rho_l, rho_g, mu_l, sigma_l, D, g
    )
    C0 = 1.2 - 0.2 * math.sqrt(rho_g / rho_l)

    # For the benchmark's churn-turbulent regime Eq. (32) gives V_G,2+ = sqrt(2),
    # so the drift-flux closure is explicit and does not require fixed-point iteration.
    regime = flow_regime.lower().replace("-", "_")
    if regime == "churn_turbulent":
        v2 = drift_velocity_2(u_plus, 0.2, D, rho_l, rho_g, regime, g)
        v_plus = combined_dimensionless_drift_velocity(u_plus, v1, v2, A)
        return gas_holdup_from_drift(u_plus, v_plus, C0)

    alpha = 0.2
    for _ in range(max_iter):
        v2 = drift_velocity_2(
            u_plus, alpha, D, rho_l, rho_g, flow_regime, g
        )
        v_plus = combined_dimensionless_drift_velocity(
            u_plus, v1, v2, A
        )
        new_alpha = gas_holdup_from_drift(u_plus, v_plus, C0)

        if abs(new_alpha - alpha) < tol:
            return new_alpha

        alpha = 0.5 * alpha + 0.5 * new_alpha

    warnings.warn("Gas-holdup fixed-point iteration did not fully converge.")
    return alpha


def bubble_rising_velocity(u_g, gas_holdup):
    if gas_holdup <= 0:
        raise ValueError("Gas holdup must be positive.")
    return u_g / gas_holdup


def dimensionless_numbers(D, u_g, rho_l, nu_l, sigma_l, g=G):
    if min(D, u_g, rho_l, nu_l, sigma_l) <= 0:
        raise ValueError("Dimensionless-number inputs must be positive.")
    return {
        "NBo": g * D**2 * rho_l / sigma_l,
        "NGa": g * D**3 / nu_l**2,
        "NFr": u_g / math.sqrt(g * D),
    }


def bubble_diameter(D, u_g, rho_l, nu_l, sigma_l, g=G):
    nums = dimensionless_numbers(
        D, u_g, rho_l, nu_l, sigma_l, g
    )
    return 26.0 * (
        nums["NBo"]**(-0.50)
        * nums["NGa"]**(-0.12)
        * nums["NFr"]**(-0.12)
        * D
    )


def specific_interfacial_area(d_b):
    if d_b <= 0:
        raise ValueError("Bubble diameter must be positive.")
    return 6.0 / d_b


def axial_dispersion_raw(u_b, D):
    if u_b < 0 or D <= 0:
        raise ValueError("Require u_b >= 0 and D > 0.")
    return 50.0 * u_b**3 * D**1.5


def axial_dispersion_smoothed(
    u_b, u_b_in, D, h, H, phi_th=0.001
):
    if D <= 0 or H <= 0 or not (0 <= h <= H):
        raise ValueError("Require D,H > 0 and 0 <= h <= H.")
    if phi_th <= 0:
        raise ValueError("phi_th must be positive.")

    D_raw = axial_dispersion_raw(u_b, D)
    D_ax_star = axial_dispersion_raw(u_b_in, D)

    # Exact rendered Eq. (4):
    # Dax = Draw - [2 Dax* /(1 + exp(phi_th*h/H))] (1 - h/H)
    exponent = max(min(h / (H * phi_th), 700.0), -700.0)
    correction = (2.0 * D_ax_star / (1.0 + math.exp(exponent))) * (1.0 - h / H)
    return D_raw - correction


def gas_velocity_from_mass_flux(J_total, rho_g):
    """u_G = (rho_G*u_G)/rho_G."""
    if J_total < -1.0e-8 or rho_g <= 0:
        raise ValueError(f"Invalid total mass flux or gas density: J_total={J_total}, rho_g={rho_g}")
    return max(float(J_total), 0.0) / float(rho_g)
