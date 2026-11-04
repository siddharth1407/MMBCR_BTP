import numpy as np
from parameters import KineticParameters, R


def reaction_rates(
    T: float,
    C_CH4: float,
    C_H2: float,
    a_s: float,
    K_C: float,
    params: KineticParameters = KineticParameters(),
) -> tuple[float, float, float]:
    if T <= 0:
        raise ValueError("Temperature must be positive.")
    if C_CH4 < 0 or C_H2 < 0 or a_s < 0 or K_C <= 0:
        raise ValueError("Invalid concentration, surface-area, or K_C.")

    driving_force = (
        1.0 - (C_H2**2) / (C_CH4 * K_C)
        if C_CH4 > 0 else 0.0
    )

    r_non = (
        params.k_non_0
        * np.exp(-params.Ea_non / (R * T))
        * C_CH4**params.n
        * driving_force
    )

    r_cat = (
        params.k_cat_0
        * np.exp(-params.Ea_cat / (R * T))
        * C_CH4
        * driving_force
        * a_s
    )

    return r_non, r_cat, r_non + r_cat
