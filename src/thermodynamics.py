"""Thermodynamic closure used by the Le et al. MMBCR reference model."""

import math
import warnings

from parameters import R, P_STD


def KC_from_dimensionless_K(K: float, T: float) -> float:
    if K <= 0 or T <= 0:
        raise ValueError("K and T must be positive.")
    return K * P_STD / (R * T)


def dimensionless_equilibrium_constant(T: float) -> float:
    """Dimensionless K used by Catalan & Rezaei and inherited by Le et al.

    In the cited 2022 catalytic MMBCR model, for 900-1200 C:
        K = exp(13.2714 - 91204.6/(R*T))
    """
    if T < 1173.15 or T > 1473.15:
        # The supplied Le et al. (2023) paper says K is obtained from Gibbs
        # free energies but does not reproduce that full temperature relation.
        # This implementation uses the cited Catalan-Rezaei correlation as the
        # available closure and explicitly warns when evaluated outside its
        # stated 900-1200 C range. This is needed because the benchmark inlet
        # gas temperature is 800 C.
        warnings.warn(
            "K(T) correlation is being evaluated outside its stated 900-1200 C range.",
            RuntimeWarning,
            stacklevel=2,
        )
    return math.exp(13.2714 - 91204.6 / (R * T))


def reaction_enthalpy(T: float) -> float:
    """Heat of reaction for CH4 -> C + 2 H2, J/mol-CH4.

    The authors' manual states Delta H_R = 37.4 kJ/mol-H2 for methane
    decomposition, i.e. 74.8 kJ/mol-CH4. The current reference implementation
    keeps that explicitly stated value rather than inventing an additional
    temperature-dependent closure not given in the supplied supplementary
    tables.
    """
    if T <= 0:
        raise ValueError("Temperature must be positive.")
    return 2.0 * 37.4e3


class Le2023Thermo:
    def dimensionless_K(self, T):
        return dimensionless_equilibrium_constant(T)

    def reaction_enthalpy(self, T):
        return reaction_enthalpy(T)
