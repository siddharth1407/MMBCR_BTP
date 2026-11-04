"""Property correlations for the Le et al. (2023) MMBCR model.

Source basis
------------
Tables S1-S4 of the supplementary material supplied with the project.
The code keeps the paper's temperature-dependent correlations explicit and
uses mole-fraction linear mixing for gas transport properties. The latter is
validated against the ranges shown in the authors' GUI manual, but the paper
text does not spell out the gas-mixture mixing rule; it is therefore isolated
in one helper so it can be replaced by the original MATLAB rule if the source
package becomes available.

Two source inconsistencies are documented explicitly:
* The S4 Ar B coefficient is printed as 0.081279 under B x 10^7. This makes
  the resulting Ar viscosity negative over the stated range. The VDI value
  is 0.81279, which is consistent with the cited VDI correlation and the GUI
  viscosity range, so the default uses 0.81279 and records the discrepancy.
* The S1 liquid-viscosity line contains a leading 10^-3 in the extracted
  equation. Using it literally gives ~1.97e-6 Pa s, whereas the authors' GUI
  reports ~1.97e-3 Pa s for the benchmark. The implementation therefore uses
  A*exp(B/RT), matching the benchmark magnitude and the underlying correlation.
"""

from __future__ import annotations

from math import exp, log, sqrt
from typing import Mapping

from parameters import M_AR, M_CH4, M_H2, R

Y_NI = 0.27
Y_BI = 1.0 - Y_NI
M_NI = 0.05863
M_BI = 0.20898
M_L = Y_NI * M_NI + Y_BI * M_BI

T_M = 1013.0
N0 = 6.02214e23

# Table S3: Shomate/JANAF coefficients used in Table S2.
# (A, B, C, D, E, F, H), with t = T/1000.
S3 = {
    "CH4_low": (-0.703029, 108.4773, -42.52157, 5.862788, 0.678565,
                 76.84376, -74.87310),
    "CH4_high": (85.81217, 11.26467, -2.114146, 0.138190, -26.42221,
                  -153.5327, -74.87310),
    "H2_low": (33.066178, -11.363417, 11.432816, -2.772874, -0.158558,
               -9.980797, 0.0),
    "H2_mid": (18.563083, 12.257357, -2.859786, 0.268238, 1.977990,
               -1.147438, 0.0),
    "Ar": (20.78600, 2.825911e-7, -1.464191e-7, 1.092131e-8,
           -3.661371e-8, -6.197350, 0.0),
}

# Table S4: coefficients are printed with scale factors in the header.
# Ar B is intentionally stored with the VDI value 0.81279 rather than the
# apparent supplementary-material typo 0.081279; see module docstring.
S4 = {
    "CH4": (-0.07759e-5, 0.50484e-7, -0.43101e-10, 0.03118e-12,
            -0.00981e-15),
    "H2": (0.18024e-5, 0.27174e-7, -0.13395e-10, 0.00585e-12,
           -0.00104e-15),
    "Ar": (0.16196e-5, 0.81279e-7, -0.41263e-10, 0.01668e-12,
           -0.00276e-15),
}

MW = {"CH4": M_CH4, "H2": M_H2, "Ar": M_AR}


def _require_T(T: float, lo: float, hi: float, label: str) -> None:
    if not lo <= T <= hi:
        raise ValueError(f"{label} correlation valid for {lo} <= T <= {hi} K; got {T} K")


def liquid_density(T: float) -> float:
    _require_T(T, 1000.0, 1500.0, "Ni-Bi density")
    rho_ni = 9091.0 - 0.4932 * T
    rho_bi = 10698.0 - 1.2064 * T
    numerator = Y_NI * M_NI + Y_BI * M_BI
    denominator = Y_NI * M_NI / rho_ni + Y_BI * M_BI / rho_bi
    return numerator / denominator


def liquid_viscosity(T: float) -> float:
    _require_T(T, 1000.0, 2000.0, "Ni-Bi viscosity")
    rho_l = liquid_density(T)
    B = 2.65 * T_M ** 1.27
    A = (
        1.7e-7
        * rho_l ** (2.0 / 3.0)
        * T_M ** 0.5
        * M_L ** (-1.0 / 6.0)
        / exp(B / (R * T_M))
    )
    # The authors' GUI benchmark reports ~1.97e-3 Pa.s at 1040 C.
    return A * exp(B / (R * T))


def liquid_heat_capacity(T: float) -> float:
    _require_T(T, 1000.0, 1500.0, "Ni-Bi heat capacity")
    x_ni = Y_NI * M_NI / M_L
    cp_ni = 36.2 + 1.9e-3 * (T - 1728.0)
    cp_bi = 20.0 + 6.15e-3 * T + 21.13e5 / T**2
    # Component equations are molar; Eq. S1 mixes on mass fraction and the
    # reactor energy balance requires J/kg/K, so convert the mixture to a
    # mass basis at the end.
    cp_mix_molar = x_ni * cp_ni + (1.0 - x_ni) * cp_bi
    return cp_mix_molar / M_L


def liquid_surface_tension(T: float) -> float:
    _require_T(T, 1200.0, 1400.0, "Ni-Bi surface tension")
    rho_bi = 10698.0 - 1.2064 * T
    sigma_bi = 0.4208 - 8.1e-5 * T
    V_bi = M_BI / rho_bi
    A_bi = 1.091 * N0 ** (1.0 / 3.0) * V_bi ** (2.0 / 3.0)
    correction = (R * T / A_bi) * log((1.21 * 0.57) / (1.33 * 0.43))
    return sigma_bi + correction


def liquid_conductivity(T: float) -> float:
    _require_T(T, 1000.0, 1500.0, "Ni-Bi thermal conductivity")
    k_ni = 50.867 + 0.021 * T
    k_bi = 6.957 + 0.015 * T
    numerator = k_ni * Y_NI * M_NI + k_bi * Y_BI * M_BI
    denominator = Y_NI * M_NI + Y_BI * M_BI
    return numerator / denominator


def _s3_key(species: str, T: float) -> str:
    species = species.upper()
    if species == "CH4":
        return "CH4_low" if T <= 1300.0 else "CH4_high"
    if species == "H2":
        return "H2_low" if T <= 1000.0 else "H2_mid"
    if species == "AR":
        return "Ar"
    raise KeyError(f"Unknown gas species: {species}")


def species_enthalpy(species: str, T: float) -> float:
    _require_T(T, 298.0, 6000.0, f"{species} enthalpy")
    A, B, C, D, E, F, H = S3[_s3_key(species, T)]
    t = T / 1000.0
    value_kj_mol = (
        A * t
        + B * t**2 / 2.0
        + C * t**3 / 3.0
        + D * t**4 / 4.0
        - E / t
        + F
        - H
    )
    return value_kj_mol * 1000.0


def species_heat_capacity_molar(species: str, T: float) -> float:
    _require_T(T, 298.0, 6000.0, f"{species} heat capacity")
    A, B, C, D, E, _, _ = S3[_s3_key(species, T)]
    t = T / 1000.0
    return A + B * t + C * t**2 + D * t**3 + E / t**2


def species_viscosity(species: str, T: float) -> float:
    _require_T(T, 298.0, 6000.0, f"{species} viscosity")
    A, B, C, D, E = S4["Ar"] if species.upper() == "AR" else S4[species.upper()]
    mu = A + B*T + C*T**2 + D*T**3 + E*T**4
    if mu <= 0.0:
        raise ValueError(
            f"Non-positive {species} viscosity from the source correlation at T={T} K: {mu:.6e} Pa.s"
        )
    return mu


def species_conductivity(species: str, T: float) -> float:
    _require_T(T, 298.0, 6000.0, f"{species} conductivity")
    t = T
    s = species.upper()
    if s == "CH4":
        return (
            8.154e-3 + 0.008e-3*t + 0.351e-6*t**2
            - 0.339e-9*t**3 + 0.1e-12*t**4
        )
    if s == "H2":
        return (
            0.651e-3 + 0.767e-3*t - 0.687e-6*t**2
            + 0.506e-9*t**3 - 0.139e-12*t**4
        )
    if s == "AR":
        return (
            -0.908e-3 + 0.112e-3*t - 0.084e-6*t**2
            + 0.569e-9*t**3 - 0.0156e-12*t**4
        )
    raise KeyError(f"Unknown gas species: {species}")


def _mole_fractions_from_mass_fractions(y: Mapping[str, float]) -> dict[str, float]:
    denom = sum(y[s] / MW[s] for s in MW)
    if denom <= 0:
        raise ValueError("Gas mass fractions must contain a positive total amount.")
    return {s: (y[s] / MW[s]) / denom for s in MW}


def gas_mixture_viscosity(T: float, y: Mapping[str, float]) -> float:
    x = _mole_fractions_from_mass_fractions(y)
    return sum(x[s] * species_viscosity(s, T) for s in MW)


def gas_mixture_conductivity(T: float, y: Mapping[str, float]) -> float:
    x = _mole_fractions_from_mass_fractions(y)
    return sum(x[s] * species_conductivity(s, T) for s in MW)


def gas_mixture_heat_capacity(T: float, y: Mapping[str, float]) -> float:
    x = _mole_fractions_from_mass_fractions(y)
    cp_molar = sum(x[s] * species_heat_capacity_molar(s, T) for s in MW)
    # For mole fractions, the mixture molar mass is sum(x_i M_i).
    M_mix = sum(x[s] * MW[s] for s in MW)
    return cp_molar / M_mix


class Le2023Properties:
    """Concrete property model wired into the reference solver."""

    def rho_liquid(self, T):
        return liquid_density(T)

    def mu_liquid(self, T):
        return liquid_viscosity(T)

    def sigma_liquid(self, T):
        return liquid_surface_tension(T)

    def k_liquid(self, T):
        return liquid_conductivity(T)

    def cp_liquid(self, T):
        return liquid_heat_capacity(T)

    def mu_gas(self, T, y):
        return gas_mixture_viscosity(T, y)

    def k_gas(self, T, y):
        return gas_mixture_conductivity(T, y)

    def cp_gas(self, T, y):
        return gas_mixture_heat_capacity(T, y)


def species_heat_capacity_molar_derivative(species: str, T: float) -> float:
    """d(Cp_molar)/dT for the S3 heat-capacity correlation, J/mol/K^2."""
    _require_T(T, 298.0, 6000.0, f"{species} heat-capacity derivative")
    A, B, C, D, E, _, _ = S3[_s3_key(species, T)]
    t = T / 1000.0
    return (B + 2.0 * C * t + 3.0 * D * t**2 - 2.0 * E / t**3) / 1000.0
