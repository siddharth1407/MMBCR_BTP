import math
import pytest

from properties import (
    Le2023Properties,
    liquid_density,
    liquid_viscosity,
    liquid_surface_tension,
    species_viscosity,
    gas_mixture_viscosity,
    gas_mixture_conductivity,
    gas_mixture_heat_capacity,
)
from thermodynamics import Le2023Thermo, KC_from_dimensionless_K


def test_bench_liquid_properties_match_manual_ranges():
    T = 1040.0 + 273.15
    assert liquid_density(T) == pytest.approx(9046.2841, rel=2e-6)
    assert liquid_viscosity(T) == pytest.approx(1.9718e-3, rel=2e-4)
    assert liquid_surface_tension(T) == pytest.approx(0.34175, rel=8e-4)


def test_gas_viscosity_is_positive_and_in_manual_range():
    T = 800.0 + 273.15
    mu = gas_mixture_viscosity(T, {"CH4": 0.8 * 0.01604 / (0.8*0.01604+0.2*0.039948),
                                  "H2": 0.0,
                                  "Ar": 0.2 * 0.039948 / (0.8*0.01604+0.2*0.039948)})
    assert mu > 0
    assert 2.5e-5 < mu < 4.5e-5


def test_exact_property_model_interface():
    p = Le2023Properties()
    T = 1040.0 + 273.15
    y = {"CH4": 0.8, "H2": 0.0, "Ar": 0.2}
    assert p.rho_liquid(T) > 0
    assert p.mu_liquid(T) > 0
    assert p.sigma_liquid(T) > 0
    assert p.k_liquid(T) > 0
    assert p.cp_liquid(T) > 0
    assert p.mu_gas(800.0+273.15, y) > 0
    assert p.k_gas(800.0+273.15, y) > 0
    assert p.cp_gas(800.0+273.15, y) > 0


def test_k_equation_and_reaction_enthalpy():
    thermo = Le2023Thermo()
    T = 1200.0
    K = thermo.dimensionless_K(T)
    assert K > 0
    assert KC_from_dimensionless_K(K, T) > 0
    assert thermo.reaction_enthalpy(T) == pytest.approx(74800.0)
