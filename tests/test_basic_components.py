import numpy as np
import pytest

from parameters import R, M_CH4, M_H2
from hydrodynamics import (
    gas_volumetric_flow,
    superficial_gas_velocity,
    gas_density,
    mixture_molar_mass,
    bubble_rising_velocity,
    specific_interfacial_area,
    axial_dispersion_raw,
)
from kinetics import reaction_rates
from thermodynamics import KC_from_dimensionless_K


def test_ideal_gas_flow_and_velocity():
    Q = gas_volumetric_flow(1e-5, 2e-5, 0.0, 1000.0, 2e5, R)
    assert Q > 0
    u = superficial_gas_velocity(Q, 0.03)
    assert u > 0


def test_gas_density():
    rho = gas_density(2e5, 0.02, 1000.0, R)
    assert rho == pytest.approx(2e5 * 0.02 / (R * 1000.0))


def test_mixture_molar_mass():
    M = mixture_molar_mass(
        0.8, 0.2, 0.0, M_CH4, M_H2, 0.002
    )
    assert M == pytest.approx(0.8 * M_CH4 + 0.2 * M_H2)


def test_bubble_area_and_dispersion():
    db = 0.007
    a_s = specific_interfacial_area(db)
    assert a_s == pytest.approx(6.0 / db)

    ub = bubble_rising_velocity(0.005, 0.025)
    Dax = axial_dispersion_raw(ub, 0.03)
    assert ub > 0 and Dax > 0


def test_kinetics():
    r_non, r_cat, r_A = reaction_rates(
        T=1300.0,
        C_CH4=100.0,
        C_H2=1.0,
        a_s=800.0,
        K_C=100.0,
    )
    assert np.isfinite(r_non)
    assert np.isfinite(r_cat)
    assert r_A == pytest.approx(r_non + r_cat)


def test_equilibrium_constant_conversion():
    Kc = KC_from_dimensionless_K(2.0, 1000.0)
    assert Kc == pytest.approx(2.0 * 1e5 / (R * 1000.0))
