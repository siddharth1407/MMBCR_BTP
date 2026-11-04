import pytest
import numpy as np

from hydrodynamics import axial_dispersion_raw, axial_dispersion_smoothed


def test_axial_dispersion_scaling_matches_rendered_eq3():
    # The supplied rendered Eq. (3) is D_ax = 50*u_b^3*D^1.5.
    ub = 0.1984217478
    D = 0.03
    expected = 50.0 * ub**3 * D**1.5
    value = axial_dispersion_raw(ub, D)
    assert np.isclose(value, expected, rtol=1e-12, atol=1e-15)
    # Bench-scale magnitude should be ~2e-3 m^2/s for the inlet state.
    assert 1.5e-3 < value < 3.0e-3


def test_axial_dispersion_smoothing_starts_near_zero():
    ub = 0.1984217478
    D = 0.03
    H = 1.15
    val0 = axial_dispersion_smoothed(ub, ub, D, 0.0, H, phi_th=0.001)
    valH = axial_dispersion_smoothed(ub, ub, D, H, H, phi_th=0.001)
    assert val0 >= 0.0
    assert val0 < 1e-10
    assert np.isclose(valH, axial_dispersion_raw(ub, D), rtol=1e-12, atol=1e-12)


def test_drift_velocity_branch_matches_rendered_bench_equation():
    # Bench case falls in N_mu <= 2.2e-3 and D_H* <= 30, so Eq. (29)
    # of the supplied paper is the applicable branch.
    from hydrodynamics import drift_velocity_1
    v = drift_velocity_1(
        u_plus=0.0030604637618977435,
        rho_l=9046.284096402256,
        rho_g=0.4667117743319254,
        mu_l=0.0019717815005498847,
        sigma_l=0.3419343240124873,
        D=0.03,
    )
    rho_ratio = 9046.284096402256 / 0.4667117743319254
    D_H = 15.275198108490294
    N_mu = 0.0007999910451327257
    expected = 0.0019 * D_H**0.809 * rho_ratio**(-0.157) * N_mu**(-0.562)
    assert v == pytest.approx(expected, rel=1e-12)
