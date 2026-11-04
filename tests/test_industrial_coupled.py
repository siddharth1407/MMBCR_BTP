from dataclasses import replace

try:
    import pytest
except ImportError:
    pytest = None

from parameters import BenchmarkCase, KineticParameters
from properties import Le2023Properties
from thermodynamics import Le2023Thermo
from model_context import ModelContext
from industrial_coupled_solver import solve_industrial_coupled


def test_industrial_coupled_solver_converges_and_matches_literature():
    params = replace(
        BenchmarkCase(),
        D=0.10,
        H=2.11,
        P_in=17.84e5,
        T_G_in_C=800.0,
        T_L_in_C=1050.0,
        T_W_C=1050.0,
        j_CH4_in=0.1606,
        j_H2_in=0.0,
        j_Ar_in=0.0,
        ch4_mole_fraction=1.0,
        h2_mole_fraction=0.0,
        ar_mole_fraction=0.0,
    )
    props = Le2023Properties()
    thermo = Le2023Thermo()
    ctx = ModelContext(properties=props, thermo=thermo, kinetics=KineticParameters())

    res = solve_industrial_coupled(params, ctx, n_points=120, max_iter=10, relax=0.5, tol=3e-3)
    assert res.converged is True
    # Backmixing creates entrance conversion jump of ~10-15% (paper reports ~10%)
    assert 9.0 <= res.X_in <= 16.0
    # Final conversion at H=2.11 m is ~66% (paper reports ~66%)
    assert 64.0 <= res.X_out <= 70.0
    # Pressure drop is approximately 1.7 bar
    delta_P_bar = (res.P[0] - res.P[-1]) / 1e5
    assert 1.5 <= delta_P_bar <= 1.9
