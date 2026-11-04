from parameters import BenchmarkCase, KineticParameters
from model_context import ModelContext
from properties import Le2023Properties
from thermodynamics import Le2023Thermo
from full_coupled_picard import run


def test_stage4_full_coupled_smoke():
    p = BenchmarkCase()
    ctx = ModelContext(
        properties=Le2023Properties(),
        thermo=Le2023Thermo(),
        kinetics=KineticParameters(),
    )
    r = run(p, ctx, n_points=40, max_iter=8, relax=0.55, tol=1e-2, verbose=False)
    assert r.converged
    assert 0.0 < r.X_out < 100.0
    assert r.P[-1] > 0.0
    assert r.T_g[-1] > p.T_G_in
    assert r.T_l[-1] > 1200.0
    assert abs(r.Q_l[-1]) < 1e-6
