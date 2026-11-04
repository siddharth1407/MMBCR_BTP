from parameters import BenchmarkCase
from properties import Le2023Properties
from thermodynamics import Le2023Thermo
from reaction_only_solver import solve_reaction_only


def test_reaction_only_benchmark_is_physical():
    result = solve_reaction_only(
        BenchmarkCase(), Le2023Properties(), Le2023Thermo(), n_eval=40
    )
    assert result.X_out > 0.0
    assert result.X_out < 100.0
    assert result.J_ch4[-1] < result.J_ch4[0]
    assert result.J_h2[-1] > result.J_h2[0]
    assert result.P[-1] < result.P[0]
    assert result.a_g.min() > 0.0
    assert result.d_b.min() > 0.0
