import numpy as np

from heat_transfer_bench import solve_bench_heat_transfer
from parameters import BenchmarkCase
from properties import Le2023Properties
from thermodynamics import Le2023Thermo


def test_heat_transfer_bench_runs_and_heats_gas():
    result = solve_bench_heat_transfer(
        BenchmarkCase(),
        Le2023Properties(),
        Le2023Thermo(),
        n_eval=120,
    )
    assert result.h.size == 120
    assert result.X_out > 80.0
    assert result.T_g[0] < result.T_g[-1]
    assert result.T_g[-1] > 1300.0
    assert np.all(result.J_ch4 > 0.0)
    assert np.all(result.J_h2 >= 0.0)
    assert np.all(result.P > 0.0)
