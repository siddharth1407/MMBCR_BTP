import numpy as np

from parameters import BenchmarkCase
from state import N_STATE
from solver import default_initial_guess
from boundary_conditions import initial_mass_fractions


def test_state_dimension():
    params = BenchmarkCase()
    h, y = default_initial_guess(params)
    assert y.shape[0] == N_STATE
    assert y.shape[1] == len(h)


def test_benchmark_mass_fraction_conversion():
    params = BenchmarkCase()
    x = initial_mass_fractions(params)
    assert abs(sum(x.values()) - 1.0) < 1e-12
    assert x["CH4"] > 0
    assert x["Ar"] > 0
