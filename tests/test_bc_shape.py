import numpy as np
from model_context import ModelContext
from parameters import BenchmarkCase
from solver import default_initial_guess
from boundary_conditions import bc
from tests.mock_models import MockProperties, MockThermo


def test_bc_has_ten_residuals_with_mock_models():
    params = BenchmarkCase()
    ctx = ModelContext(properties=MockProperties(), thermo=MockThermo())
    h, y = default_initial_guess(params, n_points=3)
    residual = bc(y[:, 0], y[:, -1], ctx, params)
    assert residual.shape == (10,)
    assert np.all(np.isfinite(residual))
