
import numpy as np

from model_context import ModelContext
from parameters import BenchmarkCase
from solver import default_initial_guess
from balances import model_rhs
from tests.mock_models import MockProperties, MockThermo


def test_reference_rhs_shape_with_mock_properties():
    params = BenchmarkCase()
    ctx = ModelContext(
        properties=MockProperties(),
        thermo=MockThermo(),
    )

    h, y = default_initial_guess(params, n_points=3)
    yp = model_rhs(h[1], y[:, 1], ctx, params)

    assert yp.shape == (10,)
    assert np.all(np.isfinite(yp))
