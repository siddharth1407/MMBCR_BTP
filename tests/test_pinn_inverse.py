"""Test suite for Physics-Informed Inverse Kinetic Parameter Discovery."""
import json
from pathlib import Path
import unittest
import numpy as np

from pinn_inverse_kinetics import solve_inverse_kinetics_scipy, TRUE_EA_CAT


def test_inverse_kinetics_convergence():
    """Verify that inverse discovery converges to the true activation energy within confidence interval."""
    summary = solve_inverse_kinetics_scipy(
        n_sensors=7,
        noise_std=0.02,
        seed=42,
        init_ea_kj=120.0,
        verbose=False,
    )

    pe = summary["parameter_estimation"]
    fit = summary["fit_metrics"]

    # True value should be 208 kJ/mol
    assert abs(pe["true_value_kJ_mol"] - 208.0) < 1e-4

    # Relative discovery error must be under 1.5% despite noisy measurements
    assert pe["relative_error_pct"] < 1.5, f"Error {pe['relative_error_pct']}% exceeds 1.5%"

    # True value must be enclosed within the 95% confidence interval
    assert pe["true_value_enclosed_in_95_ci"] is True

    # Goodness-of-fit R^2 must be > 0.99
    assert fit["r_squared"] > 0.99, f"R^2 {fit['r_squared']} is below 0.99"

    # Files must be written
    assert Path("results/ai/inverse_kinetics_summary.json").is_file()
    assert Path("results/ai/inverse_kinetics_parity.csv").is_file()


if __name__ == "__main__":
    test_inverse_kinetics_convergence()
    print("test_inverse_kinetics_convergence passed!")
