"""Automated test suite for Multi-Condition MMBCR Digital Twin."""
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from multi_condition_surrogate import (
    generate_latin_hypercube_samples,
    scale_samples_to_physics,
    simulate_operating_condition,
)
from properties import Le2023Properties
from thermodynamics import Le2023Thermo


def test_latin_hypercube_sampling_shape_and_bounds():
    """Verify that LHS generates valid uniform distributions in [0, 1]."""
    samples = generate_latin_hypercube_samples(n_samples=50, seed=42)
    assert samples.shape == (50, 6)
    assert (samples >= 0.0).all()
    assert (samples <= 1.0).all()

    df_phys = scale_samples_to_physics(samples)
    assert len(df_phys) == 50
    assert (df_phys["temperature_C"] >= 950.0).all()
    assert (df_phys["temperature_C"] <= 1120.0).all()
    assert (df_phys["height_m"] >= 0.5).all()
    assert (df_phys["height_m"] <= 2.5).all()


def test_simulate_operating_condition():
    """Verify that forward simulation returns realistic physical conversion and yields."""
    props = Le2023Properties()
    thermo = Le2023Thermo()

    sample_row = pd.Series({
        "temperature_C": 1050.0,
        "pressure_bar": 5.0,
        "superficial_velocity_m_s": 0.05,
        "height_m": 1.5,
        "diameter_m": 0.05,
        "methane_mole_fraction": 0.85,
    })

    res = simulate_operating_condition(sample_row, props, thermo)
    assert 10.0 < res["conversion_pct"] < 99.0
    assert res["delta_P_bar"] > 0.0
    assert res["h2_production_kg_day"] > 0.0
    assert res["carbon_production_kg_day"] > 0.0


def test_digital_twin_artifacts_exist():
    """Verify that digital twin dataset, metrics summary, and figures are created."""
    assert Path("results/ai/digital_twin_dataset.csv").is_file()
    assert Path("results/ai/digital_twin_summary.json").is_file()
    assert Path("results/ai/digital_twin_test_parity.csv").is_file()
    assert Path("results/figures/digital_twin_parity.svg").is_file()

    with open("results/ai/digital_twin_summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    # Verification criteria
    assert summary["dataset"]["total_operating_conditions"] == 400
    assert summary["models"]["gradient_boosting"]["r_squared"] > 0.95
    assert summary["performance"]["speedup_factor"] > 10.0


if __name__ == "__main__":
    test_latin_hypercube_sampling_shape_and_bounds()
    test_simulate_operating_condition()
    test_digital_twin_artifacts_exist()
    print("All digital twin unit tests passed!")

