from pathlib import Path
import pandas as pd


def test_final_ai_artifacts_exist():
    out = Path('results/ai')
    assert (out / 'heat_transfer_validation.csv').exists()
    assert (out / 'axial_dispersion_validation.csv').exists()
    assert (out / 'reaction_pinn_model.npz').exists()
    assert (out / 'heat_transfer_model.npz').exists()
    assert (out / 'axial_model.npz').exists()

    ht = pd.read_csv(out / 'heat_transfer_validation.csv')
    ax = pd.read_csv(out / 'axial_dispersion_validation.csv')
    assert len(ht) > 300
    assert len(ax) > 200
    assert {'h_m','X_AI_pct','X_ref_pct'} <= set(ht.columns)
    assert {'h_m','X_AI_pct','X_ref_pct'} <= set(ax.columns)

    comp = pd.read_csv(out / 'ai_results_summary_stage13.csv')
    assert len(comp) == 4
