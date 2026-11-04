"""Real-Data Inverse Kinetics: Ea Discovery from Upham et al. (Science 2017).

This script applies the physics-informed Trust-Region Reflective (TRF) inverse
algorithm to the 12 real experimental data points published by Upham et al. in
Science 2017 (T=1040°C, P=2 bar, 80/20 CH4/Ar, D=0.03 m, H=1.15 m) to
discover the catalytic activation energy Ea_cat directly from measured data,
without using any synthetically generated sensor data.

This directly addresses the circular validation weakness in pinn_inverse_kinetics.py
(which uses the true Ea to generate its own synthetic training data).

Mathematical Framework:
    - Forward Physics: Non-linear two-phase MMBCR mass balance (Le et al. 2023).
    - Inverse Optimization: TRF least-squares on real experimental residuals.
    - Uncertainty Quantification: Fisher Information Matrix → 95% CI.
    - Fit Metrics: R², RMSE, MAE, MAPE against real Upham data.

References:
    Upham et al., Science 358, 917–921 (2017).
    Le et al., Chemical Engineering Journal (2023).
    Catalan & Rezaei (2020) – Ea_cat = 208 kJ/mol.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

# Add src to path if running from project root
_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from parameters import BenchmarkCase, KineticParameters
from properties import Le2023Properties
from thermodynamics import Le2023Thermo
from reaction_only_solver import solve_reaction_only

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ---------------------------------------------------------------------------
# Upham et al. (Science 2017) experimental data: 12 real measured points
# T = 1040°C, P = 2 bar, 80/20 CH4/Ar feed, D = 0.03 m, H = 1.15 m
# Columns: [height_m, conversion_%]
# ---------------------------------------------------------------------------
UPHAM_1040C_EXP = np.array([
    [0.03,  4.5],
    [0.08, 13.8],
    [0.15, 25.2],
    [0.24, 41.0],
    [0.36, 53.5],
    [0.45, 62.0],
    [0.53, 66.5],
    [0.63, 72.5],
    [0.74, 78.0],
    [0.82, 82.0],
    [0.93, 85.5],
    [1.13, 89.8],
], dtype=float)

# Reactor conditions matching Upham experiment exactly (BenchmarkCase defaults)
CASE = BenchmarkCase()          # T_L_in_C=1040, P_in=2e5, D=0.03, H=1.15
PROPS = Le2023Properties()
THERMO = Le2023Thermo()

# Reference values
LITERATURE_EA_KJ = 208.0        # Catalan & Rezaei / Le et al. 2023 (kJ/mol)
INITIAL_GUESS_EA_KJ = 140.0     # 32.7% off from literature – deliberately poor

# Experimental arrays
EXP_HEIGHTS = UPHAM_1040C_EXP[:, 0]   # m
EXP_X_PCT   = UPHAM_1040C_EXP[:, 1]   # % conversion (0-100 scale)


def _forward_solve(ea_kj: float, n_eval: int = 200) -> np.ndarray:
    """Run forward MMBCR solver and return conversion (%) interpolated at
    experimental measurement heights."""
    kin = KineticParameters(Ea_cat=ea_kj * 1e3)
    sol = solve_reaction_only(CASE, PROPS, THERMO, kinetics=kin, n_eval=n_eval)
    return np.interp(EXP_HEIGHTS, sol.h, sol.X_ch4)


def discover_ea_from_real_data(
    init_ea_kj: float = INITIAL_GUESS_EA_KJ,
    verbose: bool = True,
) -> dict:
    """Run TRF inverse algorithm on real Upham data to discover Ea_cat.

    Parameters
    ----------
    init_ea_kj : float
        Starting guess for Ea in kJ/mol (default 140 kJ/mol, 32.7% off).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict
        Summary dict with discovered Ea, CI, fit metrics, and file paths.
    """
    n_data = len(EXP_HEIGHTS)

    if verbose:
        print("=" * 72)
        print("  REAL-DATA INVERSE KINETICS  –  Upham et al. Science 2017")
        print("=" * 72)
        print(f"  Data source   : Upham et al., Science 358, 917-921 (2017)")
        print(f"  Conditions    : T={CASE.T_L_in_C:.0f}°C, P={CASE.P_in/1e5:.0f} bar,"
              f" 80/20 CH4/Ar, D={CASE.D:.3f} m, H={CASE.H:.2f} m")
        print(f"  Data points   : {n_data} real measured (height, conversion) pairs")
        print(f"  Literature Ea : {LITERATURE_EA_KJ:.1f} kJ/mol  (Catalan & Rezaei 2020)")
        print(f"  Initial guess : {init_ea_kj:.1f} kJ/mol  "
              f"(error: {abs(init_ea_kj - LITERATURE_EA_KJ)/LITERATURE_EA_KJ*100:.1f}%)")
        print("-" * 72)

    # ------------------------------------------------------------------
    # Objective function: residuals r_i = X_model(z_i) - X_exp(z_i)
    # ------------------------------------------------------------------
    eval_count = 0
    history: list[dict] = []

    def objective(params: np.ndarray) -> np.ndarray:
        nonlocal eval_count
        ea_kj = float(params[0])
        pred_X = _forward_solve(ea_kj, n_eval=150)
        res = pred_X - EXP_X_PCT
        eval_count += 1
        history.append({
            "eval": eval_count,
            "ea_kj": round(ea_kj, 4),
            "cost": float(0.5 * np.dot(res, res)),
        })
        if verbose and eval_count % 5 == 1:
            print(f"    [{eval_count:3d}] Ea={ea_kj:7.2f} kJ/mol  cost={history[-1]['cost']:.4f}")
        return res

    # ------------------------------------------------------------------
    # Trust-Region Reflective optimisation
    # diff_step=0.5 kJ/mol for stable Jacobian with real (noisy) data
    # ------------------------------------------------------------------
    result = least_squares(
        objective,
        x0=[init_ea_kj],
        bounds=([80.0], [350.0]),
        diff_step=0.5,       # larger step for real noisy data (vs 0.1 for synthetic)
        ftol=1e-8,
        xtol=1e-8,
        method="trf",
    )

    discovered_ea_kj = float(result.x[0])
    discovered_ea_j  = discovered_ea_kj * 1e3

    abs_err_kj  = abs(discovered_ea_kj - LITERATURE_EA_KJ)
    rel_err_pct = abs_err_kj / LITERATURE_EA_KJ * 100.0

    if verbose:
        print(f"\n  Optimizer converged in {result.nfev} evaluations  "
              f"(status={result.status})")

    # ------------------------------------------------------------------
    # Uncertainty Quantification via Fisher Information Matrix (FIM)
    # ------------------------------------------------------------------
    J_mat = result.jac                        # shape (n_data, 1)
    rss   = float(np.dot(result.fun, result.fun))
    dof   = max(1, n_data - 1)
    s2    = rss / dof                         # residual variance estimate
    fim   = float((J_mat.T @ J_mat).item())   # scalar for 1-param problem
    var   = s2 / fim if fim > 0 else np.inf
    std_err_kj = float(np.sqrt(var))
    ci_95_kj   = 1.96 * std_err_kj
    lit_in_ci  = abs(discovered_ea_kj - LITERATURE_EA_KJ) <= ci_95_kj

    # ------------------------------------------------------------------
    # Final forward solve with discovered Ea → full profile + fit metrics
    # ------------------------------------------------------------------
    discovered_kin = KineticParameters(Ea_cat=discovered_ea_j)
    sol_disc = solve_reaction_only(CASE, PROPS, THERMO, kinetics=discovered_kin, n_eval=300)
    pred_X_at_data = np.interp(EXP_HEIGHTS, sol_disc.h, sol_disc.X_ch4)

    residuals = pred_X_at_data - EXP_X_PCT
    ss_res  = float(np.dot(residuals, residuals))
    ss_tot  = float(np.sum((EXP_X_PCT - np.mean(EXP_X_PCT)) ** 2))
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse    = float(np.sqrt(np.mean(residuals ** 2)))
    mae     = float(np.mean(np.abs(residuals)))
    mape    = float(np.mean(np.abs(residuals / EXP_X_PCT)) * 100.0)

    if verbose:
        print()
        print("  RESULTS")
        print("  " + "-" * 68)
        print(f"  Discovered Ea_cat : {discovered_ea_kj:.3f} kJ/mol")
        print(f"  95% CI            : ± {ci_95_kj:.3f} kJ/mol  "
              f"  → [{discovered_ea_kj - ci_95_kj:.3f}, {discovered_ea_kj + ci_95_kj:.3f}]")
        print(f"  Literature Ea     : {LITERATURE_EA_KJ:.1f} kJ/mol  "
              f"(Catalan & Rezaei 2020)")
        print(f"  Absolute error    : {abs_err_kj:.3f} kJ/mol")
        print(f"  Relative error    : {rel_err_pct:.2f}%")
        print(f"  Literature in CI  : {lit_in_ci}")
        print()
        print(f"  R²   (real data)  : {r2:.4f}")
        print(f"  RMSE              : {rmse:.3f} percentage-points")
        print(f"  MAE               : {mae:.3f} percentage-points")
        print(f"  MAPE              : {mape:.2f}%")
        print("=" * 72)

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    out_dir = Path("results/ai")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "real_data_inverse_kinetics_results.json"

    summary = {
        "description": (
            "Inverse kinetic parameter discovery from real Upham et al. "
            "(Science 2017) experimental data – no synthetic data used."
        ),
        "data_source": "Upham et al., Science 358, 917-921 (2017)",
        "reactor_conditions": {
            "temperature_C": float(CASE.T_L_in_C),
            "pressure_bar": float(CASE.P_in / 1e5),
            "feed_composition": "80/20 CH4/Ar",
            "diameter_m": float(CASE.D),
            "height_m": float(CASE.H),
        },
        "experimental_data": {
            "n_points": int(n_data),
            "heights_m": EXP_HEIGHTS.tolist(),
            "conversions_pct": EXP_X_PCT.tolist(),
            "note": "Real measured values – no added noise",
        },
        "inverse_algorithm": {
            "method": "Trust-Region Reflective (TRF) least-squares",
            "initial_guess_kJ_mol": float(init_ea_kj),
            "initial_error_pct": float(abs(init_ea_kj - LITERATURE_EA_KJ) / LITERATURE_EA_KJ * 100),
            "bounds_kJ_mol": [80.0, 350.0],
            "diff_step_kJ_mol": 0.5,
            "function_evaluations": int(result.nfev),
            "optimizer_status": int(result.status),
        },
        "parameter_estimation": {
            "parameter": "Ea_cat (Catalytic Activation Energy)",
            "discovered_kJ_mol": round(discovered_ea_kj, 4),
            "std_error_kJ_mol": round(std_err_kj, 4),
            "ci_95_kJ_mol": round(ci_95_kj, 4),
            "confidence_interval_95_pct": [
                round(discovered_ea_kj - ci_95_kj, 4),
                round(discovered_ea_kj + ci_95_kj, 4),
            ],
            "literature_value_kJ_mol": float(LITERATURE_EA_KJ),
            "absolute_error_kJ_mol": round(abs_err_kj, 4),
            "relative_error_pct": round(rel_err_pct, 4),
            "literature_value_enclosed_in_95_CI": bool(lit_in_ci),
        },
        "fit_metrics_vs_real_data": {
            "R_squared": round(r2, 6),
            "RMSE_pct_points": round(rmse, 4),
            "MAE_pct_points": round(mae, 4),
            "MAPE_pct": round(mape, 4),
            "RSS": round(ss_res, 6),
        },
        "optimisation_history": history,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n  JSON saved → {json_path}")

    # ------------------------------------------------------------------
    # Save comparison figure (matplotlib)
    # ------------------------------------------------------------------
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / "real_data_inverse_kinetics.png"

    if _HAS_MPL:
        _make_figure(
            sol_disc=sol_disc,
            init_ea_kj=init_ea_kj,
            discovered_ea_kj=discovered_ea_kj,
            ci_95_kj=ci_95_kj,
            pred_X_at_data=pred_X_at_data,
            history=history,
            png_path=png_path,
            r2=r2,
            rmse=rmse,
        )
        if verbose:
            print(f"  PNG saved  → {png_path}")
    else:
        if verbose:
            print("  (matplotlib not available – PNG not generated)")

    summary["output_files"] = {
        "json": str(json_path),
        "png": str(png_path) if _HAS_MPL else None,
    }
    return summary


def _make_figure(
    sol_disc,
    init_ea_kj: float,
    discovered_ea_kj: float,
    ci_95_kj: float,
    pred_X_at_data: np.ndarray,
    history: list[dict],
    png_path: Path,
    r2: float,
    rmse: float,
) -> None:
    """Generate a two-panel comparison figure and save as PNG."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Real-Data Inverse Kinetics  –  Upham et al. (Science 2017)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ----- Panel 1: Conversion profiles -----
    # Initial guess curve
    init_sol = solve_reaction_only(
        CASE, PROPS, THERMO,
        kinetics=KineticParameters(Ea_cat=init_ea_kj * 1e3),
        n_eval=200,
    )
    ax1.plot(
        init_sol.h, init_sol.X_ch4,
        "r--", lw=1.8, label=f"Initial guess  Ea={init_ea_kj:.0f} kJ/mol",
    )

    # Discovered curve
    ax1.plot(
        sol_disc.h, sol_disc.X_ch4,
        "b-", lw=2.2, label=f"Discovered  Ea={discovered_ea_kj:.2f} kJ/mol",
    )

    # Real data scatter
    ax1.scatter(
        EXP_HEIGHTS, EXP_X_PCT,
        s=60, c="k", marker="o", zorder=5,
        label=f"Upham 2017 data  (N={len(EXP_HEIGHTS)})",
    )

    # Text box with key results
    textstr = (
        f"Discovered Ea = {discovered_ea_kj:.2f} ± {ci_95_kj:.2f} kJ/mol\n"
        f"Literature  Ea = {LITERATURE_EA_KJ:.1f} kJ/mol\n"
        f"R² = {r2:.4f}   RMSE = {rmse:.2f} pp"
    )
    props_box = dict(boxstyle="round", facecolor="lightyellow", alpha=0.85)
    ax1.text(
        0.03, 0.97, textstr,
        transform=ax1.transAxes, fontsize=8.5,
        verticalalignment="top", bbox=props_box,
    )

    ax1.set_xlabel("Reactor height (m)", fontsize=10)
    ax1.set_ylabel("CH₄ conversion (%)", fontsize=10)
    ax1.set_title("Conversion Profile vs. Real Upham Data", fontsize=10)
    ax1.legend(fontsize=8.5, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, CASE.H)
    ax1.set_ylim(0, 100)

    # ----- Panel 2: Optimisation trajectory -----
    evals = [h["eval"] for h in history]
    ea_vals = [h["ea_kj"] for h in history]

    ax2.plot(evals, ea_vals, "bo-", lw=1.6, ms=4, label="Estimated Ea (per step)")
    ax2.axhline(LITERATURE_EA_KJ, color="green", ls="--", lw=2,
                label=f"Literature value ({LITERATURE_EA_KJ:.0f} kJ/mol)")
    ax2.axhline(init_ea_kj, color="red", ls=":", lw=1.5,
                label=f"Initial guess ({init_ea_kj:.0f} kJ/mol)")
    ax2.axhline(discovered_ea_kj, color="blue", ls="-.", lw=1.5,
                label=f"Converged ({discovered_ea_kj:.2f} kJ/mol)")

    # 95% CI band around converged value
    ax2.axhspan(
        discovered_ea_kj - ci_95_kj,
        discovered_ea_kj + ci_95_kj,
        alpha=0.15, color="blue", label="95% CI",
    )

    ax2.set_xlabel("Function evaluation step", fontsize=10)
    ax2.set_ylabel("Ea_cat (kJ/mol)", fontsize=10)
    ax2.set_title("TRF Parameter Estimation Trajectory", fontsize=10)
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    discover_ea_from_real_data(
        init_ea_kj=INITIAL_GUESS_EA_KJ,
        verbose=True,
    )


if __name__ == "__main__":
    main()
