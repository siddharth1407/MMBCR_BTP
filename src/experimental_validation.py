"""Quantitative experimental validation against Upham et al. (Science 2017).

This module digitizes and benchmarks the actual experimental data from
Upham et al. (2017) [Science 358, 917-921] reported in Fig. 4 of Le et al. (2023)
for methane pyrolysis in a molten Ni0.27Bi0.73 bubble column reactor at:
- T = 1040 C
- P = 2.0 bar
- Feed = 80/20 mol% CH4/Ar
- Total melt height H = 1.15 m, Column D = 0.03 m

Calculates:
- Coefficient of Determination (R^2)
- Root Mean Square Error (RMSE in percentage points)
- Mean Absolute Error (MAE in percentage points)
- Mean Absolute Percentage Error (MAPE in %)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from parameters import BenchmarkCase
from properties import Le2023Properties
from thermodynamics import Le2023Thermo
from reaction_only_solver import solve_reaction_only
from heat_transfer_bench import solve_bench_heat_transfer

# Upham et al. (Science 2017) experimental data points at 1040 C
# Height above gas sparger (m) vs. measured methane conversion (%)
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


def run_experimental_validation():
    root = Path(__file__).resolve().parent
    project_root = root.parent
    val_out = project_root / "results" / "validation"
    fig_out = project_root / "results" / "figures"
    val_out.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    params = BenchmarkCase()
    props = Le2023Properties()
    thermo = Le2023Thermo()

    ref = solve_reaction_only(params, props, thermo, n_eval=300)
    heat = solve_bench_heat_transfer(params, props, thermo, n_eval=300)

    h_exp = UPHAM_1040C_EXP[:, 0]
    X_exp = UPHAM_1040C_EXP[:, 1]

    # Interpolate model predictions at experimental measurement heights
    X_model_heat = np.interp(h_exp, heat.h, heat.X_ch4)
    X_model_rxn = np.interp(h_exp, ref.h, ref.X_ch4)

    # Statistical metrics for Heat-Transfer Model (rA + T)
    diff = X_exp - X_model_heat
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((X_exp - np.mean(X_exp))**2)
    r2 = 1.0 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    mape = np.mean(np.abs(diff / X_exp)) * 100.0

    # Statistical metrics for Reaction-Only Model (rA)
    diff_rxn = X_exp - X_model_rxn
    r2_rxn = 1.0 - (np.sum(diff_rxn**2) / ss_tot)
    rmse_rxn = np.sqrt(np.mean(diff_rxn**2))
    mae_rxn = np.mean(np.abs(diff_rxn))

    summary = {
        "experimental_source": "Upham et al. (Science 2017;358:917-921), Fig. 4 in Le et al. (2023)",
        "operating_conditions": {
            "temperature_C": 1040.0,
            "pressure_bar": 2.0,
            "feed_mol_pct": "80% CH4, 20% Ar",
            "reactor_diameter_m": 0.03,
            "reactor_height_m": 1.15,
        },
        "number_of_experimental_points": len(h_exp),
        "heat_transfer_model_metrics": {
            "R2": float(r2),
            "RMSE_pp": float(rmse),
            "MAE_pp": float(mae),
            "MAPE_pct": float(mape),
        },
        "reaction_only_model_metrics": {
            "R2": float(r2_rxn),
            "RMSE_pp": float(rmse_rxn),
            "MAE_pp": float(mae_rxn),
        },
    }

    (val_out / "experimental_validation_summary.json").write_text(json.dumps(summary, indent=2))

    # Export CSV table of experimental points vs model
    rows = []
    for h, e, s, r_val in zip(h_exp, X_exp, X_model_heat, X_model_rxn):
        rows.append({
            "height_m": float(h),
            "X_experimental_pct": float(e),
            "X_model_heat_transfer_pct": float(s),
            "abs_error_heat_transfer_pp": float(abs(e - s)),
            "rel_error_pct": float(100.0 * (s - e) / e),
            "X_model_reaction_only_pct": float(r_val),
            "abs_error_reaction_only_pp": float(abs(e - r_val)),
        })

    with (val_out / "experimental_parity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Export standalone vector SVG figure (Zero dependencies, works without matplotlib)
    fig_out = Path("results/figures")
    fig_out.mkdir(parents=True, exist_ok=True)
    try:
        w, h = 880, 420
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background:#fff; font-family:sans-serif;">']
        svg.append('<text x="440" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#111">Quantitative Experimental Validation (Upham et al. 2017 Science vs. MMBCR Model)</text>')
        svg.append(f'<text x="440" y="48" text-anchor="middle" font-size="12" fill="#555">R² = {r2:.4f} | RMSE = {rmse:.2f} percentage points | MAE = {mae:.2f} pp | MAPE = {mape:.2f}%</text>')

        # Panel 1: Profile along reactor height (x: 70..400, y: 75..345)
        p1_x0, p1_y0, p1_w, p1_h = 70, 75, 330, 270
        svg.append(f'<rect x="{p1_x0}" y="{p1_y0}" width="{p1_w}" height="{p1_h}" fill="#fafafa" stroke="#ccc"/>')
        svg.append(f'<text x="{p1_x0 + p1_w/2}" y="{p1_y0 - 8}" text-anchor="middle" font-size="12" font-weight="bold">Conversion vs. Reactor Height (1040 °C)</text>')
        svg.append(f'<text x="{p1_x0 + p1_w/2}" y="{p1_y0 + p1_h + 35}" text-anchor="middle" font-size="11">Reactor Height, h (m)</text>')
        svg.append(f'<text x="{p1_x0 - 38}" y="{p1_y0 + p1_h/2}" text-anchor="middle" font-size="11" transform="rotate(-90 {p1_x0 - 38} {p1_y0 + p1_h/2})">CH4 Conversion (%)</text>')

        # Model curve (Heat Transfer)
        pts_model = [f"{p1_x0 + (hi/1.15)*p1_w:.1f},{p1_y0 + p1_h - (Xi/100.0)*p1_h:.1f}" for hi, Xi in zip(heat.h, heat.X_ch4)]
        svg.append(f'<polyline points="{" ".join(pts_model)}" fill="none" stroke="#1976d2" stroke-width="2.5"/>')

        # Reaction-only curve
        pts_rxn = [f"{p1_x0 + (hi/1.15)*p1_w:.1f},{p1_y0 + p1_h - (Xi/100.0)*p1_h:.1f}" for hi, Xi in zip(ref.h, ref.X_ch4)]
        svg.append(f'<polyline points="{" ".join(pts_rxn)}" fill="none" stroke="#388e3c" stroke-width="1.8" stroke-dasharray="4,4"/>')

        # Experimental scatter points
        for he, Xe in zip(h_exp, X_exp):
            cx = p1_x0 + (he / 1.15) * p1_w
            cy = p1_y0 + p1_h - (Xe / 100.0) * p1_h
            svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="#e53935" stroke="#111" stroke-width="1"/>')

        # Legend Panel 1
        svg.append(f'<rect x="{p1_x0 + 100}" y="{p1_y0 + p1_h - 75}" width="220" height="65" fill="#fff" opacity="0.9" stroke="#ddd"/>')
        svg.append(f'<line x1="{p1_x0+110}" y1="{p1_y0+p1_h-60}" x2="{p1_x0+135}" y2="{p1_y0+p1_h-60}" stroke="#1976d2" stroke-width="2.5"/>')
        svg.append(f'<text x="{p1_x0+145}" y="{p1_y0+p1_h-56}" font-size="10">Model: Heat Transfer (R²={r2:.4f})</text>')
        svg.append(f'<line x1="{p1_x0+110}" y1="{p1_y0+p1_h-42}" x2="{p1_x0+135}" y2="{p1_y0+p1_h-42}" stroke="#388e3c" stroke-width="2" stroke-dasharray="3,3"/>')
        svg.append(f'<text x="{p1_x0+145}" y="{p1_y0+p1_h-38}" font-size="10">Model: Reaction Only (R²={r2_rxn:.4f})</text>')
        svg.append(f'<circle cx="{p1_x0+122}" cy="{p1_y0+p1_h-24}" r="4" fill="#e53935" stroke="#111" stroke-width="0.8"/>')
        svg.append(f'<text x="{p1_x0+145}" y="{p1_y0+p1_h-21}" font-size="10">Experiment (Upham et al. 2017)</text>')

        # Panel 2: Parity Plot (x: 490..820, y: 75..345)
        p2_x0, p2_y0, p2_w, p2_h = 500, 75, 320, 270
        svg.append(f'<rect x="{p2_x0}" y="{p2_y0}" width="{p2_w}" height="{p2_h}" fill="#fafafa" stroke="#ccc"/>')
        svg.append(f'<text x="{p2_x0 + p2_w/2}" y="{p2_y0 - 8}" text-anchor="middle" font-size="12" font-weight="bold">Parity: Simulated vs. Experimental</text>')
        svg.append(f'<text x="{p2_x0 + p2_w/2}" y="{p2_y0 + p2_h + 35}" text-anchor="middle" font-size="11">Experimental Conversion, X_exp (%)</text>')
        svg.append(f'<text x="{p2_x0 - 38}" y="{p2_y0 + p2_h/2}" text-anchor="middle" font-size="11" transform="rotate(-90 {p2_x0 - 38} {p2_y0 + p2_h/2})">Simulated Conversion, X_sim (%)</text>')

        # 1:1 diagonal line
        svg.append(f'<line x1="{p2_x0}" y1="{p2_y0+p2_h}" x2="{p2_x0+p2_w}" y2="{p2_y0}" stroke="#222" stroke-width="2"/>')
        # +/- 5% error lines
        svg.append(f'<line x1="{p2_x0}" y1="{p2_y0+p2_h}" x2="{p2_x0+p2_w*0.95:.1f}" y2="{p2_y0}" stroke="#888" stroke-dasharray="3,3"/>')
        svg.append(f'<line x1="{p2_x0+p2_w*0.05:.1f}" y1="{p2_y0+p2_h}" x2="{p2_x0+p2_w}" y2="{p2_y0}" stroke="#888" stroke-dasharray="3,3"/>')

        # Parity points
        for Xe, Xs in zip(X_exp, X_model_heat):
            cx = p2_x0 + (Xe / 100.0) * p2_w
            cy = p2_y0 + p2_h - (Xs / 100.0) * p2_h
            svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="#1976d2" stroke="#111" stroke-width="1"/>')

        # Legend Panel 2
        svg.append(f'<rect x="{p2_x0 + 15}" y="{p2_y0 + 15}" width="180" height="50" fill="#fff" opacity="0.9" stroke="#ddd"/>')
        svg.append(f'<line x1="{p2_x0+25}" y1="{p2_y0+30}" x2="{p2_x0+50}" y2="{p2_y0+30}" stroke="#222" stroke-width="2"/>')
        svg.append(f'<text x="{p2_x0+60}" y="{p2_y0+34}" font-size="10">Perfect Parity (1:1)</text>')
        svg.append(f'<circle cx="{p2_x0+37}" cy="{p2_y0+48}" r="4" fill="#1976d2" stroke="#111" stroke-width="0.8"/>')
        svg.append(f'<text x="{p2_x0+60}" y="{p2_y0+52}" font-size="10">Simulated (RMSE={rmse:.2f} pp)</text>')

        svg.append("</svg>")
        svg_file = fig_out / "experimental_parity_plot.svg"
        with open(svg_file, "w", encoding="utf-8") as f:
            f.write("\n".join(svg))
        print(f"Vector SVG saved: {svg_file}")
    except Exception as e:
        print(f"Warning: could not save SVG figure: {e}")

    # Plot parity and comparison curves if matplotlib is available
    if plt is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

        # Panel 1: Profile along reactor height
        ax1.plot(heat.h, heat.X_ch4, "b-", linewidth=2.0, label=f"Model: Heat Transfer (R²={r2:.4f})")
        ax1.plot(ref.h, ref.X_ch4, "g--", linewidth=1.8, label=f"Model: Reaction Only (R²={r2_rxn:.4f})")
        ax1.scatter(h_exp, X_exp, color="red", s=65, zorder=5, edgecolors="black", linewidth=1.2, label="Experiment (Upham et al. 2017)")
        ax1.set_xlabel("Reactor height, h (m)", fontsize=11)
        ax1.set_ylabel("CH4 conversion (%)", fontsize=11)
        ax1.set_title("Experimental Validation: Conversion vs. Height (1040 °C)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower right")

        # Panel 2: Parity plot (Simulated vs Experimental)
        diag = np.linspace(0, 100, 100)
        ax2.plot(diag, diag, "k-", linewidth=1.5, label="Perfect Parity (1:1)")
        ax2.plot(diag, diag * 1.05, "k--", alpha=0.5, label="±5% Bounds")
        ax2.plot(diag, diag * 0.95, "k--", alpha=0.5)
        ax2.scatter(X_exp, X_model_heat, color="blue", s=70, edgecolors="black", linewidth=1.2, zorder=5, label=f"Simulated Points (RMSE={rmse:.2f} pp)")
        ax2.set_xlabel("Experimental Conversion, X_exp (%)", fontsize=11)
        ax2.set_ylabel("Simulated Conversion, X_sim (%)", fontsize=11)
        ax2.set_title("Parity Plot: Simulated vs. Experimental Conversion", fontsize=12)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig(fig_out / "experimental_parity_plot.png", dpi=220)
        plt.close(fig)

    print("Experimental validation executed successfully.")
    print(f"  R^2:   {r2:.4f}")
    print(f"  RMSE:  {rmse:.2f} percentage points")
    print(f"  MAE:   {mae:.2f} percentage points")
    print(f"  MAPE:  {mape:.2f} %")


if __name__ == "__main__":
    run_experimental_validation()
