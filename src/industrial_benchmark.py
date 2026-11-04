"""Industrial-scale Le et al. (2023) benchmark and hydrodynamic audit.

This module validates and benchmarks the single-tube industrial case:
D=0.1 m, H=2.11 m, 1050 C liquid/wall temperature, 800 C gas inlet,
17.84 bar inlet pressure, and j_CH4,in = 0.1606 kg/m2/s.

It independently solves:
1. Reaction-only plug-flow model (isothermal)
2. Non-isothermal heat-transfer model (with gas heating)
3. Fully coupled reaction + heat transfer + axial-dispersion model with Danckwerts BCs
"""
from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from parameters import BenchmarkCase, KineticParameters
from properties import Le2023Properties
from thermodynamics import Le2023Thermo
from model_context import ModelContext
from reaction_only_solver import solve_reaction_only
from heat_transfer_bench import solve_bench_heat_transfer
from industrial_coupled_solver import solve_industrial_coupled


def main():
    root = Path(__file__).resolve().parent
    project_root = root.parent
    out = project_root / "results" / "industrial"
    fig_out = project_root / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    params = replace(
        BenchmarkCase(),
        D=0.10,
        H=2.11,
        P_in=17.84e5,
        T_G_in_C=800.0,
        T_L_in_C=1050.0,
        T_W_C=1050.0,
        j_CH4_in=0.1606,
        j_H2_in=0.0,
        j_Ar_in=0.0,
        ch4_mole_fraction=1.0,
        h2_mole_fraction=0.0,
        ar_mole_fraction=0.0,
    )
    props = Le2023Properties()
    thermo = Le2023Thermo()
    ctx = ModelContext(properties=props, thermo=thermo, kinetics=KineticParameters())

    print("Running industrial reaction-only solver...")
    reaction = solve_reaction_only(params, props, thermo, n_eval=260)

    print("Running industrial heat-transfer solver...")
    heat = solve_bench_heat_transfer(params, props, thermo, n_eval=260)

    print("Running industrial fully coupled axial-dispersion solver...")
    full_status = {"attempted": True, "success": False, "message": None}
    try:
        full = solve_industrial_coupled(
            params, ctx, n_points=180, max_iter=12, relax=0.5, tol=2e-3, verbose=True
        )
        full_status.update({
            "success": bool(full.converged),
            "message": full.message,
            "X_in_pct": float(full.X_in),
            "X_out_pct": float(full.X_out),
            "P_out_bar": float(full.P[-1] / 1e5),
            "aG_out_pct": float(full.a_g[-1] * 100.0),
            "uG_out_mm_s": float(full.u_g[-1] * 1000.0),
            "Dax_out_m2_s": float(full.D_ax[-1]),
        })
    except Exception as exc:
        full = None
        full_status["message"] = str(exc)

    # Paper-reported industrial comparison targets
    paper = {
        "X_out_pct_no_axial_dispersion": 80.0,
        "X_out_pct_with_axial_dispersion": 66.0,
        "aG_at_80_pct_pct": 14.3,
        "uG_at_80_pct_mm_s": 155.0,
        "db_at_80_pct_mm": 2.79,
        "as_at_80_pct_m2_m3": 2152.0,
        "H_m": 2.11,
        "D_m": 0.10,
    }

    rows = []
    for name, model_value, target, note in [
        ("reaction_only_conversion_pct", reaction.X_out, paper["X_out_pct_no_axial_dispersion"], "paper reports ~80% at 2.11 m"),
        ("heat_transfer_conversion_pct", heat.X_out, paper["X_out_pct_no_axial_dispersion"], "paper reports ~80% at 2.11 m"),
        ("reaction_only_outlet_pressure_bar", reaction.P[-1] / 1e5, 16.0, "paper figure shows approximately 16 bar class pressure at outlet"),
        ("reaction_only_outlet_gas_velocity_mm_s", reaction.u_g[-1] * 1000, paper["uG_at_80_pct_mm_s"], "reported at XM=80%"),
        ("reaction_only_outlet_bubble_diameter_mm", reaction.d_b[-1] * 1000, paper["db_at_80_pct_mm"], "reported at XM=80%"),
        ("reaction_only_outlet_specific_area_m2_m3", reaction.a_s[-1], paper["as_at_80_pct_m2_m3"], "reported at XM=80%"),
    ]:
        rows.append({
            "metric": name,
            "model_value": float(model_value),
            "paper_reference": float(target),
            "relative_error_pct": float(100.0 * (model_value - target) / target),
            "note": note,
        })

    if full is not None:
        for name, model_val, target_val, note_str in [
            ("coupled_axial_dispersion_conversion_pct", full.X_out, paper["X_out_pct_with_axial_dispersion"], "paper reports ~66% at 2.11 m with axial dispersion"),
            ("coupled_axial_dispersion_inlet_jump_pct", full.X_in, 10.0, "paper Fig. 12 shows ~10% conversion jump at h=0 due to backmixing"),
            ("coupled_outlet_pressure_bar", full.P[-1] / 1e5, 16.0, "paper reports ~16 bar outlet pressure (deltaP ~1.7 bar)"),
        ]:
            rows.append({
                "metric": name,
                "model_value": float(model_val),
                "paper_reference": float(target_val),
                "relative_error_pct": float(100.0 * (model_val - target_val) / target_val),
                "note": note_str,
            })

    with (out / "industrial_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for label, res in [("reaction_only", reaction), ("heat_transfer", heat)]:
        np.savetxt(
            out / f"{label}_profile.csv",
            np.column_stack([
                res.h, res.X_ch4, res.P, res.a_g, res.u_g, res.d_b, res.a_s,
                res.r_non, res.r_cat, res.r_A,
            ]),
            delimiter=",",
            header="h_m,X_CH4_pct,P_Pa,a_G,u_G_m_s,d_b_m,a_s_m2_m3,r_non,r_cat,r_A",
            comments="",
        )

    if full is not None:
        np.savetxt(
            out / "full_coupled_profile.csv",
            np.column_stack([
                full.h, full.X_ch4, full.P, full.T_g - 273.15, full.T_l - 273.15,
                full.a_g, full.u_g, full.d_b, full.a_s, full.D_ax,
            ]),
            delimiter=",",
            header="h_m,X_CH4_pct,P_Pa,T_G_C,T_L_C,a_G,u_G_m_s,d_b_m,a_s_m2_m3,D_ax_m2_s",
            comments="",
        )

    # Export standalone vector SVG figure (Zero dependencies)
    try:
        w, h = 650, 420
        x0, y0, pw, ph = 70, 60, 540, 310
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background:#fff; font-family:sans-serif;">']
        svg.append('<text x="325" y="28" text-anchor="middle" font-size="15" font-weight="bold" fill="#111">Industrial Scale-Up MMBCR (D=0.1 m, H=2.11 m, 17.8 bar, 1050 °C)</text>')
        svg.append(f'<text x="325" y="46" text-anchor="middle" font-size="11" fill="#555">Coupled BVP with Danckwerts Backmixing: Outlet Conversion = {full.X_out if full else 0:.1f}% (Paper: 66%)</text>')
        svg.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#ccc"/>')
        svg.append(f'<text x="{x0 + pw/2}" y="{y0 + ph + 35}" text-anchor="middle" font-size="11">Reactor Height, h (m)</text>')
        svg.append(f'<text x="{x0 - 38}" y="{y0 + ph/2}" text-anchor="middle" font-size="11" transform="rotate(-90 {x0 - 38} {y0 + ph/2})">CH4 Conversion (%)</text>')

        # Targets
        y80 = y0 + ph - (80.0 / 100.0) * ph
        y66 = y0 + ph - (66.0 / 100.0) * ph
        svg.append(f'<line x1="{x0}" y1="{y80:.1f}" x2="{x0+pw}" y2="{y80:.1f}" stroke="#888" stroke-dasharray="4,4"/>')
        svg.append(f'<line x1="{x0}" y1="{y66:.1f}" x2="{x0+pw}" y2="{y66:.1f}" stroke="#b71c1c" stroke-dasharray="3,3"/>')

        # Reaction only
        pts_rxn = [f"{x0 + (hi/2.11)*pw:.1f},{y0 + ph - (Xi/100.0)*ph:.1f}" for hi, Xi in zip(reaction.h, reaction.X_ch4)]
        svg.append(f'<polyline points="{" ".join(pts_rxn)}" fill="none" stroke="#1976d2" stroke-width="2"/>')

        # Heat transfer
        pts_heat = [f"{x0 + (hi/2.11)*pw:.1f},{y0 + ph - (Xi/100.0)*ph:.1f}" for hi, Xi in zip(heat.h, heat.X_ch4)]
        svg.append(f'<polyline points="{" ".join(pts_heat)}" fill="none" stroke="#388e3c" stroke-width="2" stroke-dasharray="4,3"/>')

        # Full coupled
        if full is not None:
            pts_full = [f"{x0 + (hi/2.11)*pw:.1f},{y0 + ph - (Xi/100.0)*ph:.1f}" for hi, Xi in zip(full.h, full.X_ch4)]
            svg.append(f'<polyline points="{" ".join(pts_full)}" fill="none" stroke="#d32f2f" stroke-width="2.5"/>')

        # Legend
        svg.append(f'<rect x="{x0 + pw - 270}" y="{y0 + ph - 110}" width="260" height="100" fill="#fff" opacity="0.92" stroke="#ddd"/>')
        svg.append(f'<line x1="{x0+pw-260}" y1="{y0+ph-95}" x2="{x0+pw-235}" y2="{y0+ph-95}" stroke="#1976d2" stroke-width="2"/>')
        svg.append(f'<text x="{x0+pw-225}" y="{y0+ph-91}" font-size="10">Reaction only ({reaction.X_out:.1f}%)</text>')
        svg.append(f'<line x1="{x0+pw-260}" y1="{y0+ph-78}" x2="{x0+pw-235}" y2="{y0+ph-78}" stroke="#388e3c" stroke-width="2" stroke-dasharray="4,3"/>')
        svg.append(f'<text x="{x0+pw-225}" y="{y0+ph-74}" font-size="10">Heat transfer ({heat.X_out:.1f}%)</text>')
        if full is not None:
            svg.append(f'<line x1="{x0+pw-260}" y1="{y0+ph-61}" x2="{x0+pw-235}" y2="{y0+ph-61}" stroke="#d32f2f" stroke-width="2.5"/>')
            svg.append(f'<text x="{x0+pw-225}" y="{y0+ph-57}" font-size="10">Coupled BVP w/ Dax ({full.X_out:.1f}%)</text>')
        svg.append(f'<line x1="{x0+pw-260}" y1="{y0+ph-44}" x2="{x0+pw-235}" y2="{y0+ph-44}" stroke="#888" stroke-dasharray="4,4"/>')
        svg.append(f'<text x="{x0+pw-225}" y="{y0+ph-40}" font-size="10">Paper target no-Dax (~80%)</text>')
        svg.append(f'<line x1="{x0+pw-260}" y1="{y0+ph-27}" x2="{x0+pw-235}" y2="{y0+ph-27}" stroke="#b71c1c" stroke-dasharray="3,3"/>')
        svg.append(f'<text x="{x0+pw-225}" y="{y0+ph-23}" font-size="10">Paper target with Dax (~66%)</text>')

        svg.append("</svg>")
        svg_ind = fig_out / "industrial_reaction_heat_transfer_conversion.svg"
        with open(svg_ind, "w", encoding="utf-8") as f:
            f.write("\n".join(svg))
        print(f"Vector SVG saved: {svg_ind}")
    except Exception as e:
        print(f"Warning: could not save SVG figure: {e}")

    # Generate plots if matplotlib is available
    if plt is not None:
        # Plot 1: Industrial conversion profiles
        plt.figure(figsize=(8.0, 5.5))
        plt.plot(reaction.h, reaction.X_ch4, "b-", linewidth=2.0, label="Our model: reaction only")
        plt.plot(heat.h, heat.X_ch4, "g--", linewidth=2.0, label="Our model: heat transfer (rA + T)")
        if full is not None:
            plt.plot(full.h, full.X_ch4, "r-", linewidth=2.5, label=f"Our model: full coupled (rA + T + Dax) [{full.X_out:.1f}%]")
        plt.axhline(80.0, color="gray", linestyle="--", alpha=0.7, label="Paper target without Dax (~80%)")
        plt.axhline(66.0, color="darkred", linestyle=":", alpha=0.8, label="Paper target with Dax (~66%)")
        plt.xlabel("Reactor height, h (m)", fontsize=11)
        plt.ylabel("CH4 conversion (%)", fontsize=11)
        plt.title("Industrial-Scale MMBCR: Conversion vs. Height (D=0.1 m, H=2.11 m)", fontsize=12)
        plt.grid(True, alpha=0.25)
        plt.legend(framealpha=0.9, loc="lower right")
        plt.tight_layout()
        plt.savefig(fig_out / "industrial_reaction_heat_transfer_conversion.png", dpi=220)
        plt.close()

        # Plot 2: Hydrodynamics
        fig, ax = plt.subplots(2, 2, figsize=(9.5, 7.5), sharex=True)
        ax[0, 0].plot(reaction.h, reaction.a_g * 100, "b-", label="Reaction only")
        if full is not None:
            ax[0, 0].plot(full.h, full.a_g * 100, "r-", label="Full coupled (Dax)")
        ax[0, 0].grid(True, alpha=0.25)
        ax[0, 0].set_title("Gas holdup (%)")
        ax[0, 0].legend()

        ax[0, 1].plot(reaction.h, reaction.u_g * 1000, "b-", label="Reaction only")
        if full is not None:
            ax[0, 1].plot(full.h, full.u_g * 1000, "r-", label="Full coupled (Dax)")
        ax[0, 1].grid(True, alpha=0.25)
        ax[0, 1].set_title("Superficial gas velocity (mm/s)")
        ax[0, 1].legend()

        ax[1, 0].plot(reaction.h, reaction.d_b * 1000, "b-", label="Reaction only")
        if full is not None:
            ax[1, 0].plot(full.h, full.d_b * 1000, "r-", label="Full coupled (Dax)")
        ax[1, 0].grid(True, alpha=0.25)
        ax[1, 0].set_title("Bubble diameter (mm)")
        ax[1, 0].set_xlabel("Reactor height, h (m)")
        ax[1, 0].legend()

        ax[1, 1].plot(reaction.h, reaction.a_s, "b-", label="Reaction only")
        if full is not None:
            ax[1, 1].plot(full.h, full.a_s, "r-", label="Full coupled (Dax)")
        ax[1, 1].grid(True, alpha=0.25)
        ax[1, 1].set_title("Specific interfacial area (m²/m³)")
        ax[1, 1].set_xlabel("Reactor height, h (m)")
        ax[1, 1].legend()

        fig.suptitle("Industrial Hydrodynamic Profiles (D=0.1 m, H=2.11 m, Pin=17.84 bar)", fontsize=13)
        fig.tight_layout()
        fig.savefig(fig_out / "industrial_reaction_heat_transfer_hydrodynamics.png", dpi=220)
        plt.close(fig)

    status_data = {
        "case": {
            "D_m": params.D,
            "H_m": params.H,
            "P_in_bar": params.P_in / 1e5,
            "T_G_in_C": params.T_G_in_C,
            "T_L_in_C": params.T_L_in_C,
            "j_CH4_in": params.j_CH4_in,
        },
        "reaction_only": {
            "X_out_pct": float(reaction.X_out),
            "P_out_bar": float(reaction.P[-1] / 1e5),
            "uG_out_mm_s": float(reaction.u_g[-1] * 1000),
            "aG_out_pct": float(reaction.a_g[-1] * 100),
            "d_b_out_mm": float(reaction.d_b[-1] * 1000),
            "a_s_out_m2_m3": float(reaction.a_s[-1]),
        },
        "heat_transfer": {"X_out_pct": float(heat.X_out)},
        "full_coupled": full_status,
        "paper_reference": paper,
    }
    (out / "industrial_status.json").write_text(json.dumps(status_data, indent=2))
    print("Industrial benchmark execution completed successfully.")
    print(f"  Reaction-only X_out: {reaction.X_out:.2f}%")
    print(f"  Heat-transfer X_out: {heat.X_out:.2f}%")
    if full is not None:
        print(f"  Coupled (with Dax) X_in:  {full.X_in:.2f}% (Paper: ~10%)")
        print(f"  Coupled (with Dax) X_out: {full.X_out:.2f}% (Paper: ~66%)")


if __name__ == "__main__":
    main()
