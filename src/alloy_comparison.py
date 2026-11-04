"""Alloy comparison: Pure Bi (0% Ni) vs 27% Ni-73% Bi catalyst alloy.

This script quantifies the catalytic enhancement provided by the Ni-Bi alloy
relative to a pure Bi baseline where the catalytic reaction rate is zero
(only homogeneous gas-phase thermal cracking occurs).

Usage
-----
    python src/alloy_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow running directly from the src/ directory or the project root.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from parameters import BenchmarkCase, KineticParameters
from properties import Le2023Properties
from thermodynamics import Le2023Thermo
from reaction_only_solver import solve_reaction_only

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
_ROOT = _SRC.parent
_FIG_DIR = _ROOT / "results" / "figures"
_AI_DIR = _ROOT / "results" / "ai"
_FIG_DIR.mkdir(parents=True, exist_ok=True)
_AI_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared model objects
# ---------------------------------------------------------------------------
case = BenchmarkCase()          # T=1040°C, P=2 bar, D=0.03 m, H=1.15 m, 80/20 CH4/Ar
props = Le2023Properties()
thermo = Le2023Thermo()

# ---------------------------------------------------------------------------
# Kinetic parameters for the two alloys
# ---------------------------------------------------------------------------
# Pure Bi: no Ni => k_cat_0 = 0 removes the catalytic term entirely.
# The non-catalytic (homogeneous gas-phase thermal cracking) path remains
# active through k_non_0 and Ea_non.
kin_pure_bi = KineticParameters(k_cat_0=0.0)

# 27 wt% Ni – 73 wt% Bi: default parameters fitted to the Le et al. benchmark.
kin_27ni_bi = KineticParameters()

# ---------------------------------------------------------------------------
# Solve both cases
# ---------------------------------------------------------------------------
print("Solving Pure Bi (0% Ni) case …")
result_pure_bi = solve_reaction_only(case, props, thermo, kinetics=kin_pure_bi)

print("Solving 27% Ni-73% Bi case …")
result_27ni_bi = solve_reaction_only(case, props, thermo, kinetics=kin_27ni_bi)

# ---------------------------------------------------------------------------
# Compute summary metrics
# ---------------------------------------------------------------------------
x_pure_bi = result_pure_bi.X_out           # outlet CH4 conversion (%)
x_27ni_bi = result_27ni_bi.X_out           # outlet CH4 conversion (%)

conversion_lift = x_27ni_bi - x_pure_bi   # absolute lift (pp)

# Catalytic enhancement factor: ratio of outlet conversions.
# Guard against division by zero for the pure-Bi case.
if x_pure_bi > 0.0:
    catalytic_enhancement_factor = x_27ni_bi / x_pure_bi
else:
    catalytic_enhancement_factor = float("inf")

# H2 yield factor = outlet J_H2 / inlet J_CH4  (stoichiometric max = 2)
# Use molar fluxes inferred from mass fluxes: Y_H2 = (J_H2/M_H2) / (J_CH4_in/M_CH4)
from parameters import M_H2, M_CH4

h2_yield_pure_bi = (result_pure_bi.J_h2[-1] / M_H2) / (case.j_CH4_in / M_CH4)
h2_yield_27ni_bi = (result_27ni_bi.J_h2[-1] / M_H2) / (case.j_CH4_in / M_CH4)

# Carbon yield factor = moles of C produced / moles of CH4 fed
# = moles of CH4 converted = X_out / 100
carbon_yield_pure_bi = x_pure_bi / 100.0
carbon_yield_27ni_bi = x_27ni_bi / 100.0

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
header = f"{'Metric':<35} {'Pure Bi (0% Ni)':>18} {'27% Ni-73% Bi':>18}"
sep = "-" * len(header)
print()
print(sep)
print("  ALLOY COMPARISON: Pure Bi vs 27% Ni-73% Bi")
print(sep)
print(header)
print(sep)
print(f"  {'Outlet CH4 conversion (%)':<33} {x_pure_bi:>18.4f} {x_27ni_bi:>18.4f}")
print(f"  {'H2 yield factor':<33} {h2_yield_pure_bi:>18.4f} {h2_yield_27ni_bi:>18.4f}")
print(f"  {'Carbon yield factor':<33} {carbon_yield_pure_bi:>18.4f} {carbon_yield_27ni_bi:>18.4f}")
print(sep)
print(f"  {'Conversion lift (pp)':<33} {conversion_lift:>36.4f}")
print(f"  {'Catalytic enhancement factor':<33} {catalytic_enhancement_factor:>36.4f}")
print(sep)
print()

# ---------------------------------------------------------------------------
# Plot: conversion profiles on one figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(result_pure_bi.h, result_pure_bi.X_ch4,
        color="#E07B39", lw=2.0, linestyle="--",
        label=f"Pure Bi (0% Ni) – thermal cracking only  [X_out = {x_pure_bi:.2f}%]")
ax.plot(result_27ni_bi.h, result_27ni_bi.X_ch4,
        color="#2C7BB6", lw=2.0,
        label=f"27% Ni-73% Bi – catalytic alloy  [X_out = {x_27ni_bi:.2f}%]")

ax.set_xlabel("Reactor height, h (m)", fontsize=12)
ax.set_ylabel("CH₄ conversion (%)", fontsize=12)
ax.set_title(
    "Alloy Comparison: Catalytic vs Thermal-Cracking-Only Conversion Profiles\n"
    f"T = 1040 °C, P = 2 bar, D = 0.03 m, H = 1.15 m, 80/20 CH₄/Ar",
    fontsize=11,
)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.25)
ax.set_xlim(0, case.H)
ax.set_ylim(bottom=0)

# Annotate lift arrow at mid-column
h_mid_idx = len(result_pure_bi.h) // 2
h_mid = result_pure_bi.h[h_mid_idx]
x_bi_mid = result_pure_bi.X_ch4[h_mid_idx]
x_ni_mid = result_27ni_bi.X_ch4[h_mid_idx]
ax.annotate(
    "",
    xy=(h_mid, x_ni_mid),
    xytext=(h_mid, x_bi_mid),
    arrowprops=dict(arrowstyle="<->", color="grey", lw=1.5),
)
ax.text(
    h_mid + 0.03, (x_bi_mid + x_ni_mid) / 2,
    f"Δ = {x_ni_mid - x_bi_mid:.1f} pp",
    va="center", fontsize=9, color="grey",
)

fig.tight_layout()
_out_png = _FIG_DIR / "alloy_comparison.png"
fig.savefig(_out_png, dpi=220)
plt.close(fig)
print(f"Plot saved → {_out_png}")

# ---------------------------------------------------------------------------
# Save JSON summary
# ---------------------------------------------------------------------------
summary = {
    "outlet_conversion_pure_bi_pct": round(x_pure_bi, 6),
    "outlet_conversion_27ni_bi_pct": round(x_27ni_bi, 6),
    "conversion_lift_pct": round(conversion_lift, 6),
    "catalytic_enhancement_factor": (
        round(catalytic_enhancement_factor, 6)
        if catalytic_enhancement_factor != float("inf")
        else None
    ),
}
_out_json = _AI_DIR / "alloy_comparison_results.json"
with _out_json.open("w") as fh:
    json.dump(summary, fh, indent=2)
print(f"JSON summary saved → {_out_json}")
print()
print("Done.")
