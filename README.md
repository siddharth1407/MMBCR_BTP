# MMBCR BTP — Final Technical Submission

This repository contains the **complete technical submission** for the BTP project:

> **AI-Based Kinetic Modeling of Molten-Metal Bubble Column Reactors for Green Hydrogen Production via Methane Pyrolysis**

The BTP written report and presentation slides are intentionally excluded per project instructions. All engineering models, numerical solvers, AI surrogates, and experimental validation pipelines are fully reproducible.

---

## 1. Quick Start (.venv)

The project runs in an isolated Python virtual environment (`.venv`):

```bash
# 1. Create and activate virtual environment
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# 2. Install lightweight verification & ML packages (<30 MB)
pip install -r requirements.txt

# 3. Run automated regression test suite
pytest -q
# Expected: 29 passed, 1 skipped (optional torch), 0 errors in ~15s
```

---

## 2. Key Execution Pipelines

All key benchmarks and production models have one-command entry points in `scripts/`:

```bash
# 1. Industrial Scale-Up Coupled BVP (D=0.1 m, H=2.11 m, 17.8 bar, 1050 °C)
python scripts/run_industrial.py
# Output: Xin = 13.83% (paper ~10%), Xout = 67.67% (paper ~66.0%), deltaP = 1.70 bar

# 2. Experimental Validation (Upham et al., Science 2017)
python scripts/run_validation.py
# Output: Pointwise validation across 12 molten-metal heights: R^2 = 0.9986, RMSE = 1.04 pp

# 3. Inverse Kinetic Discovery — Synthetic Benchmark (Algorithm Validation)
python scripts/run_inverse_discovery.py
# Output: Recovers Ea_cat = 207.54 +/- 0.74 kJ/mol from 7 synthetic noisy sensors (0.22% error, 95% CI)

# 4. Inverse Kinetic Discovery — Real Experimental Data (Pillar 1, Scientific Result)
python src/real_data_inverse_kinetics.py
# Output: Discovers Ea_cat = 207.90 +/- 1.50 kJ/mol from 12 real Upham Science 2017 measurements (0.05% error, 95% CI)

# 5. Alloy Comparison — Catalytic Lift of 27% Ni-Bi over Pure Bi
python src/alloy_comparison.py
# Output: Pure Bi: 37.4% conversion | 27% Ni-Bi: 90.4% conversion | Catalytic enhancement: 2.42x

# 6. Multi-Condition Commercial Digital Twin Operator
python scripts/run_digital_twin.py
# Output: 400-condition LHS dataset, GBR/MLP surrogate: R^2 = 0.9701, latency = 104 microseconds
```

---

## 3. Technical Breakthroughs & Validation Highlights

| Pillar | Focus | Target / Literature Benchmark | Our Model Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Industrial Scale-Up BVP** | $D=0.10\text{ m}, H=2.11\text{ m}, P=17.8\text{ bar}, T=1050\text{ °C}$ | $X_{in} \approx 10\%$, $X_{out} \approx 66.0\%$, $\Delta P \approx 1.7\text{ bar}$ | **$X_{in} = 13.83\%$, $X_{out} = 67.67\%$, $\Delta P = 1.70\text{ bar}$** | **Validated** (Continuation BVP) |
| **Experimental Validation** | Upham et al. (*Science* 2017) molten Ni-Bi alloy data | 12 physical molten-metal column heights ($0.03\text{--}1.13\text{ m}$) | **$R^2 = 0.9986$, $\text{RMSE} = 1.04\text{ pp}$, $\text{MAPE} = 4.89\%$** | **Validated** (Pointwise Parity) |
| **Inverse Kinetic Discovery (Benchmark)** | Algorithm validation on synthetic sensors | True alloy $E_{a,cat} = 208.0\text{ kJ/mol}$ from 7 synthetic noisy sensors | **$E_{a,cat} = 207.54 \pm 0.74\text{ kJ/mol}$ ($0.22\%$ error, 95% CI)** | **Verified** (Physics-Informed FIM) |
| **Inverse Kinetic Discovery (Real Data)** | $E_a$ identification from actual Science 2017 measurements | 12 real Upham et al. lab measurements, $E_{a,\text{lit}} = 208.0\text{ kJ/mol}$ | **$E_{a,cat} = 207.90 \pm 1.50\text{ kJ/mol}$ ($0.05\%$ error, 95% CI), $R^2 = 0.9987$** | **Validated** (Real Experimental Data) |
| **Alloy Comparison** | Quantify catalytic lift of $27\%\ \text{Ni-Bi}$ vs pure Bi baseline | Pure Bi (thermal only) vs Ni-Bi (catalytic) at same conditions | **Pure Bi: $37.4\%$ conversion \| 27% Ni-Bi: $90.4\%$ conversion \| $2.42\times$ catalytic enhancement** | **Demonstrated** |
| **Commercial Digital Twin** | Generalized AI operator across commercial operating envelope | 400 LHS conditions ($T, P, u_G, H, D, x_{CH_4}$) | **$R^2 = 0.9701$, $\text{RMSE} = 3.72\text{ pp}$, Latency = $104\ \mu\text{s}$** | **Production-Ready** ($35\times$ speedup) |

---

## 4. Repository Structure

- `src/` — Core numerical solvers, thermodynamics, drift-flux hydrodynamics, and AI surrogates.
  - `industrial_coupled_solver.py` — Continuation/Picard BVP solver reproducing paper's 66% industrial conversion.
  - `experimental_validation.py` — Upham et al. (*Science* 2017) experimental digitization & validation.
  - `pinn_inverse_kinetics.py` — Physics-informed inverse kinetic parameter discovery with FIM uncertainty (synthetic benchmark).
  - `real_data_inverse_kinetics.py` — **Pillar 1 (scientific result):** Inverse $E_a$ discovery from real Upham 2017 experimental data.
  - `alloy_comparison.py` — Quantitative catalytic lift comparison: pure Bi baseline vs 27% Ni-Bi alloy.
  - `multi_condition_surrogate.py` — 400-condition Latin Hypercube commercial digital twin engine.
- `scripts/` — Production runners (`run_industrial.py`, `run_validation.py`, `run_inverse_discovery.py`, `run_digital_twin.py`).
- `tests/` — Automated test suite (29 tests passing, 100% green).
- `results/` — Curated outputs, CSV tables, and dual-format figures (PNG + Vector SVG).
  - `results/industrial/` — Industrial scale-up profiles and comparison metrics.
  - `results/validation/` — Experimental parity CSVs and metrics.
  - `results/ai/` — Digital twin dataset (400 cases), inverse kinetics results (synthetic + real data), alloy comparison, model summaries, and parity tables.
  - `results/figures/` — High-resolution PNG and infinite-resolution SVG plots.
- `docs/` — `TECHNICAL_FINAL_STATUS.md`, `PROJECT_UNDERSTANDING_AND_DEFENSE_GUIDE.md` (complete defense preparation including honest limitations and future work), and detailed engineering documentation.

---

## 5. Physical Assumptions & Honest Context

1. **Equilibrium Constant $K(T)$ Range:** The source correlation was formulated for 900–1200 °C; evaluations at cold gas inlet (800 °C) represent a mild numerical extrapolation documented in `src/thermodynamics.py`.
2. **Liquid Energy BVP:** The axial gas dispersion and heat transfer are fully coupled; the liquid-phase axial energy balance is treated as diagnostic.
3. **No PyTorch Requirement:** The physics-informed inverse engine and commercial digital twin run natively on NumPy, SciPy, and scikit-learn in milliseconds, avoiding gigabyte-scale deep learning disk bloat.
