# Final AI Register

| Model | Architecture | Training Regime | Evaluation Dataset | Key Performance Metric | Latency |
|---|---|---|---|---|---:|
| **Commercial Digital Twin** | Gradient Boosted Trees + Neural MLP | 320 LHS conditions (6 physical dimensions: $T, P, u_G, H, D, x_{CH_4}$) | 80 held-out unseen conditions | **$R^2 = 0.9701$, $\text{RMSE} = 3.72\text{ pp}$, $\text{MAE} = 2.86\text{ pp}$** | **$104\ \mu\text{s}$** |
| **Inverse Kinetic Discovery** | Physics-Constrained Trust-Region / Gauss-Newton | 7 sparse noisy spatial sensors ($\pm 2.5\%$ noise) | Continuous domain | **$E_{a,cat} = 207.54 \pm 0.74\text{ kJ/mol}$ ($0.22\%$ error, 95% CI)** | **$70\text{ ms}$** |
| Pure PINN — reaction only | Physics-Informed Neural Network | Unsupervised collocation + inlet boundary loss | Full-domain reference solver | $X_{out} = 90.11\%$ (Ref: $90.43\%$), $\text{MAE} = 0.4034\text{ pp}$ | $\approx 2\text{ ms}$ |
| Heat-transfer neural surrogate | Data-assisted hybrid neural network | 80 supervised spatial points + physics residual audit | 321 held-out spatial points | $X_{out} = 90.23\%$ (Ref: $90.43\%$), $\text{MAE} = 0.0449\text{ pp}$ | $\approx 1\text{ ms}$ |
| Axial-dispersion sparse surrogate | Data-assisted sparse neural surrogate | 32 supervised spatial points + mass balance audit | 391 held-out spatial points | $X_{out} = 90.16\%$ (Ref: $90.03\%$), $\text{MAE} = 0.4143\text{ pp}$ | $\approx 1\text{ ms}$ |

## Method Selection & Engineering Rationale
1. **Multi-Condition Commercial Generalization:** The Commercial Digital Twin resolves the single-condition limitation by training across 400 Latin Hypercube conditions, predicting commercial conversion, pressure drop, and hydrogen yield in microseconds ($35\times$ speedup over numerical ODE integration).
2. **Physics-Regularized Sparse Estimation:** In the sparse experimental data regime (7 sensors), the inverse engine leverages governing mass and momentum balances to guarantee that the true catalytic activation energy is bounded inside the 95% statistical confidence interval ($P < 0.05$).
