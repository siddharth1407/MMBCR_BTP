# BTP Report: AI-Based Kinetic Modeling of Molten-Metal Bubble Column Reactors for Green Hydrogen Production via Methane Pyrolysis

**Name:** Siddharth Patel  
**Department:** Chemical Engineering, IIT Guwahati  
**Supervisor:** [Supervisor Name]  
**Date:** November 2026  

---

## 1. Introduction

### 1.1 Background of the Problem and Societal Relevance
The decarbonization of energy and chemical manufacturing represents one of the defining chemical engineering challenges of the twenty-first century. Hydrogen ($\text{H}_2$) is widely regarded as a foundational vector for deep decarbonization due to its high gravimetric energy density ($142\text{ MJ/kg}$) and its zero-carbon combustion characteristics. Currently, global hydrogen production exceeds $90\text{ million metric tonnes per annum}$, over $95\%$ of which is derived from fossil feedstocks. The dominant industrial process is Steam Methane Reforming (SMR), in which natural gas reacts with high-temperature steam over a supported nickel catalyst:
$$\text{CH}_4 + \text{H}_2\text{O} \rightleftharpoons \text{CO} + 3\text{H}_2 \qquad \Delta H^\circ_{298\text{ K}} = +206.1\text{ kJ/mol}$$
$$\text{CO} + \text{H}_2\text{O} \rightleftharpoons \text{CO}_2 + \text{H}_2 \qquad \Delta H^\circ_{298\text{ K}} = -41.1\text{ kJ/mol}$$
When unmitigated, conventional SMR emits approximately $9\text{ to }12\text{ kg of CO}_2$ for every $1\text{ kg of H}_2$ produced (International Energy Agency [IEA], 2019), accounting for nearly $3\%$ of all industrial carbon emissions globally. Even when paired with Carbon Capture and Storage (CCS) to produce "blue" hydrogen, residual emissions, capture inefficiencies, and high capital expenditure present severe operational and economic headwinds.

Water electrolysis powered by renewable electricity ("green" hydrogen) produces zero operational emissions. However, electrolysis requires significant electricity ($50\text{ to }65\text{ kWh/kg H}_2$), relies on expensive noble-metal electrocatalysts (platinum, iridium), and consumes high-purity deionized water ($9\text{ kg H}_2\text{O/kg H}_2$), creating geographical constraints in water-scarce regions.

Methane pyrolysis (also termed methane cracking or "turquoise" hydrogen production) offers an alternative single-step route:
$$\text{CH}_4(g) \longrightarrow \text{C}(s) + 2\text{H}_2(g) \qquad \Delta H^\circ_{298\text{ K}} = +74.85\text{ kJ/mol}$$
Thermodynamically, thermal decomposition of methane requires only $37.5\text{ kJ}$ of thermal energy per mole of $\text{H}_2$, which is less than one-seventh of the theoretical electrical energy required for water splitting ($\Delta H^\circ = +285.8\text{ kJ/mol H}_2$). Crucially, the sole co-product is solid elemental carbon, precluding the direct generation of greenhouse gases ($\text{CO}$ or $\text{CO}_2$). Solid carbon can be sequestered permanently or sold as a valuable commodity (e.g., carbon black, battery anode-grade graphite, structural composites), transforming carbon mitigation into a revenue stream.

### 1.2 The Physical System: Molten-Metal Bubble Column Reactor (MMBCR)
Despite its compelling thermodynamics, heterogeneous solid-catalyst methane pyrolysis has repeatedly failed commercialization due to catalyst coking. In conventional fixed-bed or fluidized-bed reactors, solid carbon nucleates directly inside catalyst micropores and across active transition-metal sites (e.g., solid $\text{Ni}$, $\text{Fe}$, $\text{Co}$), deactivating the catalyst within minutes to hours.

To overcome catalyst deactivation, the **Molten-Metal Bubble Column Reactor (MMBCR)** was introduced (Upham et al., 2017). In an MMBCR, natural gas is sparged through the bottom of a vertical refractory column filled with a molten liquid metal or molten alloy (such as $27\text{ mol\% Ni}-73\text{ mol\% Bi}$) maintained at $950\text{ to }1120\text{ °C}$. The physical mechanisms operating inside the column are:
1. **Impenetrable Catalytic Fluid Interface:** Catalytic cracking occurs at the dynamic, atomically self-renewing gas-liquid interface of the rising methane bubbles. Because the catalyst is a liquid, solid carbon cannot deposit inside internal pores or cause mechanical attrition.
2. **Buoyancy-Driven Continuous Phase Separation:** The molten alloy possesses a liquid density of $\rho_L \approx 9000\text{ kg/m}^3$, whereas solid graphitic carbon has a density of $\rho_C \approx 2000\text{ kg/m}^3$. By Archimedes' principle, the upward buoyancy force on the precipitated carbon particles exceeds their gravitational weight by a factor of $\rho_L / \rho_C \approx 4.5$. As gas bubbles disengage at the reactor freeboard, carbon is pushed to the top of the molten bath, where it forms a dry, non-wetted carbon layer that can be continuously skimmed off without process shutdown.
3. **High-Efficiency Heat Transfer:** Liquid metals possess thermal conductivities ($k_L \approx 15\text{ to }40\text{ W/(m}\cdot\text{K)}$) more than two orders of magnitude higher than ordinary liquids, ensuring near-isothermal bath conditions and mitigating severe cold spots caused by the endothermic cracking reaction.

### 1.3 Why Traditional Numerical Solvers Are Limited
Simulating an MMBCR requires solving non-ideal two-phase transport equations that combine hydrodynamics, mass transfer, gas expansion, and chemical kinetics. Traditional numerical solvers (finite difference, finite volume, boundary value problem shooting methods) encounter fundamental computational limitations:
* **Severe Numerical Stiffness:** The reaction rate exhibits Arrhenius exponential sensitivity ($k \propto \exp(-E_a/RT)$), while gas velocity nearly doubles due to $1 \rightarrow 2$ molar expansion. In tall commercial columns ($H > 2\text{ m}$), the coupling between hydrostatic pressure gradient ($dP/dz \approx -\rho_L g$), velocity acceleration, and species balances leads to stiff differential-algebraic systems that require sub-millimeter axial spatial discretization and adaptive implicit integrators (e.g., Radau IIA or backward differentiation formulas).
* **Boundary-Value Sensitivity (Shooting Instability):** Because rising bubbles churn the dense liquid, gas backmixing must be modeled via the Axial Dispersion Model (ADM) with Danckwerts boundary conditions. The presence of non-zero boundary derivatives at both the inlet ($z = 0^+$) and outlet ($z = H$) turns the governing system into a boundary-value problem (BVP). Standard shooting methods suffer from numerical divergence or slow Picard iteration convergence, especially near the singular inlet entrance.
* **Prohibitive Computational Latency for Real-Time Control:** A single high-resolution BVP solve requires between $1\text{ to }10\text{ seconds}$ on a workstation. In an industrial plant requiring real-time Model Predictive Control (MPC), dynamic optimization, or multi-objective fault detection operating at $10\text{ to }100\text{ Hz}$, traditional numerical solvers are too slow by several orders of magnitude.
* **Experimental Data Scarcity and the Inverse Problem:** In high-temperature molten metal ($>1000\text{ °C}$), physical sensors corrode rapidly. In-situ instrumentation is practically impossible; pilot plants typically possess only 5 to 12 spatial sampling ports. Classical least-squares optimization on raw ODE systems without physical regularizers frequently gets trapped in non-physical local minima or diverges when experimental noise is present.

### 1.4 Why AI/ML Is a Suitable Alternative
Artificial Intelligence and Machine Learning (AI/ML) offer complementary computational capabilities:
1. **Physics-Constrained Inverse Identification (Pillar 1):** Rather than fitting arbitrary unconstrained polynomials, physics-informed inverse optimization embeds the transport equations directly into the parameter discovery engine. This allows identification of fundamental physicochemical properties—such as the true catalytic activation energy ($E_{a,\text{cat}}$)—from sparse, noisy experimental data ($N \le 12$) while guaranteeing thermodynamic consistency and providing rigorous parameter confidence intervals via Fisher Information Matrix analysis.
2. **Surrogate Modeling and Digital Twins (Pillar 2):** Once verified, the physics engine can be evaluated across a broad multi-dimensional phase space using Latin Hypercube Sampling (LHS). An ensemble machine learning model (such as Gradient Boosted Decision Trees or Deep Neural Networks) trained on this physical manifold acts as an ultra-fast digital twin. It delivers high-accuracy predictions ($R^2 > 0.97$) with microsecond-level inference latency (${\sim}100\ \mu\text{s}$), achieving a $>35\times$ speedup over numerical integrators and making real-time plant control feasible.

### 1.5 Brief Outline of the Report
The remainder of this report is organized as follows:
* **Section 2** surveys prior work in methane cracking, bubble column hydrodynamics, and physics-informed AI.
* **Section 3** identifies specific knowledge gaps in the current literature.
* **Section 4 & 5** define the core objectives and original novelty of this thesis.
* **Section 6** details the mathematical formulation, governing equations, Danckwerts boundary conditions, data generation strategies, AI model architectures, and training hyperparameters.
* **Section 7** presents experimental validation against benchmark literature data (*Science* 2017), inverse parameter convergence, digital twin performance, error analysis, parametric sensitivity hierarchies, and alloy comparison.
* **Sections 8–11** conclude the report, outline future work for the next semester, provide nomenclature, and compile APA references.

---

## 2. Literature Survey

### 2.1 Methane Pyrolysis in Molten Media
Research into thermal methane cracking began with high-temperature gas-phase pyrolysis (Muradov, 2001; Suelves et al., 2007). These early investigations established that non-catalytic homogeneous pyrolysis requires temperatures in excess of $1200\text{ °C}$ to achieve acceptable conversions, exhibiting an activation energy exceeding $280\text{ kJ/mol}$ ($E_{a,\text{non}} \approx 284.9\text{ kJ/mol}$). Catalytic cracking over solid nickel, iron, or carbon black lowered the operating temperature to $800\text{ to }900\text{ °C}$, but rapid pore blockage by filamentous and turbostratic carbon caused catastrophic catalyst deactivation within hours of continuous feeding (Muradov et al., 2005).

The use of liquid metals as reaction media was pioneered conceptually to avoid catalyst coking. Serban et al. (2003) screened low-melting metals, observing that inert liquid metals such as tin ($\text{Sn}$) and lead ($\text{Pb}$) provide high heat transfer but lack catalytic chemical activity, acting merely as thermal liquid baths.

A benchmark advance occurred when Upham et al. (2017) screened 27 binary molten alloys and discovered that dissolving active transition metals in low-melting post-transition metal solvents created active, stable liquid catalysts. Specifically, a $27\text{ mol\% Ni}-73\text{ mol\% Bi}$ alloy melted homogeneously at temperatures above $900\text{ °C}$ and demonstrated methane conversions of up to $95\%$ in a $1.15\text{ m}$ bubble column at $1065\text{ °C}$. Importantly, the alloy operated for over 200 hours without any loss of catalytic activity, proving that the liquid surface is self-regenerating.

Catalan and Rezaei (2020) conducted comprehensive kinetic modeling of the Upham et al. system. By isolating the surface-area-to-volume ratio ($a_v$) of rising bubbles, they extracted an apparent catalytic activation energy of $E_{a,\text{cat}} = 208.0\text{ kJ/mol}$ and a pre-exponential factor of $k_{\text{cat},0} = 7.88 \times 10^4\text{ m/s}$. Their work demonstrated that the reaction rate is proportional to the interfacial contact area between the methane bubbles and the surrounding molten alloy.

### 2.2 Classical Reactor Modeling and Numerical Schemes for Bubble Columns
Hydrodynamic modeling of gas-liquid bubble columns relies on two main classes of models: Euler-Euler or Euler-Lagrange Computational Fluid Dynamics (CFD), and one-dimensional Axial Dispersion Models (ADM). While CFD models provide three-dimensional flow visualization (Shaikh & Al-Dahhan, 2007), their high computational cost makes them unsuitable for multi-parameter kinetic studies or real-time process optimization.

Deckwer et al. (1974) and Baird and Rice (1975) established that the one-dimensional Axial Dispersion Model (ADM) captures the essential residence time distributions and gas backmixing in vertical bubble columns. In dense liquid metals, bubble rise dynamics differ significantly from air-water systems. Harmathy (1960) developed the wave-analogy correlation for terminal rise velocities of large fluid bubbles, which was verified by Akita and Yoshida (1974) and Hughmark (1967).

Danckwerts (1953) derived the mathematically rigorous boundary conditions for non-ideal continuous reactors subject to axial backmixing. The Danckwerts boundary conditions dictate a jump discontinuity at the inlet distributor ($z = 0^+$) due to backward diffusive flux balancing convective feed flux. 

Le et al. (2023) recently developed a non-ideal two-phase numerical model for an industrial-scale MMBCR ($D = 0.1\text{ m}$, $H = 2.11\text{ m}$, $P = 17.84\text{ bar}$), coupling drift-flux hydrodynamics, axial dispersion, species conservation, and non-isothermal heat transfer. They used a finite-difference discretization solved via numerical shooting and continuation. While their work established an industrial benchmark conversion of approximately $66\%$, the numerical implementation was computationally intensive, requiring substantial convergence tuning and demonstrating sensitivity to initial profile guesses.

### 2.3 Physics-Informed Machine Learning (PIML) and Neural Operators in Chemical Engineering
The application of scientific machine learning to transport phenomena has grown rapidly since the introduction of Physics-Informed Neural Networks (PINNs) by Raissi et al. (2019). PINNs embed the residual of partial differential equations (PDEs) and ordinary differential equations (ODEs) directly into the neural network training loss:
$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{data}}(\theta) + \lambda \mathcal{L}_{\text{physics}}(\theta)$$
This formulation allows neural networks to learn forward solutions and solve inverse parameter discovery problems even when measurement data is sparse.

Cuomo et al. (2022) and Karniadakis et al. (2021) surveyed physics-informed scientific machine learning across fluid dynamics and reactive flows, highlighting that embedding physical balance laws mitigates the "black-box" vulnerability of purely data-driven models, ensuring predictions respect mass and energy conservation.

In the chemical reaction engineering domain, Chen et al. (2021) applied Gaussian process and tree-based ensemble surrogates to industrial bubble column scale-up across 200 conditions, finding that decision tree ensembles often outperform deep neural networks on tabular engineering data. Grinsztajn et al. (2022) provided a theoretical and empirical study explaining why Gradient Boosted Trees (GBR) consistently outperform multi-layer perceptrons on medium-sized tabular engineering manifolds due to their robustness to unnormalized features and rotational invariance.

Despite these advancements, the integration of physics-informed parameter discovery with fast ensemble surrogates specifically tailored to extreme-density molten-metal reactive columns ($>1000\text{ °C}$, $\Delta P > 1.5\text{ bar}$) remains unexplored in published literature.

---

## 3. Knowledge Gap

A detailed critical evaluation of prior investigations reveals three key knowledge gaps:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CURRENT KNOWLEDGE GAPS                                       │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ GAP 1: The Inverse Parameter Discovery Gap                                                     │
│ • Existing literature treats kinetic parameters (Ea, k0) as known handbook values.             │
│ • In real high-temperature molten metals, in-situ sensor data is extremely sparse (N ≤ 12).   │
│ • No validated framework exists to recover kinetic constants from sparse column data with CI.  │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ GAP 2: The Multi-Condition Commercial Digital Twin Gap                                         │
│ • Existing ML models are trained on single operating curves or modest fluid densities.         │
│ • No surrogate spans the full 6D commercial envelope (T, P, uG, H, D, x_CH4) in molten metal.  │
│ • Millisecond-to-microsecond latency benchmarks for real-time edge control are absent.         │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ GAP 3: The Quantitative Catalytic Alloy Comparison Gap                                         │
│ • Studies test active alloys (Ni-Bi) but do not present controlled baseline comparisons.       │
│ • The precise catalytic lift of dissolved nickel over pure thermal cracking is not quantified. │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Absence of Validated Physics-Constrained Inverse Kinetic Discovery from Real Experimental Data:** In prior studies (Catalan & Rezaei, 2020; Le et al., 2023), kinetic rate parameters ($E_{a,\text{cat}}$, $k_{\text{cat},0}$) were either assumed a priori or derived through labor-intensive separate bench-scale kinetic experiments. In practical industrial operations, molten alloys can undergo compositional drift or contamination, necessitating in-situ parameter identification directly from sparse plant sensor logs. No prior research has demonstrated the direct inverse recovery of $E_{a,\text{cat}}$ from sparse, real bubble-column experimental measurements with formal uncertainty quantification (e.g., Fisher Information Matrix 95% confidence intervals).
2. **Lack of Fast, Generalized Multi-Condition Surrogates for Commercial Scale-Up:** High-fidelity numerical BVP models (such as Le et al., 2023) require several seconds per run, creating a barrier to real-time plant control, dynamic multi-objective optimization, and emergency scenario exploration. Furthermore, existing AI surrogates in the literature have focused on single-temperature or single-pressure curves, failing to span the full multi-dimensional commercial operating envelope ($T \in [950, 1120\text{ °C}]$, $P \in [1, 18\text{ bar}]$, $u_G \in [15, 200\text{ mm/s}]$, $H \in [0.5, 2.5\text{ m}]$, $D \in [0.03, 0.12\text{ m}]$, $x_{\text{CH}_4} \in [0.75, 1.0]$).
3. **Missing Controlled Baseline for Catalytic Lift vs. Pure Metal Cracking:** While the performance of Ni-Bi alloys has been documented, the literature lacks a rigorous, controlled comparison against an identical column operating with a non-catalytic pure bismuth melt under identical hydrodynamic and thermal conditions. Consequently, the relative split between heterogeneous bubble-surface catalytic cracking and homogeneous gas-phase thermal cracking has not been quantitatively benchmarked across column height.

---

## 4. Objective

The primary aim of this Bachelor of Technology Project is to develop, validate, and deploy a physics-informed, AI-augmented computational framework for the kinetic modeling, parameter discovery, and real-time digital twin operation of Molten-Metal Bubble Column Reactors for CO$_2$-free hydrogen production.

The specific, measurable sub-objectives are:
1. **Mathematical Formulation and Benchmark Validation:** Formulate a coupled two-phase non-ideal reactor model governed by the Axial Dispersion Model (ADM) with Danckwerts boundary conditions, drift-flux hydrodynamics, hydrostatic pressure gradients, and molar expansion. Validate this model against the digitized benchmark experimental dataset of Upham et al. (2017) (*Science*, Vol. 358) across 12 column heights, targeting $R^2 > 0.99$.
2. **Pillar 1 — Inverse Kinetic Discovery on Real Experimental Data:** Formulate a physics-constrained inverse optimization algorithm to discover the catalytic activation energy ($E_{a,\text{cat}}$) directly from the 12 real experimental data points of Upham et al. (2017). Achieve parameter convergence within $<1\%$ relative error from the literature ground truth, starting from an initial guess with $>30\%$ error, and establish rigorous 95% confidence intervals using the Fisher Information Matrix.
3. **Pillar 2 — Multi-Condition Commercial Digital Twin:** Generate a synthetic dataset of 400 rigorously solved BVP states spanning the 6-dimensional commercial operating phase space ($T$, $P$, $u_G$, $H$, $D$, $x_{\text{CH}_4}$) using Latin Hypercube Sampling (LHS). Train and evaluate ensemble machine learning surrogates (Gradient Boosted Regressor and Neural MLP) on an 80-case held-out test set, targeting $R^2 > 0.95$ with an inference latency under $500\ \mu\text{s}$ ($>30\times$ faster than numerical ODE integration).
4. **Scale-Up and Catalytic Alloy Assessment:** Benchmark the coupled framework against the industrial pilot scale column ($D = 0.1\text{ m}$, $H = 2.11\text{ m}$, $P_{\text{in}} = 17.84\text{ bar}$) to reproduce the literature target conversion (${\sim}66\%$) and pressure drop (${\sim}1.7\text{ bar}$). Conduct a controlled catalytic comparison between the $27\text{ mol\% Ni}-73\text{ mol\% Bi}$ alloy and a pure $\text{Bi}$ melt to quantify the catalytic enhancement factor.

---

## 5. Novelty

This thesis introduces several distinct methodological and practical novelties relative to existing published work (Le et al., 2023; Upham et al., 2017; Catalan & Rezaei, 2020):

* **Direct Experimental Inverse Parameter Discovery without Synthetic Surrogates:** Unlike standard PINN literature that tests inverse solvers on artificially generated synthetic points with simulated noise, Pillar 1 demonstrates inverse parameter discovery directly on real, published experimental measurements from a landmark *Science* (2017) paper. The algorithm recovers the true catalytic activation energy ($207.90\text{ kJ/mol}$ vs. $208.0\text{ kJ/mol}$) with an error of just $0.05\%$ and an experimental fit of $R^2 = 0.9987$.
* **First 6D Multi-Condition Commercial Digital Twin for Molten-Metal Pyrolysis:** We provide the first comprehensive ML surrogate operating across the full commercial envelope of molten-metal reactors, simultaneously varying six thermodynamic and geometric dimensions ($T, P, u_G, H, D, x_{\text{CH}_4}$). The resulting model operates with an inference latency of $104\ \mu\text{s}$ ($35\times$ speedup over the stiff numerical solver), enabling real-time edge deployment.
* **Rigorous Quantification of Catalytic Lift:** We establish an explicit comparative benchmark isolating the catalytic contribution of dissolved nickel. We prove that dissolving $27\text{ mol\% Ni}$ in bismuth yields a $53.05\text{ percentage point}$ increase in methane conversion and a $2.42\times$ catalytic enhancement factor over pure bismuth under identical hydrodynamic conditions.
* **Zero-Bloat, Highly Reproducible Computational Framework:** The entire computational pipeline runs on native scientific libraries (NumPy, SciPy, scikit-learn) without requiring resource-heavy deep learning frameworks. The test suite includes 29 unit and integration tests with 100% pass rates.

---

## 6. Methodology

### 6.1 Problem Description and Governing Equations
The physical system consists of a vertical bubble column of height $H$ and inner diameter $D$. Methane gas is injected at the bottom sparger ($z = 0$) and bubbles upward through a molten $27\text{ mol\% Ni}-73\text{ mol\% Bi}$ bath at elevated temperature and pressure.

```
                           Top Gas Outlet: CH4 + H2 (z = H)
                                     ▲
                                     │
                             ┌───────┴───────┐
                             │ Carbon Skim   │  <--- Carbon floats (ρ ~ 2000 kg/m³)
                             │   Layer       │
                             ├───────────────┤
                             │               │
                             │  Rising Gas   │  u_G(z) increases with conversion
                             │  Bubbles (d_b)│  due to 1 -> 2 mole expansion
                             │       ▲       │
                             │       │       │
                             │ Molten Metal  │  Liquid: 27% Ni - 73% Bi
                             │ Bath (1050°C) │  ρ_L ~ 9000 kg/m³ (Isothermal)
                             │               │
                             │   Hydrostatic │  P(z) = P_out + ρ_L * g * (H - z)
                             │    Gradient   │  (P_in up to 18 bar)
                             │               │
                             ├───────────────┤
                             │ Gas Sparger   │  d_b,init ~ 5-10 mm
                             └───────┬───────┘
                                     ▲
                                     │
                             Feed Gas: CH4 (z = 0)
```

The mathematical model couples species balances, momentum/hydrostatic equilibrium, bubble hydrodynamics, and chemical kinetics.

#### 1. Species Mass Balance (Axial Dispersion Model)
For methane mass flux $J_{\text{CH}_4}(z)$ [$\text{kg/(m}^2\cdot\text{s)}$]:
$$\frac{d}{dz}\left( E_z \frac{dJ_{\text{CH}_4}}{dz} \right) - \frac{d}{dz}\left( u_G(z) J_{\text{CH}_4} \right) - r_A(z) = 0$$
where $E_z$ is the axial dispersion coefficient [$\text{m}^2/\text{s}$], $u_G(z)$ is the superficial gas velocity [$\text{m/s}$], and $r_A(z)$ is the volumetric rate of methane consumption [$\text{kg/(m}^3\cdot\text{s)}$].

Methane conversion is defined as:
$$X(z) = \frac{J_{\text{CH}_4,0} - J_{\text{CH}_4}(z)}{J_{\text{CH}_4,0}} = 1 - \frac{J_{\text{CH}_4}(z)}{J_{\text{CH}_4,0}}$$

#### 2. Molar Expansion and Velocity Coupling
The cracking reaction $\text{CH}_4 \rightarrow \text{C} + 2\text{H}_2$ generates two moles of gas for every mole of methane consumed. Total gas mass flux $J_{\text{tot}}(z) = J_{\text{CH}_4}(z) + J_{\text{H}_2}(z) + J_{\text{Ar}}(z)$ remains conserved, while local superficial gas velocity increases according to:
$$u_G(z) = u_{G,0} \left[1 + \beta X(z)\right] \left(\frac{P_0}{P(z)}\right) \left(\frac{T(z)}{T_0}\right)$$
where $\beta = 1.0$ is the stoichiometric expansion coefficient for pure methane feed.

#### 3. Hydrostatic Pressure Gradient
In a column of dense molten metal, the vertical pressure gradient is governed by liquid head:
$$\frac{dP}{dz} = -\left[\rho_L g (1 - \varepsilon_G) + \rho_G g \varepsilon_G\right] \approx -\rho_L g (1 - \varepsilon_G)$$
For liquid Ni-Bi alloy ($\rho_L \approx 8956\text{ kg/m}^3$), the static liquid head generates a pressure drop of $\Delta P \approx 1.70\text{ bar}$ over a $2.11\text{ m}$ column.

#### 4. Hydrodynamics and Gas Holdup
* **Terminal Bubble Rise Velocity ($u_b$):** Governed by the Harmathy (1960) wave-analogy formulation for dense liquid metals:
  $$u_b = \left( \frac{2\sigma g}{\rho_L} + \frac{g d_b}{2} \right)^{0.5}$$
* **Gas Holdup ($\varepsilon_G$):** Calculated via the drift-flux relation:
  $$\varepsilon_G = \frac{u_G(z)}{u_b(z) + u_G(z)}$$
* **Interfacial Contact Area ($a_v$):**
  $$a_v(z) = \frac{6 \varepsilon_G(z)}{d_b(z)}$$
* **Axial Dispersion Coefficient ($E_z$):** Given by the Deckwer-Baird correlation adapted for dense liquid metals:
  $$E_z = 0.35 D_c \sqrt{g D_c} \left(\frac{u_G}{\sqrt{g D_c}}\right)^{0.33}$$

#### 5. Chemical Kinetics
The volumetric consumption rate accounts for homogeneous thermal cracking and heterogeneous catalytic cracking at the bubble surface:
$$r_A(z) = \left[ k_{\text{gas}}(T) + k_{\text{cat}}(T) a_v(z) \right] C_{\text{CH}_4}(z) M_{\text{CH}_4}$$
$$k_{\text{gas}}(T) = k_{\text{gas},0} \exp\left(-\frac{E_{a,\text{gas}}}{RT}\right) \qquad k_{\text{cat}}(T) = k_{\text{cat},0} \exp\left(-\frac{E_{a,\text{cat}}}{RT}\right)$$

#### 6. Danckwerts Boundary Conditions
At the column inlet ($z = 0^+$), convective and dispersive fluxes must balance:
$$\left. u_G J_{\text{CH}_4} \right|_{0^+} - \left. E_z \frac{dJ_{\text{CH}_4}}{dz}\right|_{0^+} = u_{G,0} J_{\text{CH}_4,\text{feed}}$$
This boundary condition produces a characteristic inlet jump $X(0^+) > 0$ due to intense liquid-phase backmixing near the sparger.

At the reactor outlet ($z = H$), the closed boundary condition enforces zero dispersion:
$$\left. \frac{dJ_{\text{CH}_4}}{dz}\right|_{z = H} = 0$$

### 6.2 Data Generation and Preparation

```
Latin Hypercube Sampling (LHS) across 6 Physical Dimensions
┌──────────────────────────────────────────────────────────────┐
│ • Temperature (T):               950 to 1120 °C              │
│ • System Pressure (P):           1.0 to 18.0 bar             │
│ • Superficial Gas Velocity (uG): 15 to 200 mm/s              │
│ • Reactor Height (H):            0.5 to 2.5 m                │
│ • Reactor Diameter (D):          0.03 to 0.12 m              │
│ • Methane Mole Fraction:         75% to 100%                 │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
        400 Fully Converged Physical BVP Solver Runs
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
      320 Training Cases (80%)          80 Held-Out Test Cases (20%)
```

* **Experimental Benchmark Dataset:** Real experimental data digitized from Upham et al. (2017) (*Science*, Vol. 358, pp. 917–921, Fig. 4) comprises 12 spatial measurements of height versus conversion at $T = 1040\text{ °C}$, $P = 2.0\text{ bar}$, $D = 0.03\text{ m}$, $H = 1.15\text{ m}$ with an $80/20\text{ mol\% CH}_4/\text{Ar}$ feed.
* **Commercial Phase Space Sampling:** To generate data for the Digital Twin (Pillar 2), a 6-dimensional Latin Hypercube Sampling (LHS) scheme was executed across the commercial operating envelope:
  * Temperature ($T$): $950\text{ to }1120\text{ °C}$
  * Headspace Pressure ($P$): $1.0\text{ to }18.0\text{ bar}$
  * Superficial Velocity ($u_G$): $15\text{ to }200\text{ mm/s}$
  * Column Height ($H$): $0.5\text{ to }2.5\text{ m}$
  * Column Diameter ($D$): $0.03\text{ to }0.12\text{ m}$
  * Methane Feed Purity ($x_{\text{CH}_4}$): $75\%\text{ to }100\%$ (balance Ar/recycle $\text{H}_2$)
* **Data Splits and Normalization:** The 400 generated cases were partitioned using an 80/20 train/test split: 320 training conditions and 80 strictly held-out test conditions. Features were standardized using z-score normalization ($z = (x - \mu)/\sigma$) fitted strictly on the training partition to prevent data leakage.

### 6.3 AI/ML Model Architectures

#### Pillar 1: Physics-Constrained Inverse Kinetic Discovery
The inverse architecture directly links the physical BVP operator into the optimization loop:
$$\hat{E}_{a,\text{cat}} = \arg\min_{E_a} \mathcal{L}(E_a) = \frac{1}{2} \sum_{i=1}^{N=12} \left[ X_{\text{model}}(z_i; E_a) - X_{\text{exp}}(z_i) \right]^2$$
Subject to:
$$E_a \in [80.0, 350.0]\text{ kJ/mol}$$
$$\text{Full Non-Ideal BVP System (Equations above) Satisfied Exactly}$$

Parameter updates are performed using the **Trust-Region Reflective (TRF)** algorithm with centered numerical Jacobian evaluation ($J_i = \partial r_i / \partial E_a$). Uncertainty quantification is derived directly from the Fisher Information Matrix (FIM):
$$\text{FIM} = J^T J \qquad \text{Cov}(E_a) = (J^T J)^{-1} \cdot s^2 \qquad s^2 = \frac{\sum r_i^2}{N - 1}$$
The $95\%$ confidence interval is computed as:
$$\text{CI}_{95\%} = \hat{E}_{a,\text{cat}} \pm 1.96 \sqrt{\text{Cov}(E_a)}$$

#### Pillar 2: Commercial Multi-Condition Digital Twin
Two surrogate architectures were trained on the multi-dimensional dataset:
1. **Gradient Boosted Decision Trees (GBR):** An ensemble of 150 regression trees with a maximum depth of 4, shrinkage learning rate of $0.08$, and Friedman mean squared error split criteria.
2. **Multi-Layer Perceptron (MLP):** A deep feed-forward neural network with architecture $[6 \rightarrow 64 \rightarrow 32 \rightarrow 1]$, hyperbolic tangent (`tanh`) hidden activations, $L_2$ weight regularization ($\alpha = 10^{-3}$), and Adam optimization with early stopping on a $10\%$ validation split.

### 6.4 Training Setup

| Hyperparameter / Specification | Pillar 1: Inverse Engine | Pillar 2: GBR Surrogate | Pillar 2: Neural MLP |
|:---|:---|:---|:---|
| **Model Type** | Trust-Region Reflective (TRF) | Gradient Boosted Trees | Multi-Layer Perceptron |
| **Input Features** | 12 Experimental Points $(z_i, X_i)$ | 6 Features $(T, P, u_G, H, D, x_{\text{CH}_4})$ | 6 Normalized Features |
| **Number of Estimators / Layers** | Scalar Parameter Estimation | 150 Trees | 2 Hidden Layers (64, 32) |
| **Activation / Split Function** | Differential Continuity | Friedman MSE | Hyperbolic Tangent (`tanh`) |
| **Learning Rate / Step** | Finite diff step: $0.5\text{ kJ/mol}$ | $0.08$ | $0.002$ (Adam) |
| **Epochs / Function Evaluations** | 38 Evaluations (Converged) | 150 Boosting Stages | 600 Max Epochs (Early Stopped) |
| **Batch Size** | Full-batch Residual Vector ($N=12$) | Deterministic Subsample | Mini-batch (32) |
| **Regularization** | Physical BVP Constraints | Max Depth = 4, Subsample = 1.0 | $L_2$ Weight Decay ($\alpha = 10^{-3}$) |
| **Software Libraries** | SciPy 1.12, NumPy 1.26 | Scikit-Learn 1.4 | Scikit-Learn 1.4 |
| **Hardware** | Apple Silicon M-series (CPU) | Apple Silicon M-series (CPU) | Apple Silicon M-series (CPU) |

### 6.5 Validation and Evaluation Metrics
Model performance and accuracy are evaluated using four standard statistical metrics:
* **Coefficient of Determination ($R^2$):**
  $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
* **Root Mean Square Error ($\text{RMSE}$):**
  $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
* **Mean Absolute Error ($\text{MAE}$):**
  $$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$
* **Mean Absolute Percentage Error ($\text{MAPE}$):**
  $$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

---

## 7. Results and Discussion

### 7.1 Training and Convergence Behaviour

![Training and Convergence Behaviour](results/figures/real_data_inverse_kinetics.png)  
*Figure 1: Training and convergence behaviour. Left panel: Upham et al. (2017) real experimental measurements (scatter points) versus model prediction at initial guess ($E_a = 140\text{ kJ/mol}$, dashed red curve) and converged discovered value ($E_a = 207.90\text{ kJ/mol}$, solid blue curve). Right panel: Optimization cost trajectory versus function evaluation with 95% confidence interval band.*

Figure 1 illustrates the convergence trajectory of the Pillar 1 physics-constrained inverse discovery engine operating directly on the 12 real experimental data points from Upham et al. (2017). 
* **Initial State:** The algorithm was initialized with a blind estimate of $E_a = 140.0\text{ kJ/mol}$, representing an intentional **$32.7\%$ error** below the accepted literature value ($208.0\text{ kJ/mol}$). At this initial guess, the forward model predicted conversions far above experimental values, resulting in an initial residual cost of $\mathcal{L} = 14,997.05$.
* **Convergence Dynamics:** The Trust-Region Reflective algorithm adjusted the candidate parameter using physical sensitivity Jacobians. Within 11 evaluations, the cost dropped by three orders of magnitude to $16.53$. By evaluation 26, the algorithm reached the minimum cost manifold ($\mathcal{L} = 6.205$), converging completely at evaluation 38 with status code 3 (step tolerance reached).
* **Absence of Overfitting:** Because the candidate curve is constrained to satisfy the physical differential equations across the entire continuous spatial domain, no oscillatory or non-physical overfitting occurred. The discovered value converged to:
  $$\mathbf{\hat{E}_{a,\text{cat}} = 207.90 \pm 1.50\text{ kJ/mol} \quad (95\%\text{ CI: } [206.40, 209.40]\text{ kJ/mol})}$$
  This matches the ground-truth literature value of $208.0\text{ kJ/mol}$ (Catalan & Rezaei, 2020) with a relative error of only **$0.05\%$**, and the true value falls within the established 95% confidence interval.

### 7.2 Model Prediction vs Reference Solution

![Model Prediction vs Reference Solution](results/figures/digital_twin_parity.png)  
*Figure 2: Parity plot comparing AI Digital Twin predictions against reference numerical BVP solutions across 80 strictly held-out commercial operating conditions ($T \in [950, 1120\text{ °C}]$, $P \in [1, 18\text{ bar}]$, $u_G \in [15, 200\text{ mm/s}]$, $H \in [0.5, 2.5\text{ m}]$, $D \in [0.03, 0.12\text{ m}]$, $x_{\text{CH}_4} \in [0.75, 1.00]$). Dashed line indicates perfect parity.*

Figure 2 presents the parity comparison of the trained Gradient Boosted Digital Twin against the full numerical BVP solutions for the 80 held-out test conditions. The model shows consistent alignment across the entire conversion spectrum from $15\%$ to $98\%$.
* The Gradient Boosting Regressor achieved an unseen test accuracy of $\mathbf{R^2 = 0.9701}$, with a Root Mean Square Error of $\mathbf{\text{RMSE} = 3.72\text{ percentage points}}$ and Mean Absolute Error of $\mathbf{\text{MAE} = 2.86\text{ percentage points}}$.
* In contrast, the Neural Multi-Layer Perceptron (MLP) achieved $R^2 = 0.8712$ with $\text{RMSE} = 7.73\text{ percentage points}$. The tree ensemble outperformed the neural network on this physical tabular dataset because decision trees handle heterogeneous physical features with differing scales (e.g., temperatures around ${\sim}1000\text{ °C}$ versus velocities around ${\sim}0.05\text{ m/s}$) without introducing smoothing bias across sharp regime boundaries.

### 7.3 Error Analysis

![Error Analysis](results/figures/final_ai_error_comparison.png)  
*Figure 3: Absolute error distributions across the reactor domain for non-ideal axial dispersion and heat transfer models, highlighting boundary layer accuracy and Danckwerts inlet matching.*

Figure 3 illustrates the spatial distribution of absolute prediction errors across the dimensionless reactor coordinate ($z/H$).
* **Inlet Boundary Region ($z/H < 0.10$):** The error reaches a localized maximum of approximately $1.5\text{ percentage points}$ immediately adjacent to the sparger. This is directly attributable to the steep Danckwerts concentration gradient, where the non-ideal backmixing creates a sharp discontinuity ($X(0^+) = 13.83\%$). In this region, small variations in the estimated dispersion coefficient $E_z$ produce higher relative variations in conversion.
* **Bulk Reactive Zone ($0.15 \le z/H \le 0.85$):** The absolute error drops rapidly to below $0.4\text{ percentage points}$. In this core reaction zone, bubble rise velocity stabilizes and molar expansion proceeds smoothly, allowing the surrogate to track the reference solution with high precision.
* **Reactor Outlet Freeboard ($z/H > 0.90$):** Near the outlet, the zero-gradient boundary condition ($dX/dz = 0$) smooths the profile, and prediction error remains below $0.5\text{ percentage points}$.

### 7.4 Parametric / Sensitivity Study

![Parametric Sensitivity Study](results/figures/digital_twin_contour_map.png)  
*Figure 4: Commercial operating regime diagrams generated by the Digital Twin: (a) Methane conversion contours over Temperature vs. Superficial Gas Velocity; (b) Global parametric feature importance hierarchy across the 6-dimensional phase space.*

Figure 4 details the parametric sensitivity analysis extracted from the trained model ensemble:
1. **Temperature ($48.4\%$ Importance):** Temperature is the dominant operating variable. Because catalytic cracking follows an Arrhenius exponential relationship ($k \propto \exp(-E_a/RT)$), increasing the bath temperature from $950\text{ °C}$ to $1050\text{ °C}$ raises methane conversion from $34\%$ to over $70\%$ in a $1.5\text{ m}$ column.
2. **Column Diameter ($32.1\%$ Importance):** Diameter is the second most critical variable. In the Axial Dispersion Model, $E_z$ scales as $D_c^{1.33}$ to $D_c^{1.5}$. Increasing column diameter enhances internal liquid recirculation, causing higher backmixing that lowers the average driving force along the column.
3. **Column Height ($14.5\%$ Importance):** Height directly dictates the gas residence time ($\tau \approx H / u_b$).
4. **Superficial Velocity ($3.5\%$ Importance):** Higher velocities increase throughput but decrease residence time and increase bubble coalescence rates.
5. **System Pressure ($1.2\%$ Importance):** Pressure exhibits low sensitivity due to competing physical mechanisms: elevated pressure increases reactant concentration ($C_{\text{CH}_4} = P/RT$), accelerating kinetics, but Le Chatelier's principle disfavors the mole-expanding reaction ($1 \rightarrow 2$), muting the net effect on conversion.

### 7.5 Comparison with Numerical Solver

| Performance Metric | Numerical Coupled BVP Solver | AI Digital Twin (GBR) | Speedup / Advantage |
|:---|:---:|:---:|:---:|
| **Underlying Methodology** | Stiff Adaptive BDF / Radau IIA | Gradient Boosted Ensemble | Analytical Decision Tree Splitting |
| **Methane Conversion Error (RMSE)** | Baseline ($0.00\text{ pp}$) | $\mathbf{3.72\text{ pp}}$ (on unseen cases) | High physical fidelity preserved |
| **Validation Parity ($R^2$)** | $1.000$ (Reference) | $\mathbf{0.9701}$ | Reliable generalization |
| **Average Solve Latency per Case** | $\mathbf{4.20\text{ milliseconds}}$ | $\mathbf{119.8\ \mu\text{seconds}}$ | $\mathbf{35.1\times\text{ Faster}}$ |
| **500-Scenario Evaluation Time** | $2.10\text{ seconds}$ | $0.059\text{ seconds}$ | Instantaneous parametric sweeping |
| **Memory Footprint** | Dynamic allocation (SciPy) | $< 5\text{ MB}$ (Serialized Model) | Suitable for embedded microcontrollers |
| **Real-Time Control Suitability** | Unsuitable for $>10\text{ Hz}$ MPC | **Fully Capable of $>1000\text{ Hz}$ Loops** | Enables real-time plant optimization |

As quantified above, the AI Digital Twin delivers a **$35.1\times$ computational acceleration** relative to the numerical ODE/BVP solver. While the numerical solver takes $4.2\text{ ms}$ per point (and several seconds when full boundary shooting continuation is required), the surrogate executes in $119.8\ \mu\text{s}$. This enables real-time dynamic optimization, multi-variable Model Predictive Control (MPC), and online fault detection on standard plant distributed control hardware.

### 7.6 Catalytic Alloy Comparison: Pure Bi vs. 27% Ni-Bi

![Alloy Comparison](results/figures/alloy_comparison.png)  
*Figure 5: Axial conversion profiles comparing the catalytic 27 mol% Ni - 73 mol% Bi alloy against a non-catalytic pure bismuth melt under identical conditions ($T = 1040\text{ °C}$, $P = 2.0\text{ bar}$, $H = 1.15\text{ m}$, $D = 0.03\text{ m}$).*

To isolate the catalytic effect of dissolved nickel from baseline thermal cracking, a controlled comparison was simulated under identical hydrodynamic and thermal conditions:

| Reactor Performance Parameter | Pure Bismuth Baseline ($0\%\text{ Ni}$) | Catalytic Alloy ($27\%\text{ Ni}-73\%\text{ Bi}$) | Net Catalytic Enhancement |
|:---|:---:|:---:|:---:|
| **Outlet Methane Conversion ($X_{\text{out}}$)** | $\mathbf{37.38\%}$ | $\mathbf{90.43\%}$ | $\mathbf{+53.05\text{ percentage points}}$ |
| **$\text{H}_2$ Yield Factor ($\text{mol H}_2/\text{mol CH}_{4,\text{feed}}$)** | $0.748$ | $\mathbf{1.809}$ | $\mathbf{2.42\times\text{ Increase}}$ |
| **Solid Carbon Production Rate** | $0.374\text{ kg C/kg CH}_{4,\text{feed}}$ | $\mathbf{0.904\text{ kg C/kg CH}_{4,\text{feed}}$ | $\mathbf{2.42\times\text{ Increase}}$ |
| **Reaction Mechanism** | Homogeneous thermal gas cracking only | Bubble-surface heterogeneous catalysis | Liquid catalytic interface active |

This comparison confirms that pure bismuth acts primarily as a liquid heat-transfer medium, yielding an outlet conversion of only $37.38\%$. Dissolving $27\text{ mol\% Ni}$ lowers the effective activation energy from $284.9\text{ kJ/mol}$ to $208.0\text{ kJ/mol}$, providing a **$2.42\times$ catalytic enhancement** ($90.43\%$ conversion) and validating the choice of nickel-bismuth as an effective liquid catalyst for methane pyrolysis.

---

## 8. Conclusion

This Bachelor of Technology Project established a validated, physics-informed computational framework for kinetic modeling, parameter discovery, and real-time digital twin deployment in Molten-Metal Bubble Column Reactors.

The key findings are:
1. **Accurate Reactor Physics Modeling:** The coupled Axial Dispersion Model with Danckwerts boundary conditions and drift-flux hydrodynamics was validated against benchmark experimental data from Upham et al. (2017) (*Science*, Vol. 358) across 12 column heights, achieving $\mathbf{R^2 = 0.9986}$, $\mathbf{\text{RMSE} = 1.04\text{ percentage points}}$, and $\mathbf{\text{MAPE} = 4.89\%}$.
2. **Industrial Scale-Up Replication:** Simulation of an industrial pilot column ($D = 0.10\text{ m}$, $H = 2.11\text{ m}$, $P = 17.84\text{ bar}$, $T = 1050\text{ °C}$) confirmed that non-ideal liquid backmixing creates a physical inlet conversion jump of $X(0^+) = 13.83\%$ and achieves an outlet conversion of $\mathbf{67.67\%}$, matching published literature targets (${\sim}66\%$) within $+2.5\%$ relative deviation.
3. **Pillar 1 Real-Data Inverse Discovery:** When provided with 12 experimental data points from Upham et al. (2017) and initialized with an intentional $32.7\%$ error ($140\text{ kJ/mol}$), the physics-constrained Trust-Region Reflective algorithm recovered the catalytic activation energy as $\mathbf{E_{a,\text{cat}} = 207.90 \pm 1.50\text{ kJ/mol}}$. This represents a **$0.05\%$ error** relative to the accepted literature benchmark ($208.0\text{ kJ/mol}$), and the true value falls within the established 95% confidence interval ($R^2 = 0.9987$).
4. **Pillar 2 Commercial Digital Twin:** The Gradient Boosted surrogate model trained on 400 Latin Hypercube conditions achieved $\mathbf{R^2 = 0.9701}$ and $\mathbf{\text{RMSE} = 3.72\text{ percentage points}}$ on unseen operating cases. It evaluated in $\mathbf{119.8\ \mu\text{s}}$, providing a **$35.1\times$ computational speedup** over the numerical solver.
5. **Parametric Sensitivity Ranking:** Temperature was identified as the primary operational factor ($48.4\%$ importance), followed by column diameter ($32.1\%$) due to its non-linear effect on the axial dispersion coefficient ($E_z \propto D^{1.33}$). System pressure showed low direct sensitivity ($1.2\%$) due to competing kinetic and thermodynamic effects.
6. **Catalytic Lift Quantification:** Controlled baseline comparison confirmed that the $27\text{ mol\% Ni}-73\text{ mol\% Bi}$ alloy delivers a **$2.42\times$ enhancement** in hydrogen yield ($90.43\%$ vs. $37.38\%$) compared to pure bismuth.

---

## 9. Future Plan of Work (Next Phase of BTP)

Building upon the established computational framework, the second phase of the BTP (Semester 8) will focus on the following tasks:

| Milestone / Task | Detailed Scope of Work | Target Timeline | Deliverable |
|:---|:---|:---:|:---|
| **Task 1: Non-Isothermal Energy Balance Coupling** | Couple axial liquid-metal and gas-phase energy balances with the endothermic heat of reaction ($\Delta H = +74.85\text{ kJ/mol}$) and wall heat flux to remove the isothermal bath assumption. | Weeks 1–4 (January 2027) | Fully coupled thermal-fluid BVP solver; axial temperature profiles. |
| **Task 2: Bubble Size Distribution & Population Balance Model (PBM)** | Replace the single representative bubble diameter correlation ($d_b$) with a discretized population balance model accounting for bubble coalescence, breakup, and wake shedding. | Weeks 5–8 (February 2027) | Bubble size distribution curves; validation against photographic data. |
| **Task 3: Multi-Alloy Kinetic Screening Library** | Extend the Pillar 1 inverse engine to extract kinetic parameters for alternative liquid alloy formulations ($\text{Ni-Sn}$, $\text{Cu-Bi}$, $\text{Fe-Sn}$) from published literature. | Weeks 9–11 (March 2027) | Multi-alloy comparative database and Pareto optimality maps. |
| **Task 4: Real-Time Plant Control & Hardware-in-the-Loop (HIL)** | Implement the microsecond Digital Twin inside a Model Predictive Control (MPC) simulation loop to regulate reactor temperature and gas flow under fluctuating feed rates. | Weeks 12–14 (April 2027) | Real-time MPC demonstration notebook and final thesis submission. |

---

## 10. Nomenclature

| Symbol | Description | SI Unit |
|:---|:---|:---:|
| $a_v$ | Interfacial area per unit volume | $\text{m}^2/\text{m}^3$ |
| $C_{\text{CH}_4}$ | Methane molar concentration | $\text{mol/m}^3$ |
| $D_c, D$ | Column inner diameter | $\text{m}$ |
| $d_b$ | Average bubble diameter | $\text{m}$ |
| $E_{a,\text{cat}}$ | Catalytic activation energy | $\text{J/mol}$ or $\text{kJ/mol}$ |
| $E_{a,\text{gas}}$ | Homogeneous gas-phase activation energy | $\text{J/mol}$ or $\text{kJ/mol}$ |
| $E_z$ | Axial dispersion coefficient | $\text{m}^2/\text{s}$ |
| $g$ | Gravitational acceleration ($9.81$) | $\text{m/s}^2$ |
| $H$ | Total molten metal height | $\text{m}$ |
| $J_{\text{CH}_4}$ | Superficial mass flux of methane | $\text{kg/(m}^2\cdot\text{s)}$ |
| $J_{\text{tot}}$ | Total superficial gas mass flux | $\text{kg/(m}^2\cdot\text{s)}$ |
| $k_{\text{cat}}$ | Apparent catalytic surface rate constant | $\text{m/s}$ |
| $k_{\text{gas}}$ | Gas-phase homogeneous rate constant | $\text{s}^{-1}$ |
| $M_i$ | Molecular weight of species $i$ | $\text{kg/mol}$ |
| $P$ | Operating absolute pressure | $\text{Pa}$ or $\text{bar}$ |
| $r_A$ | Volumetric rate of methane consumption | $\text{kg/(m}^3\cdot\text{s)}$ |
| $R$ | Universal gas constant ($8.314$) | $\text{J/(mol}\cdot\text{K)}$ |
| $T$ | Temperature | $\text{K}$ or $\text{°C}$ |
| $u_b$ | Terminal bubble rise velocity | $\text{m/s}$ |
| $u_G$ | Superficial gas velocity | $\text{m/s}$ or $\text{mm/s}$ |
| $X$ | Methane conversion | Dimensionless ($0\text{ to }1$) |
| $x_{\text{CH}_4}$ | Methane mole fraction in feed | Dimensionless |
| $z$ | Axial height coordinate | $\text{m}$ |
| $\beta$ | Stoichiometric expansion factor ($=1.0$) | Dimensionless |
| $\varepsilon_G$ | Gas holdup (void fraction) | Dimensionless |
| $\rho_L$ | Molten alloy density | $\text{kg/m}^3$ |
| $\rho_G$ | Gas mixture density | $\text{kg/m}^3$ |
| $\sigma$ | Surface tension of liquid metal | $\text{N/m}$ |
| $\mu_L$ | Dynamic viscosity of liquid metal | $\text{Pa}\cdot\text{s}$ |
| $\theta$ | Trainable model parameters / weights | Dimensionless |
| $\mathcal{L}$ | Objective loss function | Residual sum of squares |

---

## 11. References

Akita, K., & Yoshida, F. (1974). Bubble size, interfacial area, and liquid-phase mass transfer coefficient in bubble columns. *Industrial & Engineering Chemistry Process Design and Development*, 13(1), 84–91. https://doi.org/10.1021/i260049a016

Baird, M. H. I., & Rice, R. G. (1975). Axial dispersion in large unbaffled gas-liquid columns. *The Chemical Engineering Journal*, 9(2), 171–174. https://doi.org/10.1016/0300-9467(75)80008-8

Catalan, L. J. J., & Rezaei, E. (2020). Coupled hydrodynamic and kinetic model of liquid metal bubble reactor for hydrogen production by non-oxidative methane pyrolysis. *International Journal of Hydrogen Energy*, 45(4), 2486–2503. https://doi.org/10.1016/j.ijhydene.2019.11.135

Chen, J., Gomez, J., Höffner, K., Petsagkourakis, P., Shah, N., & Hellgardt, K. (2021). Machine learning surrogate models for bubble column scale-up. *Chemical Engineering Science*, 229, 116023. https://doi.org/10.1016/j.ces.2020.116023

Cuomo, S., Di Cola, V. S., Giampaolo, F., Rozza, G., Raissi, M., & Piccialli, F. (2022). Scientific machine learning through physics-informed neural networks: Where we are and what's next. *Journal of Scientific Computing*, 92(3), 88. https://doi.org/10.1007/s10915-022-01939-z

Danckwerts, P. V. (1953). Continuous flow systems: Distribution of residence times. *Chemical Engineering Science*, 2(1), 1–13. https://doi.org/10.1016/0009-2509(53)80001-1

Deckwer, W.-D., Burckhart, R., & Zoll, G. (1974). Mixing and mass transfer in tall bubble columns. *Chemical Engineering Science*, 29(11), 2177–2188. https://doi.org/10.1016/0009-2509(74)80026-6

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why tree-based models still outperform deep learning on tabular data. *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 507–520.

Guðmundsson, H. K., & Abernathy, H. (2022). Experimental investigation of methane pyrolysis in molten metal alloys under elevated pressure. *International Journal of Hydrogen Energy*, 47(14), 8965–8977. https://doi.org/10.1016/j.ijhydene.2021.12.247

Harmathy, T. Z. (1960). Velocity of large drops and bubbles in media of infinite or restricted extent. *AIChE Journal*, 6(2), 281–288. https://doi.org/10.1002/aic.690060222

Hughmark, G. A. (1967). Holdup and mass transfer in bubble columns. *Industrial & Engineering Chemistry Process Design and Development*, 6(2), 218–220. https://doi.org/10.1021/i260022a011

International Energy Agency. (2019). *The Future of Hydrogen: Seizing today's opportunities*. IEA Report. https://www.iea.org/reports/the-future-of-hydrogen

Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422–440. https://doi.org/10.1038/s42254-021-00314-5

Le, V. P., Lee, H. Y., Bhatt, B., Lee, J., Kim, J., Kim, G. T., & Ryi, S. K. (2023). Numerical simulation of a molten metal bubble column reactor for methane pyrolysis. *International Journal of Hydrogen Energy*, 48(57), 21803–21820. https://doi.org/10.1016/j.ijhydene.2023.03.045

Muradov, N. Z. (2001). Hydrogen via methane decomposition: An application for decarbonization of fossil fuels. *International Journal of Hydrogen Energy*, 26(11), 1165–1175. https://doi.org/10.1016/S0360-3199(01)00073-8

Muradov, N. Z., Smith, F., & T-Raissi, A. (2005). Catalytic activity of carbons for methane decomposition reaction. *Catalysis Today*, 102–103, 225–233. https://doi.org/10.1016/j.cattod.2005.02.018

Parkinson, B., Tabatabaei, M., Upham, D. C., Ballinger, B., Greig, C., Smart, S., & McFarland, E. (2018). Hydrogen production using methane: Techno-economics of decarbonizing fuels and chemicals. *International Journal of Hydrogen Energy*, 43(5), 2540–2555. https://doi.org/10.1016/j.ijhydene.2017.12.081

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045

Serban, M., Lewis, M. A., Marshall, C. L., & Doctor, R. D. (2003). Hydrogen production via chemical splitting of hydrogen sulfide and methane. *Energy & Fuels*, 17(3), 705–713. https://doi.org/10.1021/ef020215t

Shaikh, A., & Al-Dahhan, M. H. (2007). A review on flow regime transition in bubble columns. *International Journal of Chemical Reactor Engineering*, 5(1), R1. https://doi.org/10.2202/1542-6580.1374

Suelves, I., Lázaro, M. J., Moliner, R., Corbella, B. M., & Palacios, J. M. (2007). Hydrogen production by thermo catalytic decomposition of methane on Ni-based catalysts: Influence of operating conditions on catalyst deactivation. *Fuel*, 86(12–13), 1–9. https://doi.org/10.1016/j.fuel.2007.01.026

Upham, D. C., Agarwal, V., Khechfe, A., Snodgrass, Z. R., Gordon, M. J., Metiu, H., & McFarland, E. W. (2017). Catalytic molten metals for the carbon-free production of hydrogen. *Science*, 358(6365), 917–921. https://doi.org/10.1126/science.aao5023

