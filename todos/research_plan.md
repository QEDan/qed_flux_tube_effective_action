# Research Plan: QED Effective Action in Magnetic Flux Tubes

This document outlines the research questions and specific calculation strategies enabled by the Green's Function numerics toolset.

## 1. Core Research Questions

Based on `fluxtubes.tex` and `greensfunc.tex`, the primary research objective is to understand the quantum corrections to the energy and stability of magnetic flux tubes in dense nuclear matter (e.g., neutron star crusts).

### Q1: Quantum Stability of Flux Tubes
**Question:** Do one-loop QED corrections stabilize or destabilize classical flux tube configurations against decay or dispersion?
- [ ] Implement Gaussian and Solitonic profiles in `profiles.py`.
- [ ] Compute the effective action $\Gamma$ for varying flux tube widths $\lambda$ at fixed total flux $\Phi$.
- [ ] Identify if a local minimum in the energy functional $E = -\Gamma$ exists for non-zero $\lambda$.

### Q2: Landscape of Local Extrema and Equations of Motion
**Question:** What are the stationary configurations of the effective action, and what do they reveal about the stability of magnetic fields?
- [ ] Define a `DifferentiableProfile` using splines or a small neural network in `profiles.py`.
- [ ] Use `torch.autograd` to compute the functional derivative $\frac{\delta \Gamma}{\delta B(\rho)}$.
- [ ] Solve the quantum-corrected equations of motion (EOM) derived in `docs/greensfunc.tex` (Section 5) by finding stationary points ($\delta \Gamma = 0$).
- [ ] Map out the landscape of local minima and maxima, identifying metastable field configurations.
- [ ] Calculate the Hessian of the effective action at stationary points to determine their stability (eigenvalue analysis).

### Q3: Diffusion of Effective Action Density
**Question:** Does the effective action density "leak" into the field-free exterior region for non-integer flux values $\mathcal{F}$?
- [ ] Modify `Orchestrator` to output the integrand $\rho^2 \Delta G(\rho, \rho)$ as a function of $\rho$.
- [ ] Run calculations for integer flux ($\mathcal{F}=1, 2$) vs. non-integer flux ($\mathcal{F}=0.5, 1.5$).
- [ ] Analyze the spatial distribution of the energy density in the exterior region.

### Q4: Investigation of the Global Action Landscape
**Question:** Does the effective action $S_{\text{eff}}$ possess a global minimum, or is it unbounded? How does the numerical renormalization challenge affect this?
- [ ] **Numerical Sensitivity Analysis:** Investigate how renormalization artifacts and truncation errors affect the stability of the optimization loop.
- [ ] **Landscape Mapping:** Instead of seeking a global minimum, sweep a wide range of profile parameters to map the functional value $S_{\text{eff}}[B(\rho)]$.
- [ ] **Singularity Search:** Monitor if $|S_{\text{eff}}|$ increases indefinitely as profiles become more singular (e.g., $w_{max} \to \infty$ in basis expansion), suggesting the absence of a global minimum.
- [ ] **Metastability:** Characterize the lifetime/stability of local minima as "magnetic droplets" or metastable flux tube states.

---

## 2. Calculation & Analysis Matrix (Updated)

| Research Question | Profile to Use | Parameters to Sweep | Analysis Method |
| :--- | :--- | :--- | :--- |
| **Q1 (Stability)** | Gaussian, Solitonic | $\lambda \in [0.1/m, 10/m]$, $\Phi \in [1, 10]$ | Plot $E(\lambda)$. Check for $\frac{dE}{d\lambda} = 0$ and $\frac{d^2E}{d\lambda^2} > 0$. |
| **Q2 (EOM/Landscape)**| Differentiable Spline | Multiple initial conditions | Gradient descent/Newton's method to find stationary $B(\rho)$. Map landscape. |
| **Q3 (Diffusion)** | Step-Function | $\mathcal{F} \in [0, 2.5]$ in $0.1$ steps | Plot $\Delta\mathcal{L}(\rho)$ vs $\rho$. Measure tail decay rate for $\rho > \lambda$. |
| **Q4 (Global Behavior)**| Basis Expansion / MLP | $w_{max}$, regularization $\alpha$ | Convergence analysis of $S_{\text{eff}}$. Check for unboundedness vs. renormalization noise. |

---

## 3. Research Progress Checklist

- [x] **Phase 1: Tooling & Renormalization**
  - [x] Implement UV renormalization ($B^2$ subtraction).
  - [x] Implement background subtraction ($G^0$).
  - [x] Implement batching and memory optimizations.
  - [x] Implement asymptotic threshold for large $\chi$.

- [ ] **Phase 2: Profile Implementation**
  - [ ] Add Gaussian profile to `profiles.py`.
  - [ ] Add Solitonic profile to `profiles.py`.
  - [ ] Add Spline-based differentiable profile.

- [ ] **Phase 3: Stability Analysis (Q1)**
  - [ ] Sweep $\lambda$ for Gaussian profiles.
  - [ ] Sweep $\lambda$ for Solitonic profiles.
  - [ ] Compile energy surface results.

- [ ] **Phase 4: Landscape & EOM Investigation (Q2 & Q4)**
  - [ ] Implement stationary action search loop (finding $\delta \Gamma = 0$).
  - [ ] Develop tools for mapping local extrema and calculating Hessians.
  - [ ] Analyze the impact of renormalization schemes on landscape topology.
  - [ ] Document metastable field configurations.

- [ ] **Phase 5: Non-integer Flux (Q3)**
  - [ ] Investigate exterior integral behavior for non-integer $\mathcal{F}$.
  - [ ] Map energy density diffusion.

---

## 4. Technical Constraints & Monitoring
- **Wronskian Stability:** Monitor $|W_0(\rho) - W_{const}| / |W_0|$ during sweeps. Fail if $> 10^{-6}$.
- **Summation Convergence:** Ensure $\sum_{m_l}$ is truncated only when terms are $< 10^{-8}$ relative to the partial sum.
- **Renormalization Check:** Verify that $\mathcal{I}_{ren} \to 1/\chi^4$ as $\chi \to \infty$.
