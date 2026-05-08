# Research Plan: QED Effective Action in Magnetic Flux Tubes

This document outlines the research questions and specific calculation strategies enabled by the Green's Function numerics toolset.

## 1. Core Research Questions

Based on `fluxtubes.tex` and `greensfunc.tex`, the primary research objective is to understand the quantum corrections to the energy and stability of magnetic flux tubes in dense nuclear matter (e.g., neutron star crusts).

### Q1: Quantum Stability of Flux Tubes
**Question:** Do one-loop QED corrections stabilize or destabilize classical flux tube configurations against decay or dispersion?
- [ ] Implement Gaussian and Solitonic profiles in `profiles.py`.
- [ ] Compute the effective action $\Gamma$ for varying flux tube widths $\lambda$ at fixed total flux $\Phi$.
- [ ] Identify if a local minimum in the energy functional $E = -\Gamma$ exists for non-zero $\lambda$.

### Q2: Quantum-Corrected Equations of Motion (Stationary Action)
**Question:** What is the specific field profile $B(\rho)$ that minimizes the full quantum-corrected effective action?
- [ ] Define a `DifferentiableProfile` using splines or a small neural network in `profiles.py`.
- [ ] Use `torch.autograd` to compute $\frac{\delta \Gamma}{\delta B(\rho)}$.
- [ ] Run L-BFGS optimization to find the stationary profile $B_{opt}(\rho)$.
- [ ] Compare $B_{opt}(\rho)$ to the classical Step-Function and London profiles.

### Q3: Diffusion of Effective Action Density
**Question:** Does the effective action density "leak" into the field-free exterior region for non-integer flux values $\mathcal{F}$?
- [ ] Modify `Orchestrator` to output the integrand $\rho^2 \Delta G(\rho, \rho)$ as a function of $\rho$.
- [ ] Run calculations for integer flux ($\mathcal{F}=1, 2$) vs. non-integer flux ($\mathcal{F}=0.5, 1.5$).
- [ ] Analyze the spatial distribution of the energy density in the exterior region.

### Q4: Non-Existence of a Global Minimum
**Question:** Does the effective action $S_{\text{eff}}$ possess a global minimum, or is it unbounded from below as the field profile approaches extreme configurations?
- [ ] Define the stress test procedure: Run the optimization without regularization or weight decay, and sweep the basis weight bounds $[0, w_{max}]$.
- [ ] Monitor the action magnitude: If $|S_{\text{eff}}|$ increases as $w_{max} \to \infty$ without reaching a stable stationary point, it suggests no global minimum.
- [ ] Quantify the "energy of dispersion": Calculate the effective action for a set of field profiles with varying degrees of "spikiness" to see if the action is minimized by singular field configurations.
- [ ] Document the divergence rate: Analyze the trend of the loss function vs. the field energy $\int B^2$.

---

## 2. Calculation & Analysis Matrix (Updated)

| Research Question | Profile to Use | Parameters to Sweep | Analysis Method |
| :--- | :--- | :--- | :--- |
| **Q1 (Stability)** | Gaussian, Solitonic | $\lambda \in [0.1/m, 10/m]$, $\Phi \in [1, 10]$ | Plot $E(\lambda)$. Check for $\frac{dE}{d\lambda} = 0$ and $\frac{d^2E}{d\lambda^2} > 0$. |
| **Q2 (Stationary)** | Differentiable Spline | Initial $\lambda$, fixed $\Phi$ | Gradient descent on $B(\rho)$. Compare resulting $B(\rho)$ to classical profiles. |
| **Q3 (Diffusion)** | Step-Function | $\mathcal{F} \in [0, 2.5]$ in $0.1$ steps | Plot $\Delta\mathcal{L}(\rho)$ vs $\rho$. Measure tail decay rate for $\rho > \lambda$. |
| **Q4 (Non-Existence)**| Basis Expansion | $w_{max} \in [10, 10^6]$ | Evaluate $S_{\text{eff}}$ at $w_{max}$. Check for convergence of $S_{\text{eff}}(w_{max})$. |

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

- [ ] **Phase 4: Optimization (Q2)**
  - [ ] Implement stationary action search loop.
  - [ ] Verify convergence of the field profile $B(\rho)$.

- [ ] **Phase 5: Non-integer Flux (Q3)**
  - [ ] Investigate exterior integral behavior for non-integer $\mathcal{F}$.
  - [ ] Map energy density diffusion.

---

## 4. Technical Constraints & Monitoring
- **Wronskian Stability:** Monitor $|W_0(\rho) - W_{const}| / |W_0|$ during sweeps. Fail if $> 10^{-6}$.
- **Summation Convergence:** Ensure $\sum_{m_l}$ is truncated only when terms are $< 10^{-8}$ relative to the partial sum.
- **Renormalization Check:** Verify that $\mathcal{I}_{ren} \to 1/\chi^4$ as $\chi \to \infty$.
