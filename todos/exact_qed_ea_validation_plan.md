# Exact QED Effective Action Validation Plan

## 1. Objective
Validate the numerical QED Effective Action computation against the exact solutions for the magnetic field profile $B(x) = B \cdot \mathrm{sech}^2(x/\lambda)$ as derived by Dunne & Hall (1997).

## 2. Field Profile Translation
The paper defines a field $B(x) = B \cdot \mathrm{sech}^2(x/\lambda)$ in Cartesian coordinates. Our numerical solver is cylindrically symmetric. We will use a cylindrical analog $B(\rho) = B \cdot \mathrm{sech}^2(\rho/\lambda)$. 
*   **Vector Potential ($A_\phi$):**
    For $B(\rho) = \frac{1}{\rho} \frac{d}{d\rho} (\rho A_\phi)$, we solve for $A_\phi$:
    $$ A_\phi(\rho) = \frac{1}{\rho} \int_0^\rho r B(r) dr = \frac{B}{\rho} \int_0^\rho r \mathrm{sech}^2(r/\lambda) dr $$
    Using $\int r \mathrm{sech}^2(r/\lambda) dr = \lambda r \tanh(r/\lambda) - \lambda^2 \ln(\cosh(r/\lambda))$, we define:
    $$ A_\phi(\rho) = B \lambda \tanh(\rho/\lambda) - \frac{B \lambda^2}{\rho} \ln(\cosh(\rho/\lambda)) $$

## 3. Symbolic Validations
We need to verify that the effective action integrands used by our solver match the theoretical expressions for this profile:
1.  **Effective Potential Verification:** Symbolically construct $V_{eff}(\rho)$ using $A_\phi(\rho)$ and $B(\rho)$ and confirm it is well-behaved for the solver.
2.  **Effective Action Expansion:** Implement a script to calculate the first few terms of the derivative expansion (Eq \ref{full2+1} in the paper, generalized to 3+1) and ensure our renormalized action converges to this expansion as $\lambda \to \infty$ (the uniform field limit).

## 4. Implementation Plan
### 4.1 Numerical Tests
*   **Profile Class:** Create `Sech2Profile(rho, B, lambd)` inheriting from `FieldProfile`.
*   **Validation Script (`scripts/test_sech2_exact.py`):**
    *   Compute the renormalized action for the `Sech2Profile`.
    *   Compare the result against the numerical integration of Eq (\ref{2+1int}) (generalizing to 3+1 by integrating $p_3$ and multiplying by Dirac trace).

### 4.2 Diagnostic Plots
*   **Effective Action Convergence (`scripts/plot_ea_vs_field.py`):** Plot $S_{renorm}$ vs $B$ and compare with the derivative expansion.
*   **Integrand Stability (`scripts/plot_integrand_stability.py`):** Plot the renormalized integrand $\mathcal{I}_{ren}(\chi)$ to verify it decays faster than $1/\chi^3$.

## 5. Todos
- [ ] Implement `Sech2Profile` in `src/python/profiles.py`.
- [ ] Create `symbolic_validations/verify_sech2_potential.py`.
- [ ] Create `scripts/test_sech2_exact.py`.
- [ ] Update `Makefile` to include the new validation target `validate-sech2`.
