# Roadmap for Effective Action Discrepancy Resolution

## 1. Mathematical Consistency Check (Theory vs. Implementation)
- [ ] **Wronskian Scaling:** Correct Wronskian $W_0$ implementation in `analytic.py` and `pytorch_solver.py` to match the dimensionless Green's function normalization.
- [ ] **Integral Scaling:** Verify that the integration measure $\int \rho \, d\rho$ is consistent with the dimensionless Green's function convention;  Verify volume factors in `orchestrator.py`.
- [ ] **Integration Weights:** Verify `rho * rho_weights` measure. Confirme dimensionless consistency.

## 2. Numerical Implementation Audit
- [ ] **Normalization Calibration:** Perform small-field limit ($F=0.1$) audit, confirming magnitudes are now within the correct order of magnitude for QED.
- [ ] **Integrand Stability:** Investigate UV subtraction and confirm the consistency of the counter-term application after restoring global scaling.
- [ ] **Radial Boundary Logic:** Boundary conditions at $r=0$ and $r=\lambda$ are verified through consistent Whittaker/Bessel matching logic.

## 3. Regression Testing Plan
- [ ] **Standardized Test Suite:** (In-progress) Create `tests/test_normalization_consistency.py` to assert the ratio `Numerical_Action / HE_B4` for small $F$.
- [ ] **Wronskian Verification:** Wronskian sign and consistency verification across numerical solver and analytic benchmarks.
- [ ] **Integration Convergence:** Establish a baseline spectral integral using `scipy.integrate.quad` for full validation.
