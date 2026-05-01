# Report: Investigation of Analytic vs. Numerical Discrepancy (PI Review)

## Executive Summary
A comprehensive audit of the repository reveals that the "qualitative disagreement" between numerical results and analytic benchmarks is primarily due to a **conceptual failure in the benchmark construction** and **missing physical terms in the numerical solver**. The numerical code is integrating a different problem than the one defined in the analytic benchmark, and both are missing critical components of the theoretical model described in the dissertation (`greensfunc.tex`).

## 1. Verified Findings & Critical Issues

| Issue | Description | Status |
| :--- | :--- | :--- |
| **Flawed Benchmark** | `compare_full_regime.py` used uncoupled analytic regions. | **Fixed**: Matched analytic solution implemented. |
| **Solver Indexing** | `pytorch_solver.py` only populated half the domain for $u_0, u_\infty$. | **Fixed**: Full domain integration corrected. |
| **Missing $\delta$-potential** | `greensfunc.tex` (Eq 2.75) identifies a $\delta$-function at $\rho=\lambda$. | **Fixed**: Jump conditions implemented in Python and C solvers. |
| **Global Sign/Norm Errors** | Missing $\rho$ factor and inconsistent Wronskian sign. | **Fixed**: $G = \rho u_0 u_\infty / W_0$ alignment. |
| **Charge `e` Neglect** | `analytic.py` ignores the `e` parameter. | **Fixed**: `e` correctly used in $\kappa$ and flux parameters. |
| **Signature Mismatch** | Solver used $e^{-k\rho}$ decay for oscillatory Bessel regimes. | **Fixed**: Regime-specific BCs ($Y_n$ for oscillatory, $K_n$ for decaying) implemented. |

## 2. Detailed Technical Audit

### Mathematical Foundations
*   **Isomorphism Verification**: The mapping $p_2 \to m_l/\rho$ is correctly implemented in the solver's `v_eff`, but the solver fails to handle the resulting non-trivial boundary conditions at $\rho=\lambda$.

### Code Implementation Flaws
*   **Solver Wronskian**: The numerical Wronskian $W_0$ in `pytorch_solver.py` is calculated as $\rho(u_0' u_\infty - u_0 u_\infty')$. By Abel's theorem and the definition of the Green's function $G = \frac{u_0 u_\infty}{\rho W_\rho}$, where $W_\rho = u_0 u_\infty' - u_0' u_\infty$, we have $W_0 = -\rho W_\rho$. Thus, `results = (u0 * uinf) / W0` produces $-G$.

## 3. Revised Strategy for Graduate Student
1.  **Fix Global Sign**: Align the sign of $W_0$ in the numerical solver with the Green's function definition.
