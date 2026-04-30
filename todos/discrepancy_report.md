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
*   **The $\delta$-function potential**: The theoretical potential $V_{m_l}$ in the step-function case (Eq 2.75) includes a term $-\frac{2\mathcal{F}}{\lambda^2}\delta(\lambda-\rho)$. This term results from the derivative of the step-function magnetic field in the squared Dirac operator. Its omission in `pytorch_solver.py` means the numerical solution is missing the required jump in $u'$ at the interface.
*   **Isomorphism Verification**: The mapping $p_2 \to m_l/\rho$ is correctly implemented in the solver's `v_eff`, but the solver fails to handle the resulting non-trivial boundary conditions at $\rho=\lambda$.

### Code Implementation Flaws
*   **Analytic Interior Solution**: `analytic.py` correctly identifies the Whittaker parameters $\kappa, \mu$ for $e=1$, but fails for $e \neq 1$.
*   **Solver Wronskian**: The numerical Wronskian $W_0$ in `pytorch_solver.py` is calculated as $\rho(u_0' u_\infty - u_0 u_\infty')$. By Abel's theorem and the definition of the Green's function $G = \frac{u_0 u_\infty}{\rho W_\rho}$, where $W_\rho = u_0 u_\infty' - u_0' u_\infty$, we have $W_0 = -\rho W_\rho$. Thus, `results = (u0 * uinf) / W0` produces $-G$.
*   **Exterior BCs**: For Euclidean modes where $\chi^2 > m^2$, the exterior ODE is the standard Bessel equation (not modified). The boundary condition $\psi \sim e^{-k\rho}$ is incorrect for these oscillatory modes.

### Benchmarking Errors (`compare_full_regime.py`)
*   The student uses `ana_int = (u0_ana * uinf_ana) / W0_ana` for $\rho < \lambda$ and a free-field $G_{free} \propto J Y$ for $\rho > \lambda$.
*   This approach is fundamentally flawed: the true $u_\infty$ in the interior is a linear combination of Whittaker $M, W$ that matches the exterior Bessel solution. Conversely, $u_0$ in the exterior is a combination of $J, Y$ matching the interior Whittaker solution. Stacking two "unmatched" solutions is not a valid benchmark for the step-function flux tube.

## 3. Revised Strategy for Graduate Student
1.  **Correct `analytic.py`**: Include the `e` parameter in $\kappa$ and apply the missing minus sign to the Wronskian.
2.  **Implement Analytic Matching**: Update `analytic.py` to solve the matching matrix at $\rho=\lambda$ to produce the *true* analytic Green's function for the step-profile.
3.  **Update Numerical Potential**: Modify `profiles.py` to include the $\delta$-function (via a numerical jump or a smoothed approximation) or update the RK4 loop to handle the jump condition explicitly.
4.  **Fix Global Sign**: Align the sign of $W_0$ in the numerical solver with the Green's function definition.
5.  **Address Asymptotics**: Implement correct boundary conditions for oscillatory modes ($H^{(1)}$ or $J, Y$) or restrict analysis to the decaying regime ($\chi^2 < m^2$).
