# Wrong Turns: Lessons in Green's Function Numerics

This document captures failed improvement attempts and recurring regressions encountered during development. Use this to avoid repeating the mistakes of the past.

---

## Case 1: Removing the Jump Condition in Step Profiles
**Attempted Change:** Remove the `get_discontinuities` jump condition in `StepFunctionProfile` (and its inheritance in `LocalBackgroundProfile`), assuming a step magnetic field $B$ (discontinuous but finite) should result in a continuous wave function derivative $u'$.

**Regression Symptoms:**
- **Compounding Benchmarking Errors:** `compare_analytics.py` failed with a Max Relative Error of **61%**. The error increased monotonically with $\rho$.
- **Plateau Mismatch:** The numerical effective action density plateau was significantly lower than the analytic reference ($\sim 0.00007$ vs $\sim 0.00034$).

**Lessons Learned:**
- **Mathematical Reality:** In cylindrical coordinates, the potential term $(m_l - eA_\phi)^2 / \rho^2$ with a sharp step in $A_\phi$ produces a delta-function contribution to the radial ODE.
- **Analytic Consistency:** The analytic Whittaker benchmarks (derived in the dissertation) *assume* this jump condition. Removing it in the solver creates a fundamental physical mismatch between the numerical and analytic models.
- **Rule:** Never remove jump conditions from sharp step profiles unless a corresponding change is made to the analytic benchmarks.

---

## Case 2: Normalization Factor "Trial and Error"
**Attempted Change:** Adjusting the `norm_factor` in `Orchestrator` (e.g., $1/\pi$, $1/8\pi^2$, $1/16\pi^4$) to reconcile the magnitude of the numerical effective action with the Heisenberg-Euler limit.

**Regression Symptoms:**
- **Extreme Magnitude Runaway:** The numerical density reached values of $-3000$ or $-175$ at $\rho=2$, while the physical analytic range was $\sim 10^{-4}$.
- **Sign Inversions:** The Lagrangian density flipped sign, appearing negative for fermions when it should be positive.

**Lessons Learned:**
- **Dimensional Derivation is Mandatory:** Normalization factors cannot be found by "tuning." The 4D spectral normalization factor for a 2D flux tube is exactly **$1/(4\pi^2)$**.
    - Derived from: $(1/2\pi) \times (1/2\pi) \times (1/2 \text{ Tr}) \times 4 \text{ states} = 1/(4\pi^2)$.
- **Measure Coupling:** The `norm_factor` is inextricably coupled to the spectral measure $(Q^3 - m^2 Q) dQ$. Changing one without the other will always result in a $10^6$ scale error.

---

## Case 3: Pauli-only vs. Total Heisenberg-Euler Renormalization
**Attempted Change:** Using a $b_2$ renormalization coefficient targeting only the Pauli term divergence ($(eB)^2/2$), assuming that matching the background vector potential $A$ cancels the Landau level energy shifts.

**Regression Symptoms:**
- **$O(B^2)$ Residuals:** The effective action density failed to decay at large $\rho$, settling to a constant non-zero value ($\sim 4.25$).
- **$10^4$ Discrepancy:** The numerical results were dominated by an uncancelled logarithmic divergence, making the $O(B^4)$ Heisenberg-Euler limit unreachable.

**Lessons Learned:**
- **Total Divergence Requirement:** To match the standard Heisenberg-Euler benchmark (which is defined relative to the $B=0, A=0$ vacuum), the renormalization must use the **total** spectral divergence coefficient: **$b_2 = (eB)^2/6$**.
- **Background Strategy Impact:** `NumericalBackgroundStrategy` with local $A$-matching is designed to cancel topological shifts, but it does not remove the need for standard field strength renormalization consistent with 4D QED.

---

## Case 4: Raw Bessel Initialization at Large Radii
**Attempted Change:** Initializing the backward solver `uinf` at large distances ($\rho > 100$) using standard modified Bessel functions `kv(nu, z)`.

**Regression Symptoms:**
- **Zero Lines:** Green's function comparison plots (e.g., `sech2_greens_function_comparison.png`) showed residuals consistent with zero for all radial coordinates.
- **Underflow:** The solver produced zero or near-zero results because $e^{-1000}$ is below the precision of float64.

**Lessons Learned:**
- **Stability First:** Always use **log-space initialization** or **scaled Bessel functions** (`kve`) for the backward solver.
- **Accumulator Correction:** When using scaled functions, the logarithmic accumulator `log_acc_uinf` must be updated to subtract the scaling factor (e.g., `- zm.real`) to maintain the correct Wronskian reconstruction.
