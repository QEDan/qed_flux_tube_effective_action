# Renormalization Plan: Field-Strength Renormalization for Flux Tubes

This document outlines the technical approach for implementing field-strength renormalization in the `greens_fn_numerics` project, as discussed in `greensfunc.tex`.

## 1. Overview of the Problem
The QED effective action calculation involves an integration over the frequency parameter $\chi$. This integral is ultraviolet (UV) divergent. The divergence occurs in the terms proportional to $B^2$ (field strength squared). According to the theory of renormalization, these divergences must be subtracted and absorbed into the physical field strength.

## 2. Technical Approach: WKB Subtraction
We will implement a subtraction scheme based on the WKB expansion of the Green's function. The total renormalized integrand for the effective action will be:
$$ \mathcal{I}_{ren} = G(\rho, \rho) - G^0(\rho, \rho) - G^{WKB}_{sub}(\rho, \rho) $$

### 2.1 Component 1: Background Subtraction ($G^0$)
$G^0(\rho, \rho)$ is the Green's function in the absence of any magnetic field.
- **Logic:** Compute $G^0$ analytically using Bessel functions of the first ($J_{m_l}$) and second ($Y_{m_l}$) kind.
- **Equation:** $G^0(\rho, \rho) = -\frac{\pi}{2} \rho J_{m_l}(k\rho) Y_{m_l}(k\rho)$, where $k = \sqrt{\chi^2 + m^2}$.
- **Implementation:** Use `torch.special.bessel_j0`, `bessel_j1`, etc., or `torch.special.ndtr` is not enough. We need arbitrary order $m_l$. We may need to use `scipy.special` and wrap it if PyTorch doesn't support arbitrary order Bessel functions, or use a custom implementation.

### 2.2 Component 2: UV Subtraction ($G^{WKB}_{sub}$)
This term removes the $B^2$ divergence.
- **Logic:** For a uniform field $B$, the subtraction term derived from WKB is:
  $$ G^{WKB}_{sub} = \left(\frac{eB}{2}\right)^2 \left[ \frac{\rho^3}{2k^2} \sin(\Theta) + \frac{\rho^2}{6k^3} \cos(\Theta) \right] $$
  where $\Theta = 2k\rho - \frac{1/4 - m_l^2}{k\rho}$.
- **Generalization:** For non-homogeneous fields, we use the local field $B(\rho)$ in the above expression.

## 3. Implementation Details

### 3.1 New Module: `src/python/renormalization.py`
A new module will be created to contain the renormalization logic.
- `compute_g0(chi, ml, m, rho)`: Vectorized calculation of the background Green's function.
- `compute_uv_subtraction(chi, ml, m, rho, field_profile)`: Computes the $G^{WKB}_{sub}$ terms using the local magnetic field $B(\rho)$.

### 3.2 Modifications to `Orchestrator` (`src/python/orchestrator.py`)
- Update `compute_effective_action` to:
    1. Call `solve_batch` to get $G(\rho, \rho)$.
    2. Call `compute_g0`.
    3. Call `compute_uv_subtraction`.
    4. Perform the subtraction: `results - g0 - uv_sub`.
    5. Integrate the resulting finite values.

## 3.4 Stability and Numerical Overflow Management
To prevent precision loss and numerical overflow from computing large values that analytically cancel:
- **Single-Tensor Subtraction:** $\mathcal{I}_{ren}$ will be computed as a combined tensor operation $(G - G^0 - G^{WKB}_{sub})$ to ensure floating-point cancellation occurs before integration.
- **Asymptotic Regime:** For $\chi > \chi_{threshold}$, where individual terms are large, we will transition from direct numerical computation of $G$ and $G^0$ to computing their analytic difference based on asymptotic expansions. This avoids the catastrophic cancellation of large numbers.
- **Log-Space Bessel Functions:** Where applicable, use log-space implementations (e.g., `torch.special.log_bessel_j`) to handle extreme values.


## 4. Validation Plan
1. **Convergence Check:** Ensure that the renormalized integrand $\mathcal{I}_{ren}$ decays faster than $1/\chi^3$ for large $\chi$.
2. **Step Function Benchmark:** Compare numerical results for a step function flux tube against the analytic results in Eq 2.57.
3. **Zero Field Test:** Verify that the effective action is zero when the magnetic field is zero.
