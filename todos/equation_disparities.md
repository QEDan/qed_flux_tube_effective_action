# Report: Equation Disparities and Theoretical Inconsistencies (PI Audit)

## 1. Discrepancies in Potential and Operators

### Cylindrical vs. Cartesian
The operator in `pytorch_solver.py` is:
$$ \mathcal{L} = \partial_\rho^2 + \frac{1}{\rho}\partial_\rho - V_{eff} $$
where $V_{eff} = (e A_\phi - \frac{m_l}{\rho})^2 + e \sigma_3 B + m^2 - \chi^2$.
This is correctly isomorphic to the 1D Cartesian operator $-\Pi_y^2 + e \sigma_3 B + m^2 - \chi^2$ under the mapping $p_2 \to m_l/\rho$.

### Missing Boundary Term
The theoretical step-function potential (Eq 2.75 in `greensfunc.tex`) contains an explicit $\delta$-function:
$$ V_{step}(\rho) = V_{bulk} - \frac{2\mathcal{F}}{\lambda^2}\delta(\lambda-\rho) $$
This term is missing in the numerical implementation (`profiles.py`). Consequently, the numerical solver integrates a continuous potential, missing the physical jump in the derivative $u'$ required by the theory.

## 2. Inconsistencies in Analytic Implementation

### Whittaker Parameters
The `analytic.py` code calculates $\kappa$ as:
`kappa = (lambd**2 * k2) / (4.0 * F_dim)`
This is only correct if $e=1$. The correct form should use $e F_{dim}$ in the denominator to account for the charge-flux coupling.

### Wronskian Definitions
*   **Theoretical**: $\rho W_\rho = -\frac{2\mathcal{F}}{\lambda^2} \frac{\Gamma(1+2\mu)}{\Gamma(1/2+\mu-\kappa)}$.
*   **Code (`analytic.py`)**: `W0 = (2.0 * F_dim / lambd**2) * (gamma/gamma)`. (Missing the minus sign).
*   **Code (`pytorch_solver.py`)**: `W0 = rho * (du0 * uinf - u0 * duinf)`. This is $-\rho W_\rho$.
*   **Result**: The Green's function $G = (u_0 u_\infty) / W_0$ has inconsistent signs across implementations and is globally flipped compared to the standard $L G = \delta$ definition.

## 3. Comparison Logic Failures

The current comparison script (`scripts/compare_full_regime.py`) is mathematically invalid for the following reasons:
1.  **Interior**: It uses the Green's function for an *infinite* parabolic potential, neglecting the presence of the boundary at $\lambda$.
2.  **Exterior**: It uses the *free* Green's function, neglecting the presence of the flux tube interior.
3.  **Matching**: No matching was performed to couple these regions. Stacking these two independent solutions does not yield the Green's function for a step-function profile.

## 4. Required Equation Corrections

1.  **Update $V_{eff}$**: Add the $\delta$-function term or its integrated jump condition: $\Delta u' = - \frac{2\mathcal{F}}{\lambda^2} u(\lambda)$.
2.  **Correct $\kappa$**: $\kappa = \frac{\chi^2 - m^2}{4 e F_{dim} / \lambda^2} + \frac{m_l - \sigma_3}{2}$.
3.  **Fix $W_0$**: Standardize on $\rho W_\rho$ and ensure $G = \frac{u_0 u_\infty}{\rho W_\rho}$ is implemented with consistent sign conventions (typically $G$ should be negative for $-\Delta + V + M^2$ in certain conventions, or consistent with the source term $\delta$).
