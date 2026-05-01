# Report: Equation Disparities and Theoretical Inconsistencies (PI Audit)

## 1. Discrepancies in Potential and Operators

### Cylindrical vs. Cartesian
The operator in `pytorch_solver.py` is:
$$ \mathcal{L} = \partial_\rho^2 + \frac{1}{\rho}\partial_\rho - V_{eff} $$
where $V_{eff} = (e A_\phi - \frac{m_l}{\rho})^2 + e \sigma_3 B + m^2 - \chi^2$.
This is correctly isomorphic to the 1D Cartesian operator $-\Pi_y^2 + e \sigma_3 B + m^2 - \chi^2$ under the mapping $p_2 \to m_l/\rho$.

## 2. Inconsistencies in Analytic Implementation

### Wronskian Definitions
*   **Theoretical**: $\rho W_\rho = -\frac{2\mathcal{F}}{\lambda^2} \frac{\Gamma(1+2\mu)}{\Gamma(1/2+\mu-\kappa)}$.
*   **Result**: The Green's function $G = (u_0 u_\infty) / W_0$ has inconsistent signs across implementations and is globally flipped compared to the standard $L G = \delta$ definition.

## 3. Required Equation Corrections

1.  **Fix $W_0$**: Standardize on $\rho W_\rho$ and ensure $G = \frac{u_0 u_\infty}{\rho W_\rho}$ is implemented with consistent sign conventions (typically $G$ should be negative for $-\Delta + V + M^2$ in certain conventions, or consistent with the source term $\delta$).
