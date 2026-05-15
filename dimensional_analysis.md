# Dimensional Analysis of QED Effective Action in Magnetic Flux Tubes

## 1. Physical Units (SI/CGS vs Natural)
We work in **Natural Units** where $\hbar = c = 1$.
In these units:
- Length $[L] \sim m^{-1}$ (Inverse mass)
- Energy/Mass $[E] \sim m^1$
- Charge $e$ is dimensionless (but related to $\alpha = e^2/4\pi \approx 1/137$)
- Magnetic Field $[B] \sim m^2$ (Energy squared)
- Vector Potential $[A] \sim m^1$ (Energy)

## 2. The Effective Action $\Gamma$
The effective action is dimensionless ($[\Gamma] = 1$) because it appears in the exponent $e^{i\Gamma/\hbar}$.
The effective action density (Lagrangian density) $\mathcal{L}$ has units of $[E]/[L]^3 = [m^4]$.

From `docs/greensfunc.tex` Eq 2.50:
$$ \Gamma = \Gamma_0 + \hbar \pi \sum_{\sigma^3} \sum_{m_l} \int_0^\infty \chi^3 d\chi \int_0^\infty d\rho \rho^2 \left( \frac{u_0(\rho)u_\infty(\rho)}{W_0} - G^0(\rho, \rho) \right) $$

**Dimensional Check of Eq 2.50:**
- $\chi$ has units of mass $[m]$.
- $d\chi$ has units of mass $[m]$.
- $\rho$ has units of $[1/m]$.
- $d\rho$ has units of $[1/m]$.
- $G(\rho, \rho) = \frac{\rho u_0 u_\infty}{W_0}$ has units of $[1/m]$ (Standard 1D Green's function).
- Integrand units: $[\chi^3][d\chi][d\rho][\rho^2][G] = [m^3][m][1/m][1/m^2][1/m] = 1$.
- **Result:** $\Gamma$ is dimensionless. Correct.

## 3. Effective Lagrangian Density $\Delta \mathcal{L}(\rho)$
The action $\Gamma$ relates to the Lagrangian density $\mathcal{L}$ as:
$$ \Gamma = \int dt dz \iint dx dy \mathcal{L}(\rho) = (T \cdot L_z) \int 2\pi \rho d\rho \mathcal{L}(\rho) $$
However, Eq 2.50 represents the action per unit time and unit length along $z$.
So $\Delta \mathcal{L}(\rho)$ (the quantum correction) is derived from:
$$ 2\pi \int \rho d\rho \Delta \mathcal{L}(\rho) = \pi \sum_{\sigma^3, m_l} \int \chi^3 d\chi \int d\rho \rho^2 \Delta G_{m_l}(\rho, \rho) $$
Equating the $\rho$ integrands:
$$ 2\pi \rho \Delta \mathcal{L}(\rho) = \pi \sum_{\sigma^3, m_l} \int \chi^3 d\chi \rho^2 \Delta G_{m_l}(\rho, \rho) $$
$$ \Delta \mathcal{L}(\rho) = \frac{1}{2} \sum_{\sigma^3, m_l} \int_0^\infty \chi^3 d\chi \rho \Delta G_{m_l}(\rho, \rho) $$

**Units of $\Delta \mathcal{L}(\rho)$:**
- $[m^3][m][1/m][1/m] = [m^2]$.
- **Wait:** We expect $[m^4]$ for a 4D density. 
- **Correction:** The Green's function $G_3(\vec{x}, \vec{x})$ in Eq 2.27 is a 3D Green's function, units $[m]$.
- The 4D trace involves $\int d\omega dk_z$, which after Wick rotation becomes $\int 2\pi \chi d\chi$ (polar integration in $\omega, k_z$ plane).
- So the 4D density should be:
  $$ \Delta \mathcal{L}(\rho) \propto \iint d\omega d k_z \dots \propto \int \chi d\chi \dots $$
- But Eq 2.50 has $\chi^3 d\chi$. This suggests Eq 2.50 *already* accounts for the spatial dimensions $z$ and $t$ somehow, or is specific to a 2D problem?
- Looking at Eq 2.24-2.27: The trace is over $\omega$. $\int d\omega \int d^3x$. 
- If we have cylindrical symmetry and translation invariance in $z$, the $k_z$ integral is also present.
- $\int d\omega \int dk_z \to \int 2\pi \chi d\chi$.
- If we then integrate by parts $\int \ln(Op) = \int \omega \frac{d}{d\omega} \ln(Op) = \int \omega \frac{1}{Op} 2\omega d\omega = \int 2\omega^2 d\omega \frac{1}{Op}$.
- Wick rotation: $\omega \to i\chi_1, k_z \to \chi_2$. 
- measure $d\omega dk_z \to d\chi_1 d\chi_2 = 2\pi \chi d\chi$.
- The $\omega^2$ factor becomes $(i\chi \cos\theta)^2 = -\chi^2 \cos^2\theta$.
- Average over $\theta$: $\int_0^{2\pi} \cos^2\theta d\theta = \pi$.
- So $\int 2\omega^2 d\omega dk_z \to -2 \int \chi^2 \cos^2\theta \cdot \chi d\chi d\theta = -2\pi \int \chi^3 d\chi$.
- This confirms the $\chi^3$ measure in Eq 2.50. 
- **Units check again:** $[\chi^3][d\chi][G_{3D}] = [m^3][m][m] = [m^5]$.
- $\int d^3x [m^5] = [m^2]$. Still missing 2 dimensions?
- Ah, the 3D Green's function $G_3(\vec{x}, \vec{x})$ has units of $[L^{-1}] = [m]$. 
- The trace is $\int d^3x G_3(x, x)$, which is dimensionless? No, $\int d^3x$ is $[L^3] = [m^{-3}]$. 
- So $\int d^3x G_3(x, x)$ has units $[m^{-3}][m] = [m^{-2}]$.
- Total Action $\Gamma = \int d\omega \dots = [m][m^{-2}] = [m^{-1}]$. This matches $\hbar$ (action). 
- In $c=1$, action is dimensionless? $[\Gamma] = [E][T] = [m][m^{-1}] = 1$.
- So $\Delta \mathcal{L}$ (per unit time and length) should be $\int d^2x \mathcal{L} = [m^{-2}][m^4] = [m^2]$.
- Our units: $[\chi^3][d\chi][G_{3D}] = [m^3][m][m] = [m^5]$.
- This corresponds to $\mathcal{L}_{4D}$. $[m^4]$. 
- Wait, $G_{m_l}(\rho, \rho')$ is the *radial* part of the 3D Green's function.
- $G_{3D}(\vec{x}, \vec{x}) = \sum_{m_l} \int \frac{dk_z}{2\pi} e^{im_l \Delta\phi} e^{ik_z \Delta z} G_{m_l, k_z}(\rho, \rho')$.
- The $dk_z$ integral was already combined into the $\chi$ integral.
- So $\Delta \mathcal{L}(\rho) = \frac{1}{2} \int \chi^3 d\chi \Delta G_{m_l}(\rho, \rho) \cdot (\text{factors})$.
- $\Delta G_{m_l}$ has units of $[1/m]$? No, the radial Green's function for the operator $\partial_\rho^2 + \frac{1}{\rho}\partial_\rho$ is dimensionless?
- Let's check: $[\partial_\rho^2] G = \delta(\rho-\rho')/\rho$.
- $[m^2] [G] = [m^2]$. So $G$ is dimensionless.
- Then $\Delta \mathcal{L}(\rho) \sim \chi^3 d\chi \sim [m^4]$. **Correct.**

## 4. Implementation Analysis
In `src/python/orchestrator.py`:
```python
action = np.pi * torch.sum(action_integrand * chi_weights)
# where action_integrand = chi_real**3 * total_inner_sum
# total_inner_sum = sum_ml( Integral(num_renorm * rho) )
```
`num_results` from `PyTorchSolver` returns $\frac{\rho u_0 u_\infty}{W_0}$.
Since $u$ is dimensionless and $W_0 \sim [m]$, then `num_results` is $[m^{-1}][m] = 1$ (dimensionless).
So `total_inner_sum` is $\int \text{num\_results} \cdot \rho d\rho \sim [m^{-2}]$.
Then `action` is $[m^3][m][m^{-2}] = [m^2]$.
This matches the units of $\Gamma$ per unit time and length.

## 5. Comparison with Heisenberg-Euler (LCF)
Heisenberg-Euler density (ScQED):
$$ \mathcal{L}_{HE} = -\frac{1}{8\pi^2} \int_0^\infty \frac{dT}{T^3} e^{-m^2T} \left( \frac{eBT}{\sinh(eBT)} - 1 + \frac{1}{6}(eBT)^2 \right) $$
Units: $[T]$ is proper time $[m^{-2}]$.
Measure: $[dT/T^3] = [m^{-2}] / [m^{-6}] = [m^4]$. Correct.

In `scripts/test_periodic_lcf_comparison.py`, we implemented:
```python
L_lcf = compute_lcf_density(B_vals, factor=-(1.0 / (8.0 * np.pi**2)))
```
And `compute_lcf_density` uses `chi_vals**3 * dchi`.
Wick rotation $T = 1/\chi^2 \implies dT = -2 d\chi / \chi^3$.
Then $dT/T^3 = (-2 d\chi / \chi^3) \cdot \chi^6 = -2 \chi^3 d\chi$.
So $\int dT/T^3 \dots = \int -2 \chi^3 d\chi \dots$.
The factor of $2$ is often absorbed into the spin sum or integration range.

## 6. Discovered Discrepancies
1.  **Spin Sum:** `docs/greensfunc.tex` says "sum over $\sigma^3 = \pm 1$ and multiply by overall factor of 2 to account for degenerate eigenvalues".
    - Currently, `orchestrator.py` takes `sigma3_values` as input.
    - If user provides `[1]`, we miss the `-1` and the degenerate factor.
    - **Factor of 4 discrepancy possible here.**
2.  **Normalization of $G$:**
    - `PyTorchSolver` returns $\frac{\rho u_0 u_\infty}{W_0}$.
    - `docs/greensfunc.tex` Eq 2.50 has $\frac{\pi}{2} J Y$.
    - Standard Bessel Wronskian $J Y' - J' Y = 2/(\pi x)$.
    - Our solver computes $W_0$ numerically. We must ensure $W_0$ matches the convention where $G^0 = -\frac{\pi\rho}{2} J Y$.
    - Current code: `action = np.pi * ...`. 
    - This $\pi$ comes from Eq 2.50.
3.  **LCF Factor:**
    - The factor in LCF is $1/8\pi^2$.
    - The factor in our spectral integral is $\pi \cdot \chi^3$.
    - $1/8\pi^2$ vs $\pi$? That's a factor of $8\pi^3 \approx 248$. 
    - This explains the "orders of magnitude" difference.

## 7. Conclusion & Strategy
- The "orders of magnitude" difference is due to the discrepancy between the $1/8\pi^2$ factor in standard HE Lagrangian and the $\pi$ factor derived in the dissertation's spectral form.
- We need to reconcile the normalization of the 3D Trace with the 4D Effective Lagrangian.
- Strategy: Use the `ZeroFluxProfile` or a constant field to calibrate the `Orchestrator`'s global factor against the analytic Heisenberg-Euler density.
