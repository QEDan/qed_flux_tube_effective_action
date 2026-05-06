
# Validation Checklist

We will implement validations to confirm that this software is a valid scientific instrument.

## The Delta Function Shell

$B(\rho) = \Phi \delta(\rho - R)$

This profile allows for exact boundary matching of the quantum fields without needing continuous integration through a gradient.

## Exactly Solvable Configurations

The $\text{sech}^2$ profile is exactly solvable in cartesian coordinates:
$\vec{B}(\vec{x}) = B_0 \text{sech}^2\left(\frac{x}{\lambda}\right)\hat{z}$.

The field and solution are described in ~/.gemini/tmp/greens-fn-numerics/9710062.tex.

The 1D Cartesian profile, $B(x) = B_0 \text{sech}^2(x/\lambda)$, is analytically tractable because the corresponding Dirac or Klein-Gordon equation reduces to a 1D Schrödinger equation with a reflectionless Pöschl-Teller potential. If we try to center that same profile radially, $B(\rho) = B_0 \text{sech}^2(\rho/\lambda)$, the required gauge field integration $A_\theta = \frac{1}{\rho}\int \rho' B(\rho') d\rho'$ and the centrifugal term $\frac{m_l^2}{\rho^2}$ destroy that specific mathematical symmetry. However, we can still use the 1D Cartesian exact solution to validate your cylindrical solver by using an asymptotic geometric matching approach.

### The Large-Radius Shell Limit

Instead of centering the $\text{sech}^2$ profile at the origin, you can displace it to a large radius $R$ to create a thin cylindrical shell. As the radius of this shell approaches infinity, the local curvature vanishes, and the physics must converge to the 1D Cartesian domain wall.

1. Define the Displaced ProfileInput the following magnetic field profile into your cylindrical solver:$$B(\rho) = B_0 \text{sech}^2\left(\frac{\rho - R}{\lambda}\right)$$Ensure that your chosen radius is much larger than the width of the domain wall ($R \gg \lambda$).
2. Establish the Asymptotic MetricIn the 1D Cartesian exact solution, you are calculating an effective energy density per unit area (the $x$-$y$ plane), which we can call $\mathcal{E}_{\text{1D}}$.In your 2D cylindrical solver, you will compute the total effective energy per unit length in the $z$-direction, which we can call $\mathcal{E}_{\text{cyl}}(R)$.
3. The Validation TestTo validate the solver, calculate $\mathcal{E}_{\text{cyl}}(R)$ for progressively larger values of $R$. Divide your numerical result by the circumference of the shell ($2\pi R$). Your numerical output must satisfy the following limit:$$\lim_{R \to \infty} \frac{\mathcal{E}_{\text{cyl}}(R)}{2\pi R} = \mathcal{E}_{\text{1D}}$$
#### Limitations and the Purely Cylindrical Alternative

While the large-radius limit is mathematically rigorous, it is computationally expensive. Because the magnetic field is localized far from the origin at $R$, the active physics shifts to very high angular momentum quantum numbers. You will need to sum over a massive number of $m_l$ modes to achieve convergence.

## Asymptotics and Renomalization Checks

Numerical stability in the deep ultraviolet (UV) regime is a common failure point for effective action solvers.
* WKB Approximation Limit: In the limit of asymptotically large momentum $k$, verify that your numerical homogeneous solutions $u_0(\rho)$ and $u_\infty(\rho)$ converge to the expected WKB expansions. Mathematics needed for this check are worked out in symbolic_validations/derive_wkb_limit.sage.
    * **Hypotheses to Test:**
        1. **Phase-Drift Linear Dependence:** At high $\chi$, numerical phase errors may cause $y_0$ and $y_\infty$ to become nearly parallel, leading to an artificially small Wronskian $W_y$ and inflated Green's function amplitude.
        2. **Solution Mixing:** Numerical noise in forward integration of $y_0$ (the "regular" solution) may pick up the "irregular" $y_\infty$ component, destroying the independence of the two solutions required for the Wronskian.
        3. **Higher-Order WKB Corrections:** The observed shift in the zoomed oscillatory match (approx 0.0075) may be due to $O(1/k\rho)$ terms neglected in the leading-order WKB expansion used for the visual fit.
        4. **Discretization Bias:** Even with 10,000 points, the rapid oscillations at $\chi=200$ (approx 60 cycles) may require a symplectic integrator or a non-uniform grid to maintain phase integrity.
* Flux Quantization Check: When the dimensionless flux measure $\mathcal{F}$ is an exact integer, evaluate the exterior integral ($\rho > \lambda$). Integer values correspond to the disappearance of the Aharonov-Bohm effect, and your solver should yield a zero or cleanly canceling exterior integral in these limits.  Mathematics needed for this validation are outlined in symbolic_validations/derive_flux_quantization.sage.

## Other Validations

1.  **Induced Charge Density**: Verify the vacuum polarization charge density $\rho_{ind}(r) = e \langle \bar{\psi}\gamma^0\psi \rangle$ for a step function flux tube. This is a well-studied result in QED flux tubes (e.g., Sivers, 1980s) and is less sensitive to global integration measure errors than the total action.
2.  **Landau Level Convergence**: For a constant $B$ interior, the partial trace $\text{Tr} G_{m_l}(r, r)$ should show peaks corresponding to Landau levels. We can verify the solver by comparing the numerical mode-sum density with the analytic Landau level density in the interior region.
3.  **Schwinger Rate (Imaginary Part)**: For fields $B > m^2/e$, the effective action develops an imaginary part representing pair production. Validating the ratio $\text{Im}(S_{num}) / \text{Im}(S_{Schwinger})$ for a supercritical field provides a direct test of the high-field non-perturbative regime.
4.  **Small-Mass Asymptotics**: The effective action has a known logarithmic scaling as $m \to 0$. Testing if $\partial S / \partial \log(m)$ matches the expected coefficient (proportional to the trace anomaly) would verify the global renormalization scale.

## Recommended Test Parameter Regimes

To ensure comprehensive coverage across both the classical and quantum-dominated regimes, execute the validation plan across the following parameter combinations:
| Parameter	| Symbol	| Recommended Test Values	| Justification  |
| ---           |   ---         |              ---              |   ---          |
|Dimensionless Flux	| F	|1, 2, 5.5	| Integer values test Aharonov-Bohm cancellations; non-integers test general phase effects.   |
| Tube Radius	| λ	| 0.1λC​, 1.0λC​, 10λC​	 | Tests both the narrow (neutron star core scale) and wide (laboratory superconductor scale) derivative expansion limits.   |
| Field Strength	| B0​	| 0.1Bc​, 1.0Bc​, 10Bc​ | Verifies sub-critical and supercritical (non-perturbative) field dynamics.  |
| Angular Momentum	| ml​	| 0, 1, ml​≫F | Small ml​ probes the core; large ml​ verifies Bessel function asymptotic cancellation.|


## The Checklist

- [ ] Delta-Function Shell (Mathematical framework validated in `symbolic_validations/derive_delta_shell.sage`)
- [ ] Sech2 Profile (Mathematical framework validated in `symbolic_validations/derive_sech2_shell.sage`)
- [ ] WKB Approximation Limit (Mathematical framework validated in `symbolic_validations/derive_wkb_limit.sage`)
- [ ] Flux Quantization Check (Mathematical framework validated in `symbolic_validations/derive_flux_quantization.sage`)
- [ ] Induced Charge Density (Mathematical framework defined in `symbolic_validations/derive_induced_charge.sage`)
- [ ] Landau Level Convergence (Mathematical framework defined in `symbolic_validations/derive_landau_levels.sage`)
- [ ] Schwinger Rate (Mathematical framework defined in `symbolic_validations/derive_schwinger_rate.sage`)
- [ ] Small-Mass Asymptotics (Mathematical framework defined in `symbolic_validations/derive_small_mass.sage`)