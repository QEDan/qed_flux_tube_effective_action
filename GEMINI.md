# Project Standards: Green's Function Numerics for QED

This project is a high-precision scientific instrument for computing quantum effective actions of cylindrically symmetric magnetic flux tubes. Maintenance and development require adherence to rigorous engineering and scientific standards.

## 1. Scientific Integrity Mandate
All code changes that affect mathematical results or physical observables MUST be validated against known benchmarks when possible.
- **Verification vs. Validation:** Verification confirms the code solves the equations correctly (e.g., matching numerical ODE solutions to Whittaker analytic forms). Validation confirms the equations describe the physics correctly (e.g., matching the Local Constant Field approximation in slowly varying limits).
- **Symbolic Grounding:** Use SageMath or SymPy in the `symbolic_validations/` directory to derive and confirm every derivation, normalization factor, spectral measure, and coordinate transform identity.
- **Dimensional Consistency:** Always verify the units/dimensions of every term in an expression. Document expected dimensionalities in docstrings (and comments where needed). We are working in Heaviside-Lorentz Natural Units where c=hbar=1. A change in measure (e.g., $Q dQ \to Q^3 dQ$) requires a re-derivation of the associated normalization coefficients.

## 2. Development Workflow
### Research Phase
- Identify relevant equations in `docs/*.tex`.
- Use `grep_search` to find existing implementations of related mathematical identities.
- Perform empirical reproduction: Before fixing a bug, create a script in `debug/` or a test case that demonstrates the failure state.

### Execution Phase
- **Surgical Edits:** Maintain high code quality; avoid broad refactors unless they are the primary task.
- **Sign & Factor Checks:** Sign errors and missing $2\pi$ factors are the most common sources of orders-of-magnitude errors. Every such factor must be documented in code comments with a reference to the source (e.g., "Eq 2.45 in greensfunc.tex").
- **Numerical Stability:** Use safe radial denominators (`r_safe`) and robust boundary condition initializations (log-space mpmath) to prevent NaNs at the origin or underflows at infinity.

### Validation Phase
- **Benchmark Alignment:** New features must match established benchmarks (Heisenberg-Euler, Step Function Whittaker solutions, WKB, etc.).
- **Regression Testing:** Run `pytest tests` to ensure changes have not broken any tests. Add new tests to prevent future regressions.

## 3. Tool-Specific Instructions
- **PyTorchSolver:** All radial integrations using RK4 must use a sufficiently fine grid. Grids with fewer than 100 points should be used with extreme caution. Coarse grids can be used when testing functionality, but should never be set as default.
- **Renormalizer:** The `NumericalBackgroundStrategy` must always match the local vector potential $A_\phi$ of the interacting case to ensure cancellation of topological vacuum shifts.
- **Orchestrator:** Spectral integration must use the 4D-correct $Q^3 dQ$ measure unless explicitly working in a lower-dimensional theory.

## 4. Plotting
- Every axis should have a label and include units.
- LaTeX math symbols are preferred for axes labels
- Units should be natural units with the c's, e's, and hbar's explicit (e.g. Magnetic Flux $$\left(\frac{\pi \hbar}{e}\right)$$
- File names should make clear what the plot shows (e.g. EA_vs_lambda_step_profile.png)

## 5. Documentation
- Create detailed docstrings that connect implementations to the relevant source materials, symbolic_validations, and equations. However, the code should be understandable on its own, so repeat important information if it is not part of the codebase. 
- All significant bug fixes should be accompanied by a diagnostic script in `debug/` that illustrates the "before" and "after" state, ensuring the fix is rooted in data, not just intuition. Tests should be created to prevent future regressions.

## 6. Environment
When running code:
- Use the "sage" conda environment
- Add the project root directory to the PYTHONPATH environment variable