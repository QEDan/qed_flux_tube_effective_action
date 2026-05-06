# Gauge Consistency Limitations

This directory contains test suites that demonstrate a fundamental limitation in the current numerical solver regarding Aharonov-Bohm topological invariance.

## Tests

- `tests/test_sech2_flux_cancellation.py`: Checks if the solver correctly reproduces the topological mode-shifting property ($m_l \to m_l - F$) expected in the presence of an Aharonov-Bohm flux tube.
- `tests/test_global_action_invariance.py`: Verifies if the global effective action (summed over all angular momentum modes $m_l$) remains invariant under integer flux shifts.

## Observations

1.  **Mode-Mapping Failure:** The solver fails to map modes $m_l$ to $m_l'$ in the presence of flux, showing a non-trivial, mode-dependent interaction rather than a gauge-covariant shift.
2.  **Topological Invariance Failure:** The global vacuum energy (effective action) is not invariant under integer flux shifts, proving that the phase accumulation in the numerical integrator is not gauge-covariant.
3.  **Numerical Integrator Bias:** The current RK4 approach with discrete boundary matching at a shell interface is sensitive to grid discretization, which cascades phase errors and breaks topological coherence.

## Reason for Limitation

The solver's current architecture relies on direct numerical integration of the radial equation with a localized vector potential $A_\phi$. While accurate for calculating local energy densities, this method fails for topological phases because it does not treat the flux as a global constraint on the Hilbert space. The gauge transformation required to eliminate the flux is not implemented as an operator shift on the angular momentum indices, resulting in a solver that behaves as if the flux were a localized physical scatterer rather than a topological vacuum constraint.
