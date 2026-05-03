# Dimensional Analysis Report

## Overview
This report documents the dimensional analysis performed on the `greens_fn_numerics` codebase. The project works in units where $\hbar=1, c=1$ (natural units).

## Core Definitions
- `rho`: Radial coordinate $[L]$.
- `m`: Mass $[M] = [L^{-1}]$.
- `e`: Charge (dimensionless in natural units).
- `A_phi`: Gauge potential $[L^{-1}]$.
- `B`: Magnetic field $[L^{-2}]$.
- `chi`: Spectral parameter (related to energy $\omega$) $[L^{-1}]$.
- `G`: Green's function, dimensionless.

## Inconsistencies & Notes

### `src/python/profiles.py`
- `FieldProfile`: Stores `rho` $[L]$, `a_phi` $[L^{-1}]$, `da_phi` $[L^{-2}]$.
- `StepFunctionProfile`: 
  - `B`: $[L^{-2}]$.
  - `A_phi` derived as `pre * f / rho`: `F` (flux) is dimensionless $[e=1, \hbar=1]$. `f` is dimensionless. $A_\phi \sim \text{Flux}/L \sim L^{-1}$. Correct.
  - `da_phi`: $\partial_\rho A_\phi \sim L^{-1}/L = L^{-2}$. Correct.
- `Sech2Profile`:
  - `B`: $[L^{-2}]$.
  - `A_phi`: $[L^{-1}]$.
  - `da_phi`: $[L^{-2}]$. Correct.

### `src/python/pytorch_solver.py`
- `get_v_eff`: 
  - $V_{ml} \sim e \sigma B + (m_l^2-1)/\rho^2 + e^2 A^2 - 2 e m_l A / \rho$.
  - $e \sigma B \sim L^{-2}$.
  - $1/\rho^2 \sim L^{-2}$.
  - $A^2 \sim L^{-2}$.
  - $A/\rho \sim L^{-1} / L = L^{-2}$.
  - All terms are dimensionally consistent $[L^{-2}]$. Correct.

### `src/python/renormalization.py`
- `compute_uv_subtraction`:
  - `uv_sub = (B/2)^2 / k^3`.
  - $B^2 \sim L^{-4}$. $k^3 \sim (L^{-1})^3 = L^{-3}$.
  - Result: $L^{-1}$. **Inconsistency detected.** The Green's function $G$ is dimensionless. The UV subtraction term should also be dimensionless.
  - *Correction needed*: The UV subtraction term should likely scale with $\rho$ to be dimensionless, or the $k^3$ in the denominator must be $k^2$ or related to $\rho$.

## Recommendations
1. Re-check the symbolic definition of the UV subtraction term in `symbolic_validations/verify_uv_subtraction.py` against `greensfunc.tex`.
2. Ensure all Green's function components are explicitly dimensionless.
