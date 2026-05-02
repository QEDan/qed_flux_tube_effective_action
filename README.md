# QED Flux Tube Effective Action

A high-performance numerical package for computing the non-perturbative QED effective action of cylindrically symmetric magnetic flux tubes using the Green's function method described in Mazur's thesis, [Nonperturbative Quantum Field Theory in Astrophysics](https://arxiv.org/abs/1209.4409), Chapter 7: Green's Function Method for Cylindrically Symmetric Flux Tubes.

## Features
- **PyTorch Backend:** Vectorized solver for CPU/GPU execution and Automatic Differentiation.
- **Auto-Diff Integration:** Enables finding stationary points of the effective action with respect to magnetic field profiles.
- **Robust Numerics:** Complex-plane contour integration to handle poles along the real axis.

## Architecture
- `src/python/`: Python orchestrator, backend abstraction layer, and PyTorch solver implementation.
- `tests/`: Scientific validation and unit tests using `pytest`.
- `symbolic_validations/`: Mathematica/SageMath derivations used for analytic verification.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch, NumPy, Matplotlib, pytest, SciPy

### Validations
Run the full suite of numerical and symbolic validations:
```bash
make validate
```

### Running Tests
Run the standard test suite:
```bash
make test
```

## Scientific Validation
The solvers are validated by:
1. Verifying the constancy of the Wronskian $r(u_0' u_\infty - u_0 u_\infty')$ across the radial domain.
2. Analytic benchmarks: Ensuring numerical results match exact solutions for zero-flux and simple profiles.
3. Automated regression tests for renormalization and UV subtraction consistency.
