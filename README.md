# QED Flux Tube Effective Action

A high-performance numerical package for computing the non-perturbative QED effective action of cylindrically symmetric magnetic flux tubes using the Green's function method described in Mazur's thesis, [Nonperturbative Quantum Field Theory in Astrophysics](https://arxiv.org/abs/1209.4409), Chapter 7: Green's Function Method for Cylindrically Symmetric Flux Tubes.

## Features
- **Dual Backends:** 
  - **C/OpenMP:** High-precision, shared-memory parallel solver.
  - **PyTorch:** Vectorized solver for GPU acceleration and Automatic Differentiation.
- **Auto-Diff Integration:** Enables finding stationary points of the effective action with respect to magnetic field profiles.
- **Robust Numerics:** Complex-plane contour integration to handle poles along the real axis.

## Architecture
- `src/c/`: Stateless radial ODE solver implemented in C.
- `src/python/`: Python orchestrator, backend abstraction layer, and PyTorch implementation.
- `tests/`: Scientific validation and unit tests using `pytest`.

## Getting Started

### Prerequisites
- GCC with OpenMP support
- Python 3.x
- PyTorch, NumPy, Matplotlib, pytest

### Building the C Solver
```bash
make
```

### Running Tests
```bash
pytest tests/
```

## Scientific Validation
The solvers are validated by:
1. Comparing C and PyTorch backend results (matching to within machine precision).
2. Verifying the constancy of the Wronskian $r(u_0' u_\infty - u_0 u_\infty')$ across the radial domain.
