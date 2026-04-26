# Implementation Strategy: Green's Function for Magnetic Flux Tubes

## 1. Mathematical Framework
The goal is to compute the QED effective action $\Gamma$ for a cylindrically symmetric magnetic flux tube $B(\rho)$. The effective action is expressed as:
$$\Gamma = \Gamma_0 + \hbar\pi \sum_{\sigma^3 \in \{\pm 1\}} \sum_{m_l = -\infty}^\infty \int_0^\infty \chi^3 d\chi \int_0^\infty d\rho \rho^2 \left( \frac{u_0(\rho)u_\infty(\rho)}{W_0} - G^0(\rho, \rho) \right)$$
where $u_0(\rho)$ and $u_\infty(\rho)$ are solutions to the radial ODE:
$$\left(-\frac{d^2}{d\rho^2} - \frac{1}{\rho}\frac{d}{d\rho} + V_{m_l}(\rho) - \chi^2 + m^2 + \frac{1}{\rho^2}\right)u(\rho) = 0$$
with $V_{m_l}(\rho)$ defined by the magnetic field profile $A_\phi(\rho)$.

### 1.1 Pole Analysis
The Green's function $G(\rho, \rho') = \frac{\rho u_0(\rho_{<}) u_\infty(\rho_{>})}{W_0}$ contains poles where the Wronskian $W_0(\chi) = 0$. These poles correspond to the eigenvalues of the radial operator. We will use complex-plane contour integration ($\chi \to \chi + i\epsilon$) to remain numerically robust in the presence of these poles.

## 2. Numerical Strategy (Stateless & Batch-Oriented)

### 2.1 Stateless Radial ODE Solver
To facilitate both C/OpenMP and PyTorch backends, the solver must be **stateless**:
- **Inputs:** All parameters ($\chi, m_l, \sigma^3$, field profile $A_\phi$) must be passed explicitly.
- **Independence:** Each $(\chi, m_l)$ point calculation must be independent of others, allowing for trivial vectorization.

### 2.2 Numerical Integration
- **Contour Integration:** Integration over $\chi$ will be performed along a shifted contour in the complex plane.
- **Adaptive Quadrature:** Use algorithms like Gauss-Kronrod, implemented in a way that can be batched for PyTorch.

## 3. Software Architecture

### 3.1 Backend Abstraction Layer
The package will feature a clear separation between the **Orchestrator** and the **Computation Backend**.

- **The Orchestrator (Python):** Handles grid generation for $\chi$ and $m_l$, UV renormalization logic, and high-level optimization loops.
- **The Solver Interface:** A unified API that accepts a batch of parameters and returns the Green's function values.
  - `BaseSolver.solve_batch(params, field_profile)`

### 3.2 Computation Backends
1.  **C/OpenMP Backend:** 
    - Used for high-precision validation and initial research.
    - Implemented as a C extension (e.g., via `pybind11` or `ctypes`).
    - Focuses on **Task Parallelism** across CPU cores.
2.  **PyTorch Backend (Primary for Scale):**
    - Uses **Data Parallelism** (vectorization).
    - Implements the ODE solver using PyTorch-compatible primitives (e.g., `torch.func` for vectorization or custom autograd functions).
    - Enables execution on **GPUs** for massive speedups in the $m_l$ and $\chi$ summation/integration.

## 4. Stationary Action Search via Auto-Diff
Finding the stationary points ($\delta \Gamma = 0$) is a core objective.
- **Functional Representation:** Represent the field profile $A_\phi(\rho)$ as a differentiable parameterization (e.g., a spline with learnable coefficients or a neural network).
- **Auto-differentiation:** Instead of manually deriving Eq 2.45, we will use PyTorch's `autograd` to compute the gradient of the effective action with respect to the field profile parameters:
  $$\nabla_{f} \Gamma \approx \frac{\partial \Gamma}{\partial \theta_f}$$
- **Optimization:** Use PyTorch optimizers (like L-BFGS or Adam) to find the profile that extremizes the action.

## 5. Parallelization Strategy
1.  **CPU Level:** OpenMP handles threads for individual parameter sets.
2.  **GPU Level:** PyTorch vectorizes the solver across thousands of $(\chi, m_l)$ pairs simultaneously.
3.  **Memory Management:** Use a streaming approach for very large grids to avoid exceeding GPU VRAM.

## 6. Verification and Validation
- **Benchmark:** Compare PyTorch results against the C/OpenMP results and the analytic Whittaker solutions.
- **Wronskian Check:** Monitor $|W_0(\rho) - W_{const}|$ to ensure solver stability across different backends.

## 7. Implementation Phases
1.  **Phase 1: Foundation.** Implement the Orchestrator and the C/OpenMP reference solver.
2.  **Phase 2: PyTorch Vectorization.** Implement the batch-oriented ODE solver in PyTorch.
3.  **Phase 3: Auto-Diff Integration.** Set up the optimization loop for finding stationary field profiles.
4.  **Phase 4: Scaling.** Optimize GPU memory usage and implement full UV renormalization.
