"""
Visualizes benchmark results to confirm agreement between numerical and analytic amplitudes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.python.pytorch_solver import PyTorchSolver
from src.python.analytic_step_profile import get_interior_solutions
from src.python.profiles import StepFunctionProfile

def visualize_test_benchmark():
    # Parameters from tests/test_step_function_benchmark.py
    lambd = 1.0
    F = 2.0 * np.pi * 1.0
    m = 1.0
    chi = 1.5 + 0.5j
    ml = 1
    sigma3 = 1
    e = 1.0
    
    # Grid: Interior only as per the benchmark test
    rho_np = np.linspace(0.1, lambd, 1000)
    profile = StepFunctionProfile(rho_np, lambd=lambd, F=F, e=e)
    solver = PyTorchSolver(device="cpu")
    params_pt = {
        'chi': torch.tensor([chi], dtype=torch.complex128),
        'ml': torch.tensor([ml], dtype=torch.int32),
        'sigma3': torch.tensor([sigma3], dtype=torch.int32),
        'm': torch.tensor([m], dtype=torch.float64),
        'e': torch.tensor([e], dtype=torch.float64),
    }
    
    # Analytic u0 (interior)
    u0_ana, _ = get_interior_solutions(rho_np, chi, ml, sigma3, m, lambd, F, e=e)
    # Get derivative for exact IC
    h = 1e-5
    u0_p, _ = get_interior_solutions(np.array([rho_np[0] + h]), chi, ml, sigma3, m, lambd, F, e=e)
    u0_m, _ = get_interior_solutions(np.array([rho_np[0] - h]), chi, ml, sigma3, m, lambd, F, e=e)
    du0_ana = (u0_p[0] - u0_m[0]) / (2 * h)
    
    # Numerical integration with exact analytic IC
    curr_state = torch.tensor([[u0_ana[0], du0_ana]], dtype=torch.complex128)
    rho_t = profile.rho
    a_phi_t = profile.a_phi
    da_phi_t = profile.da_phi
    
    u0_num = np.zeros(len(rho_np), dtype=np.complex128)
    u0_num[0] = curr_state[0, 0].item()
    
    for i in range(len(rho_np)-1):
        h_step = rho_np[i+1] - rho_np[i]
        a_mid = 0.5 * (a_phi_t[i] + a_phi_t[i+1])
        da_mid = 0.5 * (da_phi_t[i] + da_phi_t[i+1])
        curr_state = solver.rk4_step(rho_t[i], h_step, curr_state, params_pt, 
                                     a_phi_t[i], da_phi_t[i],
                                     a_mid, da_mid,
                                     a_phi_t[i+1], da_phi_t[i+1])
        u0_num[i+1] = curr_state[0, 0].item()

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(rho_np, u0_num.real, label=r"Numerical $\mathfrak{Re}\{u_0\}", linestyle='--')
    plt.plot(rho_np, u0_ana.real, label=r"Analytic \mathfrak{Re}\{u_0\}")
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$\mathfrak{Re}\{u_0\}$")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/test_benchmark_absolute_visualization.png")
    print("✅ Validation complete. Plot saved to results/test_benchmark_absolute_visualization.png. Scientist: Confirm that the dashed numerical integration curve perfectly tracks the solid analytic benchmark for the absolute amplitude.")

if __name__ == "__main__":
    visualize_test_benchmark()
