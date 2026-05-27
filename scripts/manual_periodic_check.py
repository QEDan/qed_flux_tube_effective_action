from src.python import constants
"""
Performs manual verification of periodic boundary condition behavior in the solver.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.python.periodic_profile import LatticeBumpProfile
from src.python.orchestrator import Orchestrator
from src.python.profiles import FieldProfile
from src.python.analytic_step_profile import heisenberg_euler_lagrangian

def compute_lcf_density(B_vals, m=constants.ELECTRON_MASS, e=constants.ELECTRON_CHARGE):
    density = np.zeros_like(B_vals)
    for i, B in enumerate(B_vals):
        density[i] = heisenberg_euler_lagrangian(B, m=m, e=e)
    return density

def run_manual_validation():
    device = "cpu"
    orchestrator = Orchestrator(device=device, batch_size=512)
    
    a_val = np.sqrt(8.0)
    lambd = 0.6 * a_val
    F_val = 2.0 * np.pi
    
    rho_vals = torch.linspace(0.01, a_val/2.0, 500).to(device)
    profile = LatticeBumpProfile(rho_vals, a=a_val, lambd=lambd, F=F_val)
    B_vals = profile.B_vals.detach().numpy()
    
    # We'll use a chi grid consistent with the orchestrator's expectations
    chi_vals = [complex(c) for c in np.linspace(0.1, 50.0, 15)]
    ml_vals = list(range(-50, 51))
    sigma3_vals = [1] # Scalar QED equivalent or single spin state
    
    print("Starting Orchestrator Integration...")
    # Compute effective action and Lagrangian density
    action, L_eff_rho = orchestrator.compute_effective_action(
        profile, chi_vals, ml_vals, sigma3_vals, m=constants.ELECTRON_MASS, e=constants.ELECTRON_CHARGE, collect_density=True
    )
    
    # Extract density values
    L_solver_scaled = L_eff_rho.detach().cpu().numpy()

    # 1. Compare Solver vs LCF at a central constant field point
    # Select rho = 0 (constant B region)
    idx = 0
    solver_density = L_solver_scaled[idx]
    b_val = B_vals[idx]
    
    # Theoretical EH density at this B_val
    theoretical_density = heisenberg_euler_lagrangian(b_val, m=constants.ELECTRON_MASS, e=constants.ELECTRON_CHARGE)
    
    print(f"\n--- Normalization Audit ---")
    print(f"B_val at center: {b_val:.4e}")
    print(f"Solver Density at center: {solver_density.real:.4e}")
    print(f"Theoretical EH Density at center: {theoretical_density:.4e}")
    if abs(theoretical_density) > 1e-15:
        print(f"Normalization Ratio: {solver_density.real / theoretical_density:.4e}")

    L_lcf = compute_lcf_density(B_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(rho_vals.cpu().numpy(), rho_vals.cpu().numpy() * L_solver_scaled.real, 'o-', label='Numerical Solver')
    plt.plot(rho_vals.cpu().numpy(), rho_vals.cpu().numpy() * L_lcf, '--', label='LCF Approximation')
    plt.axhline(0, color='black', lw=0.5)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\rho \times \Delta \mathcal{L}(\rho)$')
    plt.title("Action Density: Solver vs LCF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/periodic_manual_check.png")
    print("Plot saved to results/periodic_manual_check.png")
    
    print(f"Max Solver Density: {np.max(np.abs(L_solver_scaled))}")
    print(f"Max LCF Density: {np.max(np.abs(L_lcf))}")

if __name__ == "__main__":
    run_manual_validation()
