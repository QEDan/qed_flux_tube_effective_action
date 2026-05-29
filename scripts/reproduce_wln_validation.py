from src.python import constants
"""
Attempts to reproduce validation figures from the WLNumerics project.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.python.orchestrator import Orchestrator
from src.python.profiles import WLNFluxTubeProfile, FieldProfile
from src.python.locally_constant_field import const_field_heisenberg_euler_lagrangian as heisenberg_euler_lagrangian

def run_validation():
    print("--- Reproducing WLN Validation (Figure 4.7) ---")
    
    device = "cpu"
    orchestrator = Orchestrator(device=device, batch_size=1024)
    
    L = 10.0
    F_dim = 10.0
    F_tot = F_dim * (2.0 * np.pi)
    m_val = 1.0
    
    rho_grid = torch.linspace(0.01, 3.0 * L, 100).to(device)
    profile = WLNFluxTubeProfile(rho_grid, lambd=L, F=F_tot)
    
    _, a_p, da_p = profile.get_arrays(as_numpy=False)
    B_vals = (a_p / (rho_grid + 1e-15) + da_p).detach().cpu().numpy()
    
    chi_grid = [complex(c) for c in np.linspace(0.1, 2.0, 15)]
    ml_grid = list(range(-100, 101))
    
    print("Computing Numerical Effective Action Density...")
    # Add a custom orchestrator call to see mode contributions
    orc = Orchestrator(device=device, batch_size=1)
    batch = [{'chi': 1.0+0j, 'ml': ml, 'sigma3': 1, 'm': m_val, 'e': 1.0} for ml in range(-10, 11)]
    res, _ = orc.backend.solve_batch(batch, profile)
    res0, _ = orc.backend.solve_batch(batch, FieldProfile(rho_grid))
    diff = (res - res0).real.detach().cpu().numpy()
    print(f"DEBUG: Mode sum (partial) at center, chi=1: {np.sum(diff[:, 0] / rho_grid[0].item())}")
    for i in range(len(batch)):
        print(f"  ml={batch[i]['ml']}, dG/rho={diff[i, 0] / rho_grid[0].item():.4e}")

    action, L_eff = orchestrator.compute_effective_action(
        profile, chi_grid, ml_grid, [1, -1], m=m_val, e=constants.ELECTRON_CHARGE
    )
    
    L_num = L_eff.real.detach().cpu().numpy()
    print(f"Max Solver Density: {np.max(np.abs(L_num))}")
    for i in range(0, 100, 10):
        print(f"rho={rho_grid[i]:.2f}: L_num={L_num[i]:.4e}")
    
    print("Computing LCF Approximation...")
    L_lcf = np.array([heisenberg_euler_lagrangian(B, m=m_val) for B in B_vals])
    print(f"LCF at center: {L_lcf[0]:.4e}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(rho_grid.cpu().numpy(), L_num, 'o', label='Solver', markersize=4)
    plt.plot(rho_grid.cpu().numpy(), L_lcf, '--', label='LCF', color='red')
    
    plt.xlabel('rho')
    plt.ylabel('L_eff')
    plt.title("WLN Validation")
    plt.grid(True)
    plt.legend()
    
    plt.savefig("results/wln_reproduction_fig47.png")
    print("Plot saved to results/wln_reproduction_fig47.png")

if __name__ == "__main__":
    run_validation()
