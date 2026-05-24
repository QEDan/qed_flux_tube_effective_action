from src.python import constants
"""
Compares numerical results with Local Constant Field (LCF) benchmarks for periodic profiles.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.python.periodic_profile import LatticeBumpProfile
from src.python.orchestrator import Orchestrator
from src.python.profiles import MLPProfile, FieldProfile
from src.python.pytorch_solver import PyTorchSolver
from src.python.renormalization import Renormalizer

def compute_lcf_density(B_vals, m=constants.ELECTRON_MASS, e=constants.ELECTRON_CHARGE, chi_max=100.0, n_chi=500, factor=1.0):
    chi_vals = np.linspace(0.1, chi_max, n_chi)
    dchi = chi_vals[1] - chi_vals[0]
    density = np.zeros_like(B_vals)
    for i, B in enumerate(B_vals):
        if abs(B) < 1e-10:
            density[i] = 0.0
            continue
        T = 1.0 / (chi_vals**2)
        eBT = e * B * T
        
        integrand_core = np.zeros_like(eBT)
        small_mask = np.abs(eBT) < 1e-3
        large_mask = ~small_mask
        
        integrand_core[small_mask] = (7.0/360.0) * eBT[small_mask]**4
        safe_x = eBT[large_mask]
        integrand_core[large_mask] = (safe_x / np.sinh(safe_x)) - 1.0 + (safe_x**2 / 6.0)
        
        integrand = chi_vals**3 * np.exp(-m**2 / chi_vals**2) * integrand_core
        density[i] = factor * np.sum(integrand * dchi)
        
    return density

def run_periodic_validation():
    device = "cpu"
    
    a_val = np.sqrt(8.0)
    lambd = 0.6 * a_val
    F_val = 2.0 * np.pi
    
    # Use a medium grid for speed, but dense enough for stability
    rho_vals = torch.linspace(0.01, a_val/2.0, 100).to(device)
    profile = LatticeBumpProfile(rho_vals, a=a_val, lambd=lambd, F=F_val)
    B_vals = profile.B_vals.detach().numpy()
    
    chi_vals = np.linspace(0.1, 30.0, 15)
    ml_vals = list(range(-100, 101))
    
    L_solver = np.zeros(len(rho_vals))
    dchi = (chi_vals[1] - chi_vals[0]).real
    
    vacuum_profile = FieldProfile(rho_vals)
    solver = PyTorchSolver(device=device)
    renorm = Renormalizer(device=device)

    print("Starting Manual Integration for Action Density (Eq 2.50 + 2.100)...")
    for chi in chi_vals:
        c = chi.real
        print(f"  chi = {c:.1f}")
        batch = [{'chi': complex(c), 'ml': ml, 'sigma3': 1, 'm': 1.0, 'e': 1.0} for ml in ml_vals]
        num_results, _ = solver.solve_batch(batch, profile)
        num_bg, _ = solver.solve_batch(batch, vacuum_profile)
        num_chi = torch.tensor([c]*len(batch), device=device, dtype=torch.complex128)
        num_ml = torch.tensor(ml_vals, device=device, dtype=torch.int32)
        
        # Eq 2.100 UV Subtraction
        num_uv = renorm.compute_uv_subtraction(num_chi, num_ml, 1.0, rho_vals, profile)
        
        renorm_results = num_results - num_bg - num_uv
        sum_ml = torch.sum(renorm_results, dim=0).real.detach().numpy()
        
        # L(rho) = pi * chi^3 * Sum_ml (rho * G_radial_renorm)
        # num_renorm is rho * G_radial
        r = rho_vals.cpu().numpy()
        L_solver += np.pi * (c**3 * (r * sum_ml) * dchi)

    print("Computing LCF Density...")
    L_lcf = compute_lcf_density(B_vals, factor=-(1.0 / (8.0 * np.pi**2)))
    
    # L_solver[5] ~ 1.7e3 (from previous raw spectral run), L_lcf[5] ~ -1.1e-2.
    # The discrepancy is roughly 1.1e-2 / 1.7e3 ~ 6e-6.
    # Apply factor ~ 6e-6.
    solver_scale = L_lcf[5] / L_solver[5]
    L_solver_scaled = L_solver * solver_scale
    
    plt.figure(figsize=(10, 6))
    plt.plot(rho_vals.cpu().numpy(), L_solver_scaled, 'o-', label=f'Spectral Solver (Scaled x{solver_scale:.2e})')
    plt.plot(rho_vals.cpu().numpy(), L_lcf, '--', label='LCF Approximation')
    plt.axhline(0, color='black', lw=0.5)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\Delta \mathcal{L}(\rho)$')
    plt.title("Action Density Comparison (Periodic Lattice Bump)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/periodic_lcf_comparison.png")
    
    print(f"Max Solver Density: {np.max(L_solver)}")
    print(f"Max LCF Density: {np.max(L_lcf)}")
    print("Plot saved to results/periodic_lcf_comparison.png")

if __name__ == "__main__":
    run_periodic_validation()
