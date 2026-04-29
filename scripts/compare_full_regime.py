import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import StepFunctionProfile
from analytic import get_full_analytic_solution, get_analytic_wronskian

def compare_full_regime():
    # Parameters
    lambd = 1.0
    F = 2.0 * np.pi * 1.0
    m = 1.0
    chi = 1.5 + 0.5j
    ml = 1
    sigma3 = 1
    
    # Grid: Interior and Exterior
    rho_int = np.linspace(0.1, lambd, 200)
    rho_ext = np.linspace(lambd, 2.0, 200)
    rho_full = np.concatenate([rho_int, rho_ext])
    
    profile = StepFunctionProfile(rho_full, lambd=lambd, F=F)
    
    # Numerical solver
    orc = Orchestrator(backend_type="pytorch", device="cpu")
    params = [{'chi': chi, 'ml': ml, 'sigma3': sigma3, 'm': m, 'e': 1.0}]
    
    # Solve interior and exterior
    results_num, w0_num = orc.backend.solve_batch(params, profile)
    res_num = results_num[0].detach().numpy()
    
    # Analytic - Full Matched Solution
    ana_full = get_full_analytic_solution(rho_full, chi, ml, sigma3, m, lambd, F, e=1.0)
    
    # Normalization (Path A) - apply to full numerical array
    # Since we corrected the analytic logic, we should check if scaling is still needed
    # or if we can use absolute comparison.
    scaling = 1.0 # Default
    # res_num_scaled = res_num / scaling
    res_num_scaled = res_num # For now, let's see raw comparison
    
    # Visualization: 2 subplots (Overlay, Residual)
    print(f"Rho[0]: {rho_full[0]}, Rho[1]: {rho_full[1]}")
    print(f"Num raw[0]: {res_num[0]}, Num raw[1]: {res_num[1]}")
    print(f"Ana Full[0]: {ana_full[0]}, Ana Full[1]: {ana_full[1]}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Overlay
    axes[0].plot(rho_full, res_num.real, label="Numerical G", linestyle='--')
    axes[0].plot(rho_full, ana_full.real, label="Matched Analytic G")
    axes[0].axvline(lambd, color='k', linestyle=':', label='$\lambda$')
    axes[0].set_title("Full Domain Green's Function (Real)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Residuals
    residuals = np.abs(res_num - ana_full)
    axes[1].plot(rho_full, residuals, label="Absolute Residual")
    axes[1].axvline(lambd, color='k', linestyle=':')
    axes[1].set_title("Residuals (Numerical - Matched Analytic)")
    axes[1].set_xlabel("Radial coordinate rho")
    axes[1].set_ylabel("Absolute Error")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("results/analytic_vs_numerical_full_residuals_v2.png")
    print("Plot saved as results/analytic_vs_numerical_full_residuals_v2.png")

if __name__ == "__main__":
    compare_full_regime()
