import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import StepFunctionProfile
from analytic import get_full_analytic_solution

def compare_analytic_vs_numerical():
    # Parameters
    lambd = 1.0
    F = 2.0 * np.pi * 1.0
    m = 1.0
    chi = 1.5 + 0.5j
    ml = 1
    sigma3 = 1
    
    # Grid: Full domain
    rho_full = np.linspace(0.1, 2.0, 400)
    
    # Use smoothed profile to allow numerical solver to handle boundary
    smooth = 0.05
    profile = StepFunctionProfile(rho_full, lambd=lambd, F=F, smooth_width=smooth)
    
    # Numerical solver
    orc = Orchestrator(backend_type="pytorch", device="cpu")
    params = [{'chi': chi, 'ml': ml, 'sigma3': sigma3, 'm': m, 'e': 1.0}]
    results_num, _ = orc.backend.solve_batch(params, profile)
    res_num = results_num[0].detach().numpy()
    
    # Full Analytic result (Matched)
    ana_full = get_full_analytic_solution(rho_full, chi, ml, sigma3, m, lambd, F)
    
    # Visualization: 3 subplots (Real, Imag, Residuals)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # 1. Real part
    axes[0].plot(rho_full, res_num.real, label="Numerical G", linestyle='--')
    axes[0].plot(rho_full, ana_full.real, label="Matched Analytic G")
    axes[0].axvline(lambd, color='k', linestyle=':', label=r'$\lambda$')
    axes[0].set_title("Green's Function Comparison: Real Part")
    axes[0].set_ylabel("Re(G)")
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Imaginary part
    axes[1].plot(rho_full, res_num.imag, label="Numerical G", linestyle='--')
    axes[1].plot(rho_full, ana_full.imag, label="Matched Analytic G")
    axes[1].axvline(lambd, color='k', linestyle=':', label=r'$\lambda$')
    axes[1].set_title("Green's Function Comparison: Imaginary Part")
    axes[1].set_ylabel("Im(G)")
    axes[1].legend()
    axes[1].grid(True)

    # 3. Residuals
    residuals = np.abs(res_num - ana_full)
    axes[2].plot(rho_full, residuals, label="Absolute Residual", color='red')
    axes[2].axvline(lambd, color='k', linestyle=':', label=r'$\lambda$')
    axes[2].set_title("Numerical Residuals (|Numerical - Analytic|)")
    axes[2].set_xlabel("Radial coordinate rho")
    axes[2].set_ylabel("Absolute Error")
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("results/analytic_vs_numerical.png")
    print("Validation complete. Plot saved to results/analytic_vs_numerical.png. Scientist: Verify high-degree overlap in the top two panels and confirm residuals in the bottom panel are near machine precision (allowing for slight peaks at the boundary due to smoothing).")

if __name__ == "__main__":
    compare_analytic_vs_numerical()
