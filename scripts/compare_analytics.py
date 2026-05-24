import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.python import constants
"""
Compares numerical Green's function results against analytic Whittaker/Bessel benchmarks.
"""

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
    m = constants.ELECTRON_MASS
    chi = 1.5 + 0.5j
    ml = 1
    sigma3 = 1
    
    # Grid: Focus on the interior and matching region
    rho_full = np.linspace(0.01, 2.0, 500)
    
    # Use sharp profile to match analytic benchmark exactly
    profile = StepFunctionProfile(rho_full, lambd=lambd, F=F, smooth_width=None)
    
    # Numerical solver
    orc = Orchestrator(device="cpu")
    params = [{"chi": chi, "ml": ml, "sigma3": sigma3, "m": m, "e": constants.ELECTRON_CHARGE}]
    results_num, W0_num = orc.backend.solve_batch(params, profile)
    res_num = results_num[0].detach().numpy()
    
    # Full Analytic result (Matched)
    ana_full = get_full_analytic_solution(rho_full, chi, ml, sigma3, m, lambd, F)
    
    # Visualization: 3 subplots (Real, Imag, Residuals)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # 1. Real part
    axes[0].plot(rho_full, res_num.real, label=r"$G_{\rm num}(\rho, \rho)$", linestyle='--')
    axes[0].plot(rho_full, ana_full.real, label=r"$G_{\rm ana}(\rho, \rho)$")
    axes[0].axvline(lambd, color='k', linestyle=':', label=r'$\lambda$')
    axes[0].set_ylabel(r"$\mathfrak{Re}\{G(\rho, \rho)\}$")
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Imaginary part
    axes[1].plot(rho_full, res_num.imag, label=r"$G_{\rm num}(\rho, \rho)$", linestyle='--')
    axes[1].plot(rho_full, ana_full.imag, label=r"$G_{\rm ana}(\rho, \rho)$")
    axes[1].axvline(lambd, color='k', linestyle=':', label=r'$\lambda$')
    axes[1].set_ylabel(r"$\mathfrak{Im}\{G(\rho, \rho)\}$")
    axes[1].legend()
    axes[1].grid(True)

    # 3. Residuals
    residuals = np.abs(res_num - ana_full)
    max_abs_err = np.max(residuals)
    max_rel_err = np.max(residuals / (np.abs(ana_full) + 1e-10))
    
    axes[2].plot(rho_full, residuals, label=r"$|G_{\rm num} - G_{\rm ana}|$", color='red')
    axes[2].axvline(lambd, color='k', linestyle=':', label=r'$\lambda$')

    axes[2].set_xlabel(r"$\rho$")
    axes[2].set_ylabel(r"$|G_{\rm num} - G_{\rm ana}|$")
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("results/analytic_vs_numerical.png")
        
    # Sample point comparison
    mid_idx = len(rho_full) // 2
    print(f"Sample Point (rho={rho_full[mid_idx]:.2f}):")
    print(f"  Numerical: {results_num[0, mid_idx].item()}")
    print(f"  Analytic:  {ana_full[mid_idx]}")
    
    print("Plot saved to results/analytic_vs_numerical.png. Scientist: Verify high-degree overlap in the top two panels and confirm residuals in the bottom panel are sufficiently small (allowing for smoothing peaks).")

    if max_abs_err < 1.0e-2:
        print(f"✅ Validation complete. Max Absolute Error: {max_abs_err:.2e}, Max Relative Error: {max_rel_err:.2e}")
    else:
        print(f"❌ Validation failed. Max Absolute Error: {max_abs_err:.2e}, Max Relative Error: {max_rel_err:.2e}")
        raise AssertionError

if __name__ == "__main__":
    compare_analytic_vs_numerical()
