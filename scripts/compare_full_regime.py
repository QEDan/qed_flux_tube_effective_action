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
from analytic import get_interior_solutions, get_analytic_wronskian

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
    
    # Analytic
    # Interior
    u0_ana, uinf_ana = get_interior_solutions(rho_int, chi, ml, sigma3, m, lambd, F)
    W0_ana = get_analytic_wronskian(chi, ml, sigma3, m, lambd, F)
    ana_int = (u0_ana * uinf_ana) / W0_ana
    
    # Exterior
    k_ext = np.sqrt(chi**2 - m**2 + 0j)
    # G0 approx -pi/2 * rho * J * Y
    ana_ext = -0.5 * np.pi * rho_ext * jv(ml, k_ext * rho_ext) * yv(ml, k_ext * rho_ext)
    ana_full = np.concatenate([ana_int, ana_ext])
    
    # Normalization (Path A) - apply to full numerical array
    scaling = w0_num[0].detach().numpy() / W0_ana
    res_num_scaled = res_num / scaling
    
    # Visualization: 2 subplots (Overlay, Residual)
    print(f"Num Scaled[0]: {res_num_scaled[0]}")
    print(f"Num Scaled[100]: {res_num_scaled[100]}")
    print(f"Num Scaled[-1]: {res_num_scaled[-1]}")
    print(f"Ana Full[0]: {ana_full[0]}")
    print(f"Ana Full[100]: {ana_full[100]}")
    print(f"Ana Full[-1]: {ana_full[-1]}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Overlay
    axes[0].plot(rho_full, res_num_scaled.real, label="Numerical G (Scaled)", linestyle='--')
    axes[0].plot(rho_int, ana_int.real, label="Analytic Interior")
    axes[0].plot(rho_ext, ana_ext.real, label="Analytic Exterior")
    axes[0].axvline(lambd, color='k', linestyle=':', label='$\lambda$')
    axes[0].set_title("Full Domain Green's Function (Real)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Residuals
    residuals = np.abs(res_num_scaled - ana_full)
    axes[1].plot(rho_full, residuals, label="Absolute Residual")
    axes[1].axvline(lambd, color='k', linestyle=':')
    axes[1].set_title("Residuals (Numerical - Analytic)")
    axes[1].set_xlabel("Radial coordinate rho")
    axes[1].set_ylabel("Absolute Error")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("results/analytic_vs_numerical_full_residuals_v2.png")
    print("Plot saved as results/analytic_vs_numerical_full_residuals_v2.png")

if __name__ == "__main__":
    compare_full_regime()
