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

def compare_analytic_vs_numerical():
    # Parameters
    lambd = 1.0
    F = 2.0 * np.pi * 1.0
    m = 1.0
    chi = 1.5 + 0.5j
    ml = 1
    sigma3 = 1
    
    # Use smoothed profile to allow numerical solver to handle boundary
    smooth = 0.05
    
    # Grid: Interior and Exterior
    rho_int = np.linspace(0.1, lambd, 100)
    rho_ext = np.linspace(lambd, 2.0, 100)
    rho_full = np.concatenate([rho_int, rho_ext])
    
    profile = StepFunctionProfile(rho_full, lambd=lambd, F=F, smooth_width=smooth)
    
    # Numerical solver
    orc = Orchestrator(backend_type="pytorch", device="cpu")
    params = [{'chi': chi, 'ml': ml, 'sigma3': sigma3, 'm': m, 'e': 1.0}]
    results_num, w0_num = orc.backend.solve_batch(params, profile)
    results_num = results_num[0].detach().numpy()
    
    # Analytic results
    # Interior: Whittaker
    u0_ana, uinf_ana = get_interior_solutions(rho_int, chi, ml, sigma3, m, lambd, F)
    W0_ana = get_analytic_wronskian(chi, ml, sigma3, m, lambd, F)
    ana_int = (u0_ana * uinf_ana) / W0_ana
    
    # Exterior: Bessel
    k_ext = np.sqrt(chi**2 - m**2 + 0j)
    ana_ext = -0.5 * np.pi * rho_ext * jv(ml, k_ext * rho_ext) * yv(ml, k_ext * rho_ext)
    
    # Normalization (Path A)
    scaling = w0_num[0].detach().numpy() / W0_ana
    results_num_scaled = results_num / scaling
    
    # Visualization: 2 subplots (Real, Imag)
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Real part
    axes[0].plot(rho_full, results_num_scaled.real, label="Numerical G (Scaled)", linestyle='--')
    axes[0].plot(rho_int, ana_int.real, label="Analytic Interior (Whittaker)")
    axes[0].plot(rho_ext, ana_ext.real, label="Analytic Exterior (Bessel)")
    axes[0].axvline(lambd, color='k', linestyle=':', label='$\lambda$')
    axes[0].set_title("Green's Function Comparison (Real Part, Smoothed)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Imaginary part
    axes[1].plot(rho_full, results_num_scaled.imag, label="Numerical G (Scaled)", linestyle='--')
    axes[1].plot(rho_int, ana_int.imag, label="Analytic Interior (Whittaker)")
    axes[1].plot(rho_ext, res_ext := results_num_scaled[100:].imag, label="Num Exterior (Scaled)", linestyle='--')
    axes[1].plot(rho_ext, ana_ext.imag, label="Analytic Bessel Exterior")
    axes[1].axvline(lambd, color='k', linestyle=':', label='$\lambda$')
    axes[1].set_title("Green's Function Comparison (Imaginary Part, Smoothed)")
    axes[1].set_xlabel("Radial coordinate rho")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("results/analytic_vs_numerical_smoothed.png")
    print("Plot saved as results/analytic_vs_numerical_smoothed.png")

if __name__ == "__main__":
    compare_analytic_vs_numerical()
