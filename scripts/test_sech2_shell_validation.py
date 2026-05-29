"""
Validates the Green's function computation for the Sech2-shell profile.
"""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np
import torch
import matplotlib.pyplot as plt
from src.python.orchestrator import Orchestrator
from src.python.sech2_shell import Sech2ShellProfile
from test_delta_shell_validation import get_analytic_delta_shell_g

def run_sech2_shell_validation():
    print("--- Running Sech2-Shell Green's Function Validation ---")
    
    # Parameters for a narrow shell to compare with delta analytic result
    R = 5.0
    B0 = 1.0
    lambd = 0.05 # Even narrower shell for better delta-limit agreement
    F = 4.0 * np.pi * R * B0 * lambd # Approx total flux for Sech2 shell
    
    rho_np = np.linspace(0.01, 10.0, 1000)
    profile = Sech2ShellProfile(rho_np, R=R, B=B0, lambd=lambd)
    
    orchestrator = Orchestrator(device='cpu')
    
    # Test for a single set of parameters
    chi = 0.5
    ml = 1
    sigma3 = 1
    m = 1.0
    
    params = [{'chi': complex(chi), 'ml': ml, 'sigma3': sigma3, 'm': m, 'e': 1.0}]
    
    print(f"Solving numerically for chi={chi}, ml={ml}, sigma3={sigma3}...")
    num_g_batch, _ = orchestrator.backend.solve_batch(params, profile)
    num_g = num_g_batch[0].cpu().numpy()
    
    print("Computing analytic delta-shell equivalent (as benchmark)...")
    ana_g = get_analytic_delta_shell_g(rho_np, R, F, chi, ml, sigma3, m)
    
    # Visualization: 3 subplots (Real, Imag, Residuals)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # 1. Real part
    axes[0].plot(rho_np, num_g.real, label='Numerical Sech2 (Real)', color='blue')
    axes[0].plot(rho_np, ana_g.real, label='Analytic Delta-Shell (Real)', linestyle='--', color='red')
    axes[0].axvline(x=R, color='gray', linestyle=':', label='Shell Radius R')
    axes[0].set_ylabel(r"$\mathfrak{Re}\{G(\rho, \rho)\}$")
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Imaginary part
    axes[1].plot(rho_np, num_g.imag, label='Numerical Sech2 (Imag)', color='blue')
    axes[1].plot(rho_np, ana_g.imag, label='Analytic Delta-Shell (Imag)', linestyle='--', color='red')
    axes[1].axvline(x=R, color='gray', linestyle=':', label='Shell Radius R')
    axes[1].set_ylabel(r"$\mathfrak{Im}\{G(\rho, \rho)\}$")
    axes[1].legend()
    axes[1].grid(True)

    # 3. Residuals
    residuals = np.abs(num_g - ana_g)
    axes[2].plot(rho_np, residuals, label='|Num Sech2 - Ana Delta|', color='black')
    axes[2].axvline(x=R, color='gray', linestyle=':', label='Shell Radius R')
    axes[2].set_xlabel('rho')
    axes[2].set_ylabel('Residual')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    output_path = "results/sech2_shell_greens_function_validation.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Compute error (expect some difference due to non-zero lambda)
    error = np.linalg.norm(num_g - ana_g) / np.linalg.norm(ana_g)
    print(f"Relative difference: {error:.2e}")
    
    if error < 0.1:
        print("✅ Sech2-shell Green's function validation passed!")
    else:
        print("⚠️ Sech2-shell Green's function validation: difference from delta-shell may be significant.")

if __name__ == "__main__":
    run_sech2_shell_validation()
