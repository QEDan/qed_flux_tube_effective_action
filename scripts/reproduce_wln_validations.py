import sys
import os
sys.path.append(os.path.abspath("."))
from src.python import constants
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.python.orchestrator import Orchestrator
from src.python.profiles import WLNFluxTubeProfile
from src.python.analytic import heisenberg_euler_lagrangian

def reproduce_fig_7_3():
    """
    Reproduces Figure 7.3 from WLNumerics.tex:
    Effective Action Density vs rho for a flux tube with lambda = 10 * lambda_e.
    """
    device = "cpu"
    # Use numerical strategy for maximum precision matching the thesis worldline approach
    orchestrator = Orchestrator(device=device, strategy="analytic", batch_size=2048)
    
    # Parameters from Fig 7.3 and context
    lambd = 10.0
    F_cal = 1.0  # F_cal = e*Phi / 2pi
    e_val = constants.ELECTRON_CHARGE
    F_val = F_cal * (constants.TWO_PI / e_val)
    
    # Spectral parameters
    # Increase chi resolution for smoother integration
    chi_vals = [complex(c) for c in np.linspace(0.1, 5.0, 10)]
    # Symmetrical mode range around the flux F_cal
    ml_vals = list(range(-150, 151)) 
    sigma3_vals = [1, -1]
    
    # Grid in rho
    rho = torch.linspace(0.01, 20.0, 10)
    profile = WLNFluxTubeProfile(rho, lambd=lambd, F=F_val, e=e_val)

    print(f"Computing Effective Action Density for lambda={lambd}, F_cal={F_cal}...")
    # lcf_threshold set to 5.0 to ensure UV tail is captured by HE analytic form
    action, L_eff_rho = orchestrator.compute_effective_action(
        profile, chi_vals, ml_vals, sigma3_vals, collect_density=True, lcf_threshold=5.0
    )

    # Orchestrator returns the density as L_eff.
    L_eff_np = L_eff_rho.real.detach().cpu().numpy()
    rho_np = rho.detach().cpu().numpy()

    print(f"L_eff_np[0]: {L_eff_np[0]:.6e}")
    print(f"L_eff_np[-1]: {L_eff_np[-1]:.6e}")

    # Compute LCF approximation (Heisenberg-Euler)
    B_theory = F_val * lambd**2 / (np.pi * (lambd**2 + rho_np**2)**2)
    L_lcf = np.array([-heisenberg_euler_lagrangian(B) for B in B_theory])

    print(f"B_theory max: {np.max(B_theory):.4e}")
    print(f"L_lcf max: {np.max(L_lcf):.4e}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(rho_np, L_eff_np, 'o', label='Numerical (Spectral Sum)', markersize=5)
    plt.plot(rho_np, L_lcf, '-', label='LCF (Heisenberg-Euler)', alpha=0.7)

    plt.xlabel(r'$\rho / \lambda_e$')
    plt.ylabel(r'$\mathcal{L}_{\rm eff}(\rho)$')
    plt.title(fr'Effective Action Density Reproduction ($\lambda={lambd}, \mathcal{{F}}={F_cal}$)')

    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results/wln_reproduction_fig7_3.png')
    print("Saved plot to results/wln_reproduction_fig7_3.png")

if __name__ == "__main__":
    reproduce_fig_7_3()
