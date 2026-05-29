import torch
import numpy as np
import matplotlib.pyplot as plt
from src.python.orchestrator import Orchestrator
from src.python.profiles import WLNFluxTubeProfile
from src.python.locally_constant_field import const_field_heisenberg_euler_lagrangian as heisenberg_euler_lagrangian

def reproduce_fig_7_3():
    """
    Reproduces Figure 7.3 from WLNumerics.tex:
    Effective Action Density vs rho for a flux tube with lambda = 10 * lambda_e.
    Note: lambda_e = 1/m = 1.0 in our units.
    """
    device = "cpu"
    orchestrator = Orchestrator(device=device, strategy="numerical")
    
    # Parameters from Fig 7.3
    lambd = 13.4
    F_cal = 10.0 
    F_val = F_cal * 2.0 * np.pi
    # Thesis says F_cal = eF/2pi. So F = F_cal * 2pi / e.
    # In Fig 7.3, it doesn't specify F, but Fig 7.2 uses F_cal = 10.
    
    # Spectral parameters
    chi_vals = [complex(c) for c in np.logspace(-1, 1.3, 10)]
    # Shifted mode range to capture the physics of the flux tube
    ml_vals = list(range(-1000, 1001)) 
    sigma3_vals = [1, -1]
    
    rho = torch.linspace(0.01, 30.0, 10)
    profile = WLNFluxTubeProfile(rho, lambd=lambd, F=F_val)

    print("Computing Effective Action Density...")
    action, L_eff_rho = orchestrator.compute_effective_action(
        profile, chi_vals, ml_vals, sigma3_vals, collect_density=True, lcf_threshold=2.0
    )

    # The effective action density is defined as L_eff, 
    # where Action = integral 2*pi*rho * L_eff * d_rho.
    # The thesis defines the density as L_eff.
    # Current code computes action = sum(L_eff * rho * drho).
    # Since area = pi * rho^2, d(Area) = 2*pi*rho * drho.
    # So Action = integral (2*pi*rho) * L_eff * drho / (2*pi) ??
    # Let's normalize by 2*pi*rho to match the thesis density definition.

    L_eff_np = -1.0 * L_eff_rho.real.detach().cpu().numpy()
    rho_np = rho.detach().cpu().numpy()

    print(f"L_eff_np[0]: {L_eff_np[0]:.6e}")
    print(f"L_eff_np[-1]: {L_eff_np[-1]:.6e}")

    # Compute LCF approximation (Heisenberg-Euler)
    B_theory = F_val * lambd**2 / (np.pi * (lambd**2 + rho_np**2)**2)
    L_lcf = np.array([heisenberg_euler_lagrangian(B) for B in B_theory])
    
    print(f"B_theory max: {np.max(B_theory):.4e}")
    print(f"L_lcf max: {np.max(L_lcf):.4e}")

    # 2. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(rho_np, L_eff_np, 'o', label='Numerical (Spectral)')
    plt.plot(rho_np, -L_lcf, '-', label='LCF (Heisenberg-Euler)')
    # Use standard linear scale as the range is small
    plt.xlabel(r'$\rho$ (Compton Wavelengths)')
    plt.ylabel(r'Effective Action Density')
    plt.title(f'Effective Action Density ($\lambda={lambd} \lambda_e$)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/wln_reproduction_fig7_3.png')
    print("Saved plot to results/wln_reproduction_fig7_3.png")

def reproduce_fig_7_2():
    """
    Reproduces Figure 7.2 from WLNumerics.tex:
    Proper time integrand vs T for a flux tube with lambda = 1.
    """
    # ... Similar logic but plotting integrand vs T ...
    # Our solver works in Q. We can plot vs Q and explain.
    pass

if __name__ == "__main__":
    reproduce_fig_7_3()
