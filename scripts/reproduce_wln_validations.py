import torch
import numpy as np
import matplotlib.pyplot as plt
from src.python import constants
from src.python.orchestrator import Orchestrator
from src.python.profiles import WLNFluxTubeProfile
from src.python.locally_constant_field import const_field_heisenberg_euler_lagrangian as heisenberg_euler_lagrangian

def reproduce_fig_7_3():
    """
    Reproduces Figure 7.3 from WLNumerics.tex:
    Effective Action Density vs rho for a flux tube with lambda = 10 * lambda_e.
    Note: lambda_e = 1/m.
    """
    device = "cpu"
    m_val = constants.ELECTRON_MASS
    e_val = constants.ELECTRON_CHARGE
    orchestrator = Orchestrator(device=device, strategy="numerical", global_mode=False)
    
    # Parameters from Fig 7.3: lambda = 13.4 * lambda_e = 13.4 / m
    m_val = constants.ELECTRON_MASS
    e_val = constants.ELECTRON_CHARGE
    lambd = 13.4 / m_val
    F_cal = 10.0 
    F_val = F_cal * (2.0 * np.pi / e_val)
    
    # Spectral parameters: Low-Q for solver, High-Q for analytic tail
    chi_low = np.logspace(-1, 1.3, 20) # Up to ~20
    chi_high = np.logspace(1.3, 3.0, 50) # Up to 1000 for tail
    chi_vals = [complex(c) for c in np.concatenate([chi_low, chi_high])]
    
    ml_vals = list(range(-300, 301)) 
    sigma3_vals = [1, -1]
    
    rho = torch.linspace(0.01, 30.0 / m_val, 20)
    profile = WLNFluxTubeProfile(rho, lambd=lambd, F=F_val, e=e_val)

    print("Computing Effective Action Density...")
    action, L_eff_rho = orchestrator.compute_effective_action(
        profile, chi_vals, ml_vals, sigma3_vals, m=m_val, e=e_val, collect_density=True, lcf_threshold=20.0
    )

    L_eff_np = -1.0 * L_eff_rho.real.detach().cpu().numpy()
    rho_np = rho.detach().cpu().numpy()

    # Compute LCF approximation (Heisenberg-Euler)
    # Get physical B field from profile
    _, _, _ = profile.get_arrays(as_numpy=True)
    B_vals = profile.B_vals.detach().cpu().numpy()
    L_lcf = np.array([heisenberg_euler_lagrangian(B, m=m_val, e=e_val) for B in B_vals])
    
    print(f"L_eff_np[0]: {L_eff_np[0]:.6e}")
    print(f"L_lcf[0]:    {L_lcf[0]:.6e}")
    print(f"Ratio Num/LCF: {L_eff_np[0]/L_lcf[0]:.4f}")

    # 2. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(rho_np, L_eff_np, 'o', label='Numerical (Spectral)')
    plt.plot(rho_np, L_lcf, '-', label='LCF (Heisenberg-Euler)')
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
