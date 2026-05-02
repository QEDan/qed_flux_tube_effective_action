import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.special import bernoulli

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import Sech2Profile

def exact_dunne_hall_expansion(B_peak, m, lambd):
    """
    Computes the renormalized QED effective action using the derivative expansion:
    S = \int d^3x \mathcal{L}_{eff}(B(\rho))
    \mathcal{L}_{eff} = \frac{e^2 B^2}{24 \pi^2} \ln(m^2 / eB) (leading term for small B)
    """
    e = 1.0
    # Integration over rho: 2*pi \int rho drho \mathcal{L}(B(\rho))
    # B(\rho) = B_peak * sech^2(rho/lambd)
    rho_grid = np.linspace(0.0, 20.0, 1000)
    B_profile = B_peak / (np.cosh(rho_grid / lambd)**2)
    
    # Avoid log(0)
    B_safe = np.where(B_profile < 1e-10, 1e-10, B_profile)
    
    L_eff = (e * B_safe)**2 / (24.0 * np.pi**2) * np.log(m**2 / (e * B_safe))
    
    action_per_length = 2.0 * np.pi * np.trapz(rho_grid * L_eff, rho_grid)
    return action_per_length

def plot_exact_comparison():
    # Setup
    rho = torch.linspace(0.01, 15.0, 1000, dtype=torch.float64)
    lambd = 1.0
    m = 1.0
    B_range = np.linspace(0.05, 0.3, 6)
    
    # Use a better grid for chi and ml
    chi_values = np.linspace(1.0, 10.0, 25).tolist()
    ml_values = list(range(-15, 16))
    sigma3_values = [1, -1]
    
    orc = Orchestrator(device="cpu")
    
    num_actions = []
    ana_actions = []
    
    print("Comparing Numerical vs Dunne-Hall Exact Theory...")
    for B in B_range:
        profile = Sech2Profile(rho, B=B, lambd=lambd)
        action = orc.compute_effective_action(profile, chi_values, ml_values, sigma3_values, m=m)
        num_actions.append(action.real.item())
        
        # Analytic comparison
        ana = exact_dunne_hall_expansion(B, m, lambd)
        ana_actions.append(ana)
    
    # Plotting
    print(f"Numerical Actions: {num_actions}")
    print(f"Analytic Actions:  {ana_actions}")
    plt.figure(figsize=(10, 6))
    plt.plot(B_range, num_actions, marker='o', label=r"Numerical $S_{renorm}$")
    plt.plot(B_range, ana_actions, linestyle='--', label=r"Theoretical Derivative Expansion")
    plt.title(r"Comparison with Dunne & Hall Exact QED Effective Action")
    plt.xlabel(r"Field Strength $B$")
    plt.ylabel(r"Effective Action $S$")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/exact_theory_comparison.png")
    print("Plot saved to results/exact_theory_comparison.png")

if __name__ == "__main__":
    plot_exact_comparison()
