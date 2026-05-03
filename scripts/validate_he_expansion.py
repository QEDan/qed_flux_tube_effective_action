import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.integrate import quad

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import Sech2Profile
from analytic import heisenberg_euler_lagrangian

def validate_he_expansion():
    # Setup parameters
    lambd_values = [2.0, 5.0, 10.0, 15.0] # Width of profile
    B_peak = 0.5
    m = 1.0
    
    # Grid for numerical solver
    chi_values = np.linspace(1.1, 10.0, 10).tolist()
    ml_values = list(range(-200, 201))
    sigma3_values = [1, -1]
    
    num_actions = []
    he_actions = []
    
    orc = Orchestrator(device="cpu", batch_size=512)
    
    print(f"--- Heisenberg-Euler Validation (B_peak={B_peak}) ---")
    
    for lambd in lambd_values:
        print(f"\nProcessing lambda={lambd}...")
        rho_max = 2.0 * lambd + 5.0
        rho = torch.linspace(0.01, rho_max, 1000, dtype=torch.float64)
        profile = Sech2Profile(rho, B=B_peak, lambd=lambd)
        
        # 1. Numerical Solver
        action = orc.compute_effective_action(profile, chi_values, ml_values, sigma3_values, m=m)
        num_val = action.real.item()
        num_actions.append(num_val)
        print(f"  Numerical Action: {num_val:.4e}")
        
        # 2. Heisenberg-Euler Benchmark (local limit)
        # S_HE = 2*pi * integral(rho * L_HE(B(rho)))
        def he_integrand(r):
            B = B_peak / (np.cosh(r / lambd)**2)
            return r * heisenberg_euler_lagrangian(B, m=m)
        
        res_he, _ = quad(he_integrand, 0, rho_max)
        he_val = 2.0 * np.pi * res_he
        he_actions.append(he_val)
        print(f"  HE Benchmark Action: {he_val:.4e}")

    num_actions = np.array(num_actions)
    he_actions = np.array(he_actions)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(lambd_values, np.abs(num_actions), 'o-', label="Numerical Solver")
    plt.plot(lambd_values, np.abs(he_actions), 's--', label="Local Heisenberg-Euler (Benchmark)")
    
    plt.yscale('log')
    plt.xlabel(r"Profile Width $\lambda$")
    plt.ylabel(r"Integrated Effective Action $|S|$")
    plt.title(f"Heisenberg-Euler Convergence (B_peak={B_peak})")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.savefig("results/he_validation.png")
    print("\n✅ Plot saved to results/he_validation.png")
    
    # Final check: Does the ratio approach 1 for large lambda?
    final_ratio = num_actions[-1] / he_actions[-1]
    print(f"Final Ratio (Numerical/HE): {final_ratio:.4f}")
    
    if 0.5 < abs(final_ratio) < 2.0:
        print(f"✅ Validation successful: Numerical result matches HE magnitude.")
    else:
        print(f"❌ Validation failed: Large magnitude discrepancy persists.")

if __name__ == "__main__":
    validate_he_expansion()
