import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.integrate import quad

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import ZeroFluxProfile
from analytic import heisenberg_euler_lagrangian

def validate_he_expansion():
    # Setup parameters
    lambd_values = [2.0, 5.0, 10.0]
    B_peak = 0.5
    m = 1.0
    
    # Grid for numerical solver
    chi_values = (np.linspace(1.1, 8.0, 5) + 0.1j).tolist()
    ml_values = list(range(-60, 61))
    sigma3_values = [1, -1]
    
    num_actions = []
    he_actions = []
    
    orc = Orchestrator(device="cpu", batch_size=512)
    
    print(f"--- Heisenberg-Euler Validation (B_peak={B_peak}) ---")
    
    for lambd in lambd_values:
        print(f"\nProcessing lambda={lambd}...")
        rho_max = 2.0 * lambd + 5.0
        rho = torch.linspace(0.01, rho_max, 1000, dtype=torch.float64)
        
        # We compute action at B and B/2 to eliminate the B^2 term
        # S(B) = c*B^2 + d*B^4
        # S(B/2) = c*(B^2/4) + d*(B^4/16)
        # 4*S(B/2) - S(B) = d*(B^4/4 - B^4) = -3/4 * d*B^4
        # So d*B^4 = (S(B) - 4*S(B/2)) / (1 - 1/4) = (S(B) - 4*S(B/2)) / 0.75? No.
        # S(B) - 4*S(B/2) = d*B^4 - 4*d*B^4/16 = d*B^4 - d*B^4/4 = 0.75 * d*B^4
        # d*B^4 = (S(B) - 4*S(B/2)) / 0.75
        
        profile_full = ZeroFluxProfile(rho, B=B_peak, lambd=lambd)
        profile_half = ZeroFluxProfile(rho, B=B_peak/2.0, lambd=lambd)
        
        action_full = orc.compute_effective_action(profile_full, chi_values, ml_values, sigma3_values, m=m)
        action_half = orc.compute_effective_action(profile_half, chi_values, ml_values, sigma3_values, m=m)
        
        num_val = (action_full.real.item() - 4.0 * action_half.real.item()) / 0.75
        num_actions.append(num_val)
        print(f"  Numerical B^4 Action: {num_val:.4e}")
        
        # 2. Heisenberg-Euler Benchmark (local limit)
        # S_HE = 2*pi * integral(rho * L_HE(B(rho)))
        def he_integrand(r):
            # B(rho) = B * (1 - 2*rho^2/lambd^2) for rho < lambd, else 0
            if r < lambd:
                B = B_peak * (1.0 - 2.0 * r**2 / lambd**2)
            else:
                B = 0.0
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
        raise AssertionError

if __name__ == "__main__":
    validate_he_expansion()
