import torch
import numpy as np
import sys
import os

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import StepFunctionProfile

def exact_step_function_ea(B, lambd, m):
    """
    Theoretical result for Step Function profile EA (leading order).
    S = \pi \lambda^2 \mathcal{L}_{eff}(B)
    """
    e = 1.0
    L_eff = (e * B)**2 / (24.0 * np.pi**2) * np.log(m**2 / (e * B))
    return np.pi * (lambd**2) * L_eff

def test_step_action_small():
    lambd = 1.0
    B = 0.05
    F = 2.0 * np.pi * (lambd**2 * B / 2.0)
    m = 1.0
    
    # Small domain and high resolution
    rho = torch.linspace(0.01, 2.0, 1000, dtype=torch.float64)
    profile = StepFunctionProfile(rho, lambd=lambd, F=F)
    
    # Small chi range where solver is accurate
    chi_values = np.linspace(1.1, 5.0, 20).tolist()
    ml_values = list(range(-15, 16))
    sigma3_values = [1, -1]
    
    orc = Orchestrator(device="cpu")
    
    print(f"Computing Small-Scale Numerical EA (B={B}, rho_max=2.0)...")
    action = orc.compute_effective_action(profile, chi_values, ml_values, sigma3_values, m=m)
    
    num_val = action.real.item()
    
    # Analytic prediction for the whole domain
    # Since rho_max > lambda, we include the interior action
    ana_val = exact_step_function_ea(B, lambd, m)
    
    print(f"Numerical Action: {num_val}")
    print(f"Analytic Action:  {ana_val}")
    
    if abs(num_val / ana_val - 1.0) < 0.1:
        print("✅ Test passed: Numerical result matches analytic expectation.")
    else:
        print(f"❌ Test failed: Ratio is {num_val / ana_val if ana_val != 0 else 'inf'}")

if __name__ == "__main__":
    test_step_action_small()
