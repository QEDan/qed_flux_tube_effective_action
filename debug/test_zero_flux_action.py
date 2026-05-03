import torch
import numpy as np
import sys
import os

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import ZeroFluxProfile

def test_zero_flux_action():
    lambd = 1.0
    B = 0.5
    m = 1.0
    
    rho = torch.linspace(0.01, 5.0, 200, dtype=torch.float64)
    profile = ZeroFluxProfile(rho, B=B, lambd=lambd)
    
    # Grid
    chi_values = np.linspace(1.1, 20.0, 20).tolist()
    ml_values = list(range(-20, 21))
    sigma3_values = [1, -1]
    
    orc = Orchestrator(device="cpu")
    
    print("Computing Numerical EA for ZeroFluxProfile...")
    action = orc.compute_effective_action(profile, chi_values, ml_values, sigma3_values, m=m)
    
    print(f"Action: {action.real.item()}")

if __name__ == "__main__":
    test_zero_flux_action()
