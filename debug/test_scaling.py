import torch
import numpy as np
import sys
import os

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import Sech2Profile

def test_scaling():
    lambd = 1.0
    m = 1.0
    B_range = [1e-3, 2e-3, 5e-3, 1e-2]
    
    rho = torch.linspace(0.01, 10.0, 500, dtype=torch.float64)
    
    chi_values = [2.0, 5.0]
    ml_values = list(range(-5, 6))
    sigma3_values = [1, -1]
    
    orc = Orchestrator(device="cpu")
    
    print("Testing scaling with B...")
    for B in B_range:
        profile = Sech2Profile(rho, B=B, lambd=lambd)
        action = orc.compute_effective_action(profile, chi_values, ml_values, sigma3_values, m=m)
        print(f"B={B}: Action={action.real.item()}")

if __name__ == "__main__":
    test_scaling()
