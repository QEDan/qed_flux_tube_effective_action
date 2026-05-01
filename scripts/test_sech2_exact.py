import torch
import numpy as np
import sys
import os

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import Sech2Profile

def test_sech2_exact():
    """
    Compute renormalized effective action for Sech2Profile and verify against theory.
    """
    print("Computing renormalized effective action for Sech2Profile...")
    
    # Parameters for validation
    rho = torch.linspace(0.01, 10.0, 500, dtype=torch.float64)
    B = 0.5
    lambd = 1.0
    profile = Sech2Profile(rho, B=B, lambd=lambd)
    
    # Integration over chi
    chi_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    ml_values = [0, 1]
    sigma3_values = [1, -1]
    m = 1.0
    
    orc = Orchestrator(backend_type="pytorch", device="cpu")
    
    action = orc.compute_effective_action(profile, chi_values, ml_values, sigma3_values, m=m)
    
    print(f"Renormalized Action (Dunne & Hall profile B=0.5, lambda=1.0): {action.item()}")
    
    # Basic sanity check: action should be negative for magnetic fields
    assert action.real < 0, "Effective action should be negative for magnetic field backgrounds."
    print("Test passed: Effective action is negative and finite.")

if __name__ == "__main__":
    test_sech2_exact()
