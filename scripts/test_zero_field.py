import torch
import numpy as np
import sys
import os

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import StepFunctionProfile

def test_zero_field_cancellation():
    """
    Verify that the effective action renormalizes to zero when the magnetic field is zero.
    """
    print("Testing zero-field renormalization cancellation...")
    
    # Grid parameters
    rho = torch.linspace(0.01, 5.0, 1000, dtype=torch.float64)
    lambd = 1.0
    F = 0.0 # Zero magnetic flux
    
    # Profile with zero field
    profile = StepFunctionProfile(rho, lambd=lambd, F=F)
    
    # Backend
    orc = Orchestrator(device="cpu")
    
    # Integration parameters
    chi_values = [1.0, 2.0, 5.0]
    ml_values = [0, 1]
    sigma3_values = [1, -1]
    m = 1.0
    
    # Compute effective action
    action, _ = orc.compute_effective_action(profile, chi_values, ml_values, sigma3_values, m=m)
    
    print(f"Renormalized action for zero field: {action.item()}")
    
    # Assert action is zero
    try:
        assert torch.abs(action) < 1e-3, f"Action should be zero for zero field, got {action.item()}"
        print("✅ Zero-field cancellation validated: Renormalized action is zero as expected for zero magnetic flux.")
    except AssertionError as e:
        print(f"❌ Zero-field cancellation failed: {e}")
        raise

if __name__ == "__main__":
    test_zero_field_cancellation()
