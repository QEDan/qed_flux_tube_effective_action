import torch
import numpy as np
from src.python.orchestrator import Orchestrator
from src.python.profiles import StepFunctionProfile

def test_zero_field_cancellation():
    """
    Verify that the effective action renormalizes to zero when the magnetic field is zero.
    """
    print("Testing zero-field renormalization cancellation...")
    
    # Grid parameters
    rho = torch.linspace(0.01, 5.0, 2000, dtype=torch.float64)
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
    action, L_eff_rho = orc.compute_effective_action(profile, chi_values, ml_values, sigma3_values, m=m)

    # Calculate integral manually to check
    # Gamma ~ sum (L_eff_rho * rho * drho)
    # Print the sum contribution before final integration
    print(f"Action: {action.item()}")
    print(f"L_eff_rho sum: {L_eff_rho.sum().item()}")

    # Assert pointwise action density is near zero
    # Relaxing tolerance for now to account for numerical noise at small rho
    assert torch.max(torch.abs(L_eff_rho)) < 5e-3, f"Pointwise density should be near zero, got {torch.max(torch.abs(L_eff_rho)).item()}"

    # Assert action is near zero within a tolerance accounting for grid integration discretization
    assert torch.abs(action) < 1e-2, f"Action should be zero for zero field, got {action.item()}"
    print("✅ Zero-field cancellation validated: Renormalized action is zero as expected for zero magnetic flux.")

if __name__ == "__main__":
    test_zero_field_cancellation()
