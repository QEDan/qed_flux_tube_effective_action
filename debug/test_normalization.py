import torch
import numpy as np
import sys
import os

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import StepFunctionProfile

def schwinger_ea_density(B, m=1.0, e=1.0):
    """
    Leading order (logarithmic) Schwinger effective Lagrangian density.
    L = - (eB)^2 / (24 pi^2) * ln(Lambda^2 / m^2) ?? 
    Actually, the finite part is what we usually compute.
    Let's use the standard form:
    L = - (eB)^2 / (24 pi^2) * ln(m^2 / (eB))  (approx for eB << m^2)
    Wait, the full formula is an integral.
    For eB = 0.1, m = 1, the formula is:
    """
    # Using the same approximation as test_step_action.py
    return -(e * B)**2 / (24.0 * np.pi**2) * np.log(m**2 / (e * B))

def test_normalization():
    # Use a medium lambda
    lambd = 5.0
    B = 0.5
    F = np.pi * (lambd**2) * B
    m = 1.0
    
    # Integrate rho from 0.01 to 2.0 (inside lambda)
    # This is a region of uniform field B=0.5
    rho_max = 2.0
    rho = torch.linspace(0.01, rho_max, 100, dtype=torch.float64)
    profile = StepFunctionProfile(rho, lambd=lambd, F=F)
    
    # Use a large chi range for convergence
    # Start at chi=2.0 to be safely away from m=1.0
    chi_values = np.linspace(2.0, 50.0, 100).tolist()
    ml_values = list(range(-15, 16))
    sigma3_values = [1, -1]
    
    orc = Orchestrator(device="cpu")
    
    print(f"Computing EA for quasi-uniform field B={B}...")
    action = orc.compute_effective_action(profile, chi_values, ml_values, sigma3_values, m=m)
    
    num_val = action.real.item()
    
    # Theoretical prediction: Area * L_eff
    # Area = pi * rho_max^2
    area = np.pi * (rho_max**2)
    expected_val = area * schwinger_ea_density(B, m)
    
    print(f"Numerical Result (S / T Lz): {num_val}")
    print(f"Expected Result (Area * L_eff): {expected_val}")
    if expected_val != 0:
        print(f"Ratio: {num_val / expected_val}")
    
    # Check if we are closer to the 1/8pi^2 vs pi factor
    print(f"Factor -1/(8*pi^2) is { -1.0 / (8.0 * np.pi**2) }")
    print(f"Factor pi is { np.pi }")

if __name__ == "__main__":
    test_normalization()
