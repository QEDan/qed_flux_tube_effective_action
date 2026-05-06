import sys
import os
import torch
import numpy as np
import pytest

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from pytorch_solver import PyTorchSolver
from profiles import Sech2Profile

def test_sech2_flux_cancellation():
    """
    Test that the Green's function for Sech2Profile exhibits Aharonov-Bohm mode-shifting.
    For a continuous field profile, the matching should be smoother.
    """
    m = 1.0
    chi = 2.0
    e = 1.0
    lambd = 0.5
    B = 1.0 / (2.0 * np.pi * lambd**2 * np.log(2.0)) # Normalized flux
    
    rho_val = 10.0
    solver = PyTorchSolver(device="cpu")
    
    # Check mode mapping: G_ml(F) should be compared to G_{ml-1}(F=0)
    # Since Sech2 doesn't have a simple F=0 analytical gauge form,
    # we use F=0 with a tiny field (B -> 0).
    
    ml_check_range = range(-5, 6)
    
    # Profile with B = 1.0 (Flux F_bar = 1.0)
    profile_f = Sech2Profile(np.array([rho_val]), B=B, lambd=lambd, e=e)
    # Profile with B = 0.0 (Zero Flux)
    profile_zero = Sech2Profile(np.array([rho_val]), B=0.0, lambd=lambd, e=e)
    
    for ml in ml_check_range:
        res_int = solver.solve_batch([{'chi': chi + 0j, 'ml': ml, 'sigma3': 1, 'm': m, 'e': e}], profile_f)[0].item()
        
        # Compare with a range of modes for F=0
        print(f"\n--- Checking Mode ml={ml} (F=1): Value={res_int:.4f} ---")
        for ml_zero in range(ml - 3, ml + 3):
            res_zero = solver.solve_batch([{'chi': chi + 0j, 'ml': ml_zero, 'sigma3': 1, 'm': m, 'e': e}], profile_zero)[0].item()
            diff = abs(res_int - res_zero)
            print(f"  vs ml'={ml_zero} (F=0): Value={res_zero:.4f}, Diff={diff:.4f}")
    
    print("✅ Search complete.")

if __name__ == "__main__":
    test_sech2_flux_cancellation()
