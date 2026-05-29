import torch
import numpy as np
import pytest

from src.python.pytorch_solver import PyTorchSolver
from src.python.sech2_shell import Sech2ShellProfile

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
    R = 2.0 * lambd
    profile_f = Sech2ShellProfile(np.array([rho_val]), R=R, B=B, lambd=lambd, e=e)
    profile_zero = Sech2ShellProfile(np.array([rho_val]), R=R, B=0.0, lambd=lambd, e=e)
    
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
