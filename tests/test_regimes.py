import sys
import os
import torch
import numpy as np
import pytest

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from pytorch_solver import PyTorchSolver
from profiles import StepFunctionProfile

def test_regime_smoothness():
    """
    Test that the Green's function is smooth and well-behaved in both 
    oscillatory (chi^2 > m^2) and decaying (chi^2 < m^2) regimes.
    """
    lambd = 1.0
    F = 2.0 * np.pi * 1.0
    m = 1.0
    ml = 1
    sigma3 = 1
    e = 1.0
    
    rho = np.linspace(0.1, 5.0, 1000)
    profile = StepFunctionProfile(rho, lambd=lambd, F=F, e=e)
    solver = PyTorchSolver(device="cpu")
    
    # 1. Oscillatory Regime (chi = 2.0 + 0j, chi^2 = 4 > 1)
    params_osc = [{'chi': 2.0 + 0j, 'ml': ml, 'sigma3': sigma3, 'm': m, 'e': e}]
    results_osc, _ = solver.solve_batch(params_osc, profile)
    res_osc = results_osc[0].detach().numpy()
    
    # Check for NaNs and smoothness (no large jumps)
    assert not np.any(np.isnan(res_osc))
    diff_osc = np.diff(res_osc)
    assert np.max(np.abs(diff_osc)) < 0.1 # Heuristic for 1000 points
    
    # 2. Decaying Regime (chi = 0.5 + 0j, chi^2 = 0.25 < 1)
    params_dec = [{'chi': 0.5 + 0j, 'ml': ml, 'sigma3': sigma3, 'm': m, 'e': e}]
    results_dec, _ = solver.solve_batch(params_dec, profile)
    res_dec = results_dec[0].detach().numpy()
    
    print(f"Decaying regime: res_dec[0]={res_dec[0]}, res_dec[mid]={res_dec[len(rho)//2]}, res_dec[-1]={res_dec[-1]}")
    
    assert not np.any(np.isnan(res_dec))
    diff_dec = np.diff(res_dec)
    assert np.max(np.abs(diff_dec)) < 0.1
    
    # Check that decaying regime is stable (converges to a constant -1/2kappa)
    # kappa = sqrt(m^2 - chi^2) = sqrt(1 - 0.25) = 0.866
    # -1/2kappa = -0.577
    theoretical_const = -1.0 / (2.0 * np.sqrt(m**2 - params_dec[0]['chi'].real**2))
    assert np.abs(res_dec[-1] - theoretical_const) < 0.1
    
    # Ensure it's not growing exponentially
    assert np.abs(res_dec[-1]) < np.abs(res_dec[0]) * 100.0 
    
    print("✅ Regime smoothness tests passed.")

if __name__ == "__main__":
    test_regime_smoothness()
