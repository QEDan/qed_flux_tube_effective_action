
import numpy as np
import torch
import pytest
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src/python'))
from pytorch_solver import PyTorchSolver
from profiles import StepFunctionProfile

def test_wkb_approximation_limit():
    """
    Validates that numerical solutions u0, uinf converge to WKB forms for large chi.
    Specifically checks that the phase derivative S' matches sqrt(Q).
    """
    # Parameters
    lambd = 1.0
    F = 2.0 * np.pi
    e = 1.0
    m = 1.0
    ml = 0
    sigma3 = 1
    
    # Large chi for WKB limit
    chi_values = [200.0, 400.0]
    errors = []
    
    for chi in chi_values:
        # Grid: ensure enough points for large chi
        n_points = int(25 * chi) 
        rho = np.linspace(0.01, 1.0 * lambd, n_points)
        
        profile = StepFunctionProfile(rho, lambd=lambd, F=F, e=e)
        solver = PyTorchSolver(device='cpu')
        
        params = [{'chi': chi, 'ml': ml, 'sigma3': sigma3, 'm': m, 'e': e}]
        
        # Solve numerically
        g_num_t, _ = solver.solve_batch(params, profile)
        g_num = g_num_t[0].detach().cpu().numpy().real
        
        # Calculate Q(rho)
        B = F / (np.pi * lambd**2)
        k2_eff = chi**2 - m**2 - e * sigma3 * B + e * ml * B
        Q = k2_eff + (0.25 - ml**2) / (rho**2) - 0.25 * (e * B * rho)**2
        
        # Numerical phase of G using Hilbert transform
        from scipy.signal import hilbert
        g_a = hilbert(g_num)
        phase_num = np.unwrap(np.angle(g_a))
        
        # G ~ sin(2S), so phase_num derivative is 2*S' = 2*sqrt(Q)
        # Avoid edge effects
        valid = slice(n_points//10, -n_points//10)
        s_prime_num = 0.5 * np.abs(np.gradient(phase_num, rho))
        s_prime_wkb = np.sqrt(Q)
        
        # Use mean of S' to avoid jitter from Hilbert transform
        mean_s_prime_num = np.mean(s_prime_num[valid])
        mean_s_prime_wkb = np.mean(s_prime_wkb[valid])
        
        error = np.abs(mean_s_prime_num - mean_s_prime_wkb) / mean_s_prime_wkb
        print(f"Mean S' relative error for chi={chi}: {error:.6e}")
        errors.append(error)
    
    # Check that error is small
    assert errors[0] < 0.05
    assert errors[1] < 0.05

if __name__ == "__main__":
    test_wkb_approximation_limit()
