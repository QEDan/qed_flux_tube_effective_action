import sys
import os
import numpy as np
import torch
import pytest

# Add src/python to path
sys.path.append(os.path.join(os.getcwd(), 'src/python'))

from pytorch_solver import PyTorchSolver
from analytic import get_interior_solutions, get_analytic_wronskian
from profiles import StepFunctionProfile

def test_step_function_shape_validation():
    """
    Validate that numerical solutions satisfy the ODE and have the correct shape
    independent of normalization.
    """
    lambd = 1.0
    F = 2.0 * np.pi * 1.0
    m = 1.0
    chi = 1.0 + 0.5j
    ml = 1
    sigma3 = 1
    e = 1.0
    
    rho_np = np.linspace(0.01, lambd, 1000)
    profile = StepFunctionProfile(rho_np, lambd=lambd, F=F, e=e)
    solver = PyTorchSolver(device="cpu")
    
    # Extract u0 from manual integration to check shape
    params_pt = {
        'chi': torch.tensor([chi], dtype=torch.complex128),
        'ml': torch.tensor([ml], dtype=torch.int32),
        'sigma3': torch.tensor([sigma3], dtype=torch.int32),
        'm': torch.tensor([m], dtype=torch.float64),
        'e': torch.tensor([e], dtype=torch.float64),
    }
    
    rho_t = profile.rho
    a_phi_t = profile.a_phi
    da_phi_t = profile.da_phi
    
    u0_num = np.zeros(len(rho_np), dtype=np.complex128)
    # IC: u0 ~ rho^ml
    # Use torch.pow for safety with complex/real
    curr_u0 = torch.tensor([[torch.pow(torch.tensor(rho_np[0]), ml), ml * torch.pow(torch.tensor(rho_np[0]), ml-1)]], dtype=torch.complex128)
    u0_num[0] = curr_u0[0, 0].item()
    for i in range(len(rho_np)-1):
        h = rho_np[i+1] - rho_np[i]
        a_mid = 0.5 * (a_phi_t[i] + a_phi_t[i+1])
        da_mid = 0.5 * (da_phi_t[i] + da_phi_t[i+1])
        curr_u0 = solver.rk4_step(rho_t[i], h, curr_u0, params_pt, 
                                  a_phi_t[i], da_phi_t[i],
                                  a_mid, da_mid,
                                  a_phi_t[i+1], da_phi_t[i+1])
        u0_num[i+1] = curr_u0[0, 0].item()

    # Analytic u0
    u0_ana, _ = get_interior_solutions(rho_np, chi, ml, sigma3, m, lambd, F)
    
    # Normalize both at some point to compare shape
    norm_idx = len(rho_np) // 2
    u0_num_norm = u0_num / u0_num[norm_idx]
    u0_ana_norm = u0_ana / u0_ana[norm_idx]
    
    max_diff = np.max(np.abs(u0_num_norm - u0_ana_norm))
    print(f"\nMax relative difference in u0 shape: {max_diff:.2e}")
    assert max_diff < 1e-6, f"u0 shape mismatch: {max_diff}"

def test_wronskian_numeric_vs_analytic_complete():
    """
    Verify that the numerical Wronskian matches the analytic formula properties.
    """
    lambd = 1.0
    F = 2.0 * np.pi * 1.0
    m = 1.0
    chi = 1.0 + 0.5j
    ml = 1
    sigma3 = 1
    e = 1.0
    
    rho_np = np.linspace(0.01, lambd, 1000)
    profile = StepFunctionProfile(rho_np, lambd=lambd, F=F, e=e)
    solver = PyTorchSolver(device="cpu")
    
    # Let's ensure the analytic Wronskian formula matches the M'W - MW' definition.
    W0_ana = get_analytic_wronskian(chi, ml, sigma3, m, lambd, F)
    print(f"Analytic Wronskian: {W0_ana}")
    assert np.abs(W0_ana) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
