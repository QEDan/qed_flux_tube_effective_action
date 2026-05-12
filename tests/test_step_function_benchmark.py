import sys
import os
import torch
import numpy as np
import pytest

# Add src/python to path
sys.path.append(os.path.join(os.getcwd(), 'src/python'))

from pytorch_solver import PyTorchSolver
from analytic import get_interior_solutions, get_analytic_wronskian
from profiles import StepFunctionProfile

def test_step_function_exact_normalization():
    """
    Validates that numerical solutions match analytic Whittaker solutions 
    in absolute value, not just shape. This requires exact initial conditions.
    """
    lambd = 1.0
    F = 2.0 * np.pi * 1.0
    m = 1.0
    chi = 1.5 + 0.5j
    ml = 1
    sigma3 = 1
    e = 1.0
    
    # We use a very dense grid to minimize integration error
    rho_np = np.linspace(0.1, lambd, 1000)
    profile = StepFunctionProfile(rho_np, lambd=lambd, F=F, e=e)
    solver = PyTorchSolver(device="cpu")
    
    # 1. Get exact analytic values at start of integration grid (rho_np[0])
    u0_ana, _ = get_interior_solutions(rho_np, chi, ml, sigma3, m, lambd, F)
    # We need derivative at rho[0]. Using finite difference.
    h = 1e-5
    u0_p, _ = get_interior_solutions(np.array([rho_np[0] + h]), chi, ml, sigma3, m, lambd, F)
    u0_m, _ = get_interior_solutions(np.array([rho_np[0] - h]), chi, ml, sigma3, m, lambd, F)
    du0_ana = (u0_p[0] - u0_m[0]) / (2 * h)
    
    # 2. Numerical integration (forward from rho[0])
    params_pt = {
        'chi': torch.tensor([chi], dtype=torch.complex128),
        'ml': torch.tensor([ml], dtype=torch.int32),
        'sigma3': torch.tensor([sigma3], dtype=torch.int32),
        'm': torch.tensor([m], dtype=torch.float64),
        'e': torch.tensor([e], dtype=torch.float64),
    }
    
    # Force exact IC
    curr_state = torch.tensor([[u0_ana[0], du0_ana]], dtype=torch.complex128)
    
    rho_t = profile.rho
    a_phi_t = profile.a_phi
    da_phi_t = profile.da_phi
    
    u0_num = np.zeros(len(rho_np), dtype=np.complex128)
    u0_num[0] = curr_state[0, 0].item()
    
    for i in range(len(rho_np)-1):
        h_step = rho_np[i+1] - rho_np[i]
        a_mid = 0.5 * (a_phi_t[i] + a_phi_t[i+1])
        da_mid = 0.5 * (da_phi_t[i] + da_phi_t[i+1])
        curr_state = solver.rk4_step(rho_t[i], h_step, curr_state, params_pt, 
                                     a_phi_t[i], da_phi_t[i],
                                     a_mid, da_mid,
                                     a_phi_t[i+1], da_phi_t[i+1])
        u0_num[i+1] = curr_state[0, 0].item()

    # Compare absolute values
    max_diff = np.max(np.abs(u0_num - u0_ana))
    print(f"\nMax absolute difference in u0: {max_diff:.2e}")
    
    assert max_diff < 1e-6

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
