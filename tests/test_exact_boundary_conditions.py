import torch
import numpy as np
import pytest
import sys
import os

# Add src/python to path
sys.path.append(os.path.join(os.getcwd(), 'src/python'))

from pytorch_solver import PyTorchSolver
from analytic import get_interior_solutions, get_analytic_wronskian
from profiles import StepFunctionProfile

def test_exact_boundary_conditions():
    """
    Diagnostic test: Force numerical solver to use exact analytic Whittaker BCs
    at both interior and exterior boundaries to isolate whether the discrepancy
    comes from native BC approximations or the ODE itself.
    """
    lambd = 1.0
    F = 2.0 * np.pi * 1.0
    m = 1.0
    chi = 1.5 + 0.5j
    ml = 1
    sigma3 = 1
    e = 1.0
    
    # We will solve the ODE on the interior domain [0.1, lambd]
    # to isolate interior dynamics from the complex exterior jump.
    rho_np = np.linspace(0.1, lambd, 500)
    # We use a lambd_eff slightly larger than the grid max to avoid the boundary jump
    # in the StepFunctionProfile's update logic at the last grid point.
    # CRITICAL: We MUST use the SAME lambda for the analytic reference.
    lambd_eff = lambd * 1.05
    profile = StepFunctionProfile(rho_np, lambd=lambd_eff, F=F, e=e)
    solver = PyTorchSolver(device="cpu")
    
    # 1. Analytic reference at boundaries [0.1, rho_np[-1]]
    # We use lambd_eff to match the profile.
    u0_ana, uinf_ana = get_interior_solutions(rho_np, chi, ml, sigma3, m, lambd_eff, F)
    
    # Finite difference derivatives for exact ICs
    h = 1e-5
    # Start: rho[0]
    u0_p, _ = get_interior_solutions(np.array([rho_np[0] + h]), chi, ml, sigma3, m, lambd_eff, F)
    u0_m, _ = get_interior_solutions(np.array([rho_np[0] - h]), chi, ml, sigma3, m, lambd_eff, F)
    du0_ana = (u0_p[0] - u0_m[0]) / (2 * h)
    
    # End: rho[-1]
    _, uinf_p = get_interior_solutions(np.array([rho_np[-1] + h]), chi, ml, sigma3, m, lambd_eff, F)
    _, uinf_m = get_interior_solutions(np.array([rho_np[-1] - h]), chi, ml, sigma3, m, lambd_eff, F)
    duinf_ana = (uinf_p[0] - uinf_m[0]) / (2 * h)

    # 2. Manual RK4 integration using Exact BCs
    params_pt = {
        'chi': torch.tensor([chi], dtype=torch.complex128),
        'ml': torch.tensor([ml], dtype=torch.int32),
        'sigma3': torch.tensor([sigma3], dtype=torch.int32),
        'm': torch.tensor([m], dtype=torch.float64),
        'e': torch.tensor([e], dtype=torch.float64),
    }
    
    # Forward: Integrate u0 from rho[0] with analytic ICs
    curr_state_u0 = torch.tensor([[u0_ana[0], du0_ana]], dtype=torch.complex128)
    rho_t = profile.rho
    a_phi_t = profile.a_phi
    da_phi_t = profile.da_phi
    
    u0_num = np.zeros(len(rho_np), dtype=np.complex128)
    u0_num[0] = curr_state_u0[0, 0].item()
    for i in range(len(rho_np)-1):
        h_step = rho_np[i+1] - rho_np[i]
        curr_state_u0 = solver.rk4_step(rho_t[i], h_step, curr_state_u0, params_pt, 
                                        a_phi_t[i], da_phi_t[i],
                                        0.5*(a_phi_t[i]+a_phi_t[i+1]), 0.5*(da_phi_t[i]+da_phi_t[i+1]),
                                        a_phi_t[i+1], da_phi_t[i+1])
        u0_num[i+1] = curr_state_u0[0, 0].item()

    # Backward: Integrate uinf from rho[-1] with analytic ICs
    curr_state_uinf = torch.tensor([[uinf_ana[-1], duinf_ana]], dtype=torch.complex128)
    print(f"IC uinf: {curr_state_uinf[0,0].item()}, duinf: {curr_state_uinf[0,1].item()}")
    uinf_num = np.zeros(len(rho_np), dtype=np.complex128)
    uinf_num[-1] = curr_state_uinf[0, 0].item()
    for i in range(len(rho_np)-1, 0, -1):
        h_step = rho_np[i-1] - rho_np[i]
        curr_state_uinf = solver.rk4_step(rho_t[i], h_step, curr_state_uinf, params_pt,
                                          a_phi_t[i], da_phi_t[i],
                                          0.5*(a_phi_t[i]+a_phi_t[i-1]), 0.5*(da_phi_t[i]+da_phi_t[i-1]),
                                          a_phi_t[i-1], da_phi_t[i-1])
        uinf_num[i-1] = curr_state_uinf[0, 0].item()
        if i == len(rho_np)-1:
            print(f"First backward step: rho={rho_np[i]}, h={h_step}, next_uinf={uinf_num[i-1]}, ana_uinf={uinf_ana[i-1]}")
            
    # Compare results
    # u0 should match analytic u0, uinf should match analytic uinf
    err_u0 = np.max(np.abs(u0_num - u0_ana))
    err_uinf = np.max(np.abs(uinf_num - uinf_ana))
    
    print(f"Uinf[0]: num={uinf_num[0]}, ana={uinf_ana[0]}")
    print(f"Uinf[mid]: num={uinf_num[len(rho_np)//2]}, ana={uinf_ana[len(rho_np)//2]}")
    print(f"Uinf[-1]: num={uinf_num[-1]}, ana={uinf_ana[-1]}")
    
    print(f"\nMax diff u0 (Exact BC): {err_u0:.2e}")
    print(f"Max diff uinf (Exact BC): {err_uinf:.2e}")
    
    assert err_u0 < 1e-4
    assert err_uinf < 1e-4

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
