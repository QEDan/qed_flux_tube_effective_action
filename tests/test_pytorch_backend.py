import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src/python'))

import numpy as np
import torch
from orchestrator import Orchestrator, generate_params_grid
from profiles import StepFunctionProfile

def test_pytorch_vs_c():
    """
    Compare PyTorch backend results with C backend results.
    """
    rho = np.linspace(0.01, 5.0, 200)
    profile = StepFunctionProfile(rho, lambd=1.0, F=2*np.pi)
    
    # Small grid for comparison
    chi_values = [1.0 + 0.1j, 2.0]
    ml_values = [0, 1]
    sigma3_values = [1, -1]
    params_grid = generate_params_grid(chi_values, ml_values, sigma3_values)
    
    # Compute with C backend
    orc_c = Orchestrator(backend_type="c", lib_path="./libsolver.so")
    results_c = orc_c.backend.solve_batch(params_grid, profile)
    
    # Compute with PyTorch backend
    orc_pt = Orchestrator(backend_type="pytorch", device="cpu")
    results_pt, _ = orc_pt.backend.solve_batch(params_grid, profile)
    
    # Compare
    res_c = results_c[0]
    res_pt = results_pt[0].detach().numpy()
    print(f"C backend [0]: {res_c[0]}")
    print(f"PT backend [0]: {res_pt[0]}")
    print(f"C backend [mid]: {res_c[len(rho)//2]}")
    print(f"PT backend [mid]: {res_pt[len(rho)//2]}")
    print(f"C backend [-1]: {res_c[-1]}")
    print(f"PT backend [-1]: {res_pt[-1]}")
    
    max_diff = np.max(np.abs(results_c - results_pt.detach().numpy()))
    assert max_diff < 1e-10, f"Difference too large: {max_diff}"

def test_wronskian_consistency():
    """
    Check if the Wronskian computed in PyTorch is constant.
    This is a scientific validation of the ODE solver.
    """
    from pytorch_solver import PyTorchSolver
    
    rho = np.linspace(0.01, 5.0, 500)
    profile = StepFunctionProfile(rho, lambd=1.0, F=2*np.pi)
    
    solver = PyTorchSolver(device="cpu")
    
    params_list = [{'chi': 1.0+0.5j, 'ml': 1, 'sigma3': 1, 'm': 1.0, 'e': 1.0}]
    
    # Batch parameters
    params = {
        'chi': torch.tensor([p['chi'] for p in params_list], dtype=torch.complex128),
        'ml': torch.tensor([p['ml'] for p in params_list], dtype=torch.int32),
        'sigma3': torch.tensor([p['sigma3'] for p in params_list], dtype=torch.int32),
        'm': torch.tensor([p['m'] for p in params_list], dtype=torch.float64),
        'e': torch.tensor([p['e'] for p in params_list], dtype=torch.float64),
    }
    
    rho_t = profile.rho
    a_phi_t = profile.a_phi
    da_phi_t = profile.da_phi
    
    # Solve u0
    u0 = torch.zeros((1, len(rho)), dtype=torch.complex128)
    du0 = torch.zeros((1, len(rho)), dtype=torch.complex128)
    
    curr_state = torch.tensor([[rho[0]**1, 1.0*rho[0]**0]], dtype=torch.complex128)
    u0[0, 0] = curr_state[0, 0]
    du0[0, 0] = curr_state[0, 1]
    
    for i in range(len(rho)-1):
        h = rho[i+1] - rho[i]
        # Approximate midpoint for profile
        a_mid = 0.5 * (a_phi_t[i] + a_phi_t[i+1])
        da_mid = 0.5 * (da_phi_t[i] + da_phi_t[i+1])
        curr_state = solver.rk4_step(rho_t[i], h, curr_state, params, 
                                     a_phi_t[i], da_phi_t[i],
                                     a_mid, da_mid,
                                     a_phi_t[i+1], da_phi_t[i+1])
        u0[0, i+1] = curr_state[0, 0]
        du0[0, i+1] = curr_state[0, 1]

    # Similarly for uinf (backward)
    uinf = torch.zeros((1, len(rho)), dtype=torch.complex128)
    duinf = torch.zeros((1, len(rho)), dtype=torch.complex128)
    
    rho_max = torch.tensor(rho[-1], dtype=torch.float64)
    k = torch.sqrt(params['chi']*params['chi'] + params['m']*params['m'])
    u_inf_init = torch.exp(-k * rho_max) / torch.sqrt(rho_max)
    du_inf_init = (-k - 0.5/rho_max) * u_inf_init
    
    curr_state = torch.stack([u_inf_init, du_inf_init], dim=1)
    uinf[0, -1] = curr_state[0, 0]
    duinf[0, -1] = curr_state[0, 1]
    
    for i in range(len(rho)-1, 0, -1):
        h = rho[i-1] - rho[i]
        # Approximate midpoint for profile
        a_mid = 0.5 * (a_phi_t[i] + a_phi_t[i-1])
        da_mid = 0.5 * (da_phi_t[i] + da_phi_t[i-1])
        curr_state = solver.rk4_step(rho_t[i], h, curr_state, params, 
                                     a_phi_t[i], da_phi_t[i],
                                     a_mid, da_mid,
                                     a_phi_t[i-1], da_phi_t[i-1])
        uinf[0, i-1] = curr_state[0, 0]
        duinf[0, i-1] = curr_state[0, 1]
        
    # Wronskian W0 = rho * (u0' * uinf - u0 * uinf')
    W0 = rho_t * (du0[0] * uinf[0] - u0[0] * duinf[0])
    W0_np = W0.detach().numpy()
    
    # Check constancy (ignoring endpoints where BC might be slightly off)
    W0_mid = W0_np[10:-10]
    variation = np.abs(W0_mid - np.mean(W0_mid)) / np.abs(np.mean(W0_mid))
    max_var = np.max(variation)
    
    assert max_var < 1e-5, f"Wronskian not constant enough: {max_var}"
