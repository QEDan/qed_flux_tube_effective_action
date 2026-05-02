import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src/python'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from orchestrator import Orchestrator
from orchestrator import generate_params_grid
from profiles import StepFunctionProfile

def test_solver():
    # Setup rho grid
    rho = np.linspace(0.01, 5.0, 500)
    
    # Setup profile: lambda=1.0, F=2*np.pi (so F_cal = 1.0)
    profile = StepFunctionProfile(rho, lambd=1.0, F=2*np.pi)
    
    # Setup orchestrator
    orc = Orchestrator()
    
    # Generate parameters grid
    chi_values = [1.0, 2.0]
    ml_values = [0, 1]
    sigma3_values = [1, -1]
    params_grid = generate_params_grid(chi_values, ml_values, sigma3_values)
    
    results, w0 = orc.backend.solve_batch(params_grid, profile)
    
    if isinstance(results, torch.Tensor):
        results = results.detach().cpu().numpy()
    
    assert results.shape == (len(params_grid), len(rho))
    assert not np.any(np.isnan(results))
    # Green's function should be non-zero
    assert np.max(np.abs(results)) > 0

    # Optional: Plotting could be moved to a separate script or only run in certain modes
    # but keeping it for now as it was there.
    plt.figure(figsize=(10, 6))
    for i in range(min(5, len(params_grid))):
        p = params_grid[i]
        label = f"chi={p['chi']}, ml={p['ml']}, s3={p['sigma3']}"
        plt.plot(rho, results[i].real, label=label)
    
    plt.title("Green's Function G(rho, rho) - Real Part")
    plt.xlabel("rho")
    plt.ylabel("G(rho, rho)")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_solver_result.png")
