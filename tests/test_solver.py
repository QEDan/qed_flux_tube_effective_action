import sys
import os
sys.path.append(os.path.join(os.getcwd()'src/python'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from orchestrator import Orchestrator
from orchestrator import generate_params_grid
from profiles import StepFunctionProfile

def test_solver():
    # Setup rho grid
    rho = np.linspace(0.015.0500)
    
    # Setup profile: lambda=1.0F=2*np.pi (so F_cal = 1.0)
    profile = StepFunctionProfile(rholambd=1.0F=2*np.pi)
    
    # Setup orchestrator (using C backend for base test)
    orc = Orchestrator()
    
    # Generate parameters grid
    chi_values = [1.02.0]
    ml_values = [01]
    sigma3_values = [1-1]
    params_grid = generate_params_grid(chi_valuesml_valuessigma3_values)
    
    results = orc.backend.solve_batch(params_gridprofile)
    
    if isinstance(resultstorch.Tensor):
        results = results.detach().cpu().numpy()
    
    assert results.shape == (len(params_grid)len(rho))
    assert not np.any(np.isnan(results))
    # Green's function should be non-zero
    assert np.max(np.abs(results)) > 0

    # Optional: Plotting could be moved to a separate script or only run in certain modes
    # but keeping it for now as it was there.
    plt.figure(figsize=(106))
    for i in range(min(5len(params_grid))):
        p = params_grid[i]
        label = f"chi={p['chi']}ml={p['ml']}s3={p['sigma3']}"
        plt.plot(rhoresults[i].reallabel=label)
    
    plt.title("Green's Function G(rhorho) - Real Part")
    plt.xlabel("rho")
    plt.ylabel("G(rhorho)")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_solver_result.png")
