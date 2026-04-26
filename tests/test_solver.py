import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src/python'))

import numpy as np
import matplotlib.pyplot as plt
from orchestrator import Orchestrator, generate_params_grid
from profiles import StepFunctionProfile

def test_solver():
    # Setup rho grid
    rho = np.linspace(0.01, 5.0, 500)
    
    # Setup profile: lambda=1.0, F=2*np.pi (so F_cal = 1.0)
    profile = StepFunctionProfile(rho, lambd=1.0, F=2*np.pi)
    
    # Setup orchestrator
    orc = Orchestrator(lib_path="./libsolver.so")
    
    # Generate parameters grid
    chi_values = [1.0, 2.0, 5.0]
    ml_values = [0, 1, 2]
    sigma3_values = [1, -1]
    params_grid = generate_params_grid(chi_values, ml_values, sigma3_values)
    
    print(f"Computing {len(params_grid)} Green's functions...")
    results = orc.compute_greens_function_batch(params_grid, profile)
    
    print(f"Results shape: {results.shape}")
    
    # Plot a few results
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
    print("Plot saved to test_solver_result.png")

if __name__ == "__main__":
    test_solver()
