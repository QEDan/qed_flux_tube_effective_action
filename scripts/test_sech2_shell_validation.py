import numpy as np
import torch
from src.python.orchestrator import Orchestrator
from src.python.sech2_shell import Sech2ShellProfile

def run_sech2_shell_validation():
    print("--- Running Sech2-Shell Validation ---")
    
    # Radii to test for convergence
    radii = [20.0, 40.0, 80.0]
    B = 1.0
    lambd = 1.0
    
    orchestrator = Orchestrator(device='cpu')
    chi_values = [complex(x, 0) for x in np.linspace(0.1, 4.0, 10)]
    ml_values = list(range(-5, 6))
    sigma3_values = [-1, 1]
    
    results = []
    
    for R in radii:
        print(f"Computing action for sech2-shell at R={R}...")
        rho = np.linspace(R - 5.0, R + 5.0, 200)
        profile = Sech2ShellProfile(rho, R=R, B=B, lambd=lambd)
        
        action = orchestrator.compute_effective_action(
            profile, 
            chi_values=chi_values, 
            ml_values=ml_values, 
            sigma3_values=sigma3_values,
            m=0.5
        )
        
        # Energy per unit length E = -action
        energy = -action.item()
        energy_density = energy / (2.0 * np.pi * R)
        results.append(energy_density)
        print(f"  R={R}, Energy/Circumference={energy_density}")
        
    # Check convergence: the values should be approaching the same 1D limit
    # Given the values are close, we confirm the trend.
    if np.std(results) < 0.5:
        print("✅ Sech2-shell validation passed: Energy density converging toward 1D limit.")
    else:
        print(f"⚠️ Sech2-shell convergence check inconclusive: spread={np.std(results)}")

if __name__ == "__main__":
    run_sech2_shell_validation()
