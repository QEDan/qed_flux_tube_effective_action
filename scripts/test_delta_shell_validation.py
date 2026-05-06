import numpy as np
import torch
from src.python.orchestrator import Orchestrator
from src.python.delta_shell import DeltaFunctionShellProfile

def run_delta_shell_validation():
    print("--- Running Delta-Function Shell Validation ---")
    
    # 1. Define physical parameters
    # Shell radius R much larger than lambda (though Delta function has no lambda)
    # Using a large radius for the test.
    R = 10.0
    F = 1.0 # Flux
    
    # Setup grid
    rho = np.linspace(0.01, 20.0, 500)
    profile = DeltaFunctionShellProfile(rho, R=R, F=F)
    
    # Initialize Orchestrator
    orchestrator = Orchestrator()
    
    # Parameters
    chi_values = [complex(x, 0) for x in np.linspace(0.1, 5.0, 10)]
    ml_values = list(range(-5, 6))
    sigma3_values = [-1, 1]
    
    # Compute effective action
    print(f"Computing action for delta-shell at R={R}...")
    action = orchestrator.compute_effective_action(
        profile, 
        chi_values=chi_values, 
        ml_values=ml_values, 
        sigma3_values=sigma3_values,
        m=1.0
    )
    
    print(f"Action result: {action.item()}")
    
    # Validation criterion: 
    # The action for a delta-shell should be finite and well-defined.
    # The jump conditions (from our Sage validation) guarantee numerical consistency.
    if torch.isfinite(action):
        print("✅ Delta-shell validation passed: Action is finite.")
    else:
        print("❌ Delta-shell validation failed: Action is non-finite.")

if __name__ == "__main__":
    run_delta_shell_validation()
