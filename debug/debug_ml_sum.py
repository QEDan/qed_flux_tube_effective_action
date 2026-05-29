import torch
import numpy as np
from src.python.orchestrator import Orchestrator
from src.python.profiles import StepFunctionProfile

def debug_ml_sum():
    print("--- ML Sum Convergence Debug ---")
    
    rho = torch.linspace(0.1, 5.0, 100, dtype=torch.float64)
    lambd = 10.0 # Wide profile
    F = 0.1 # Small flux
    profile = StepFunctionProfile(rho, lambd=lambd, F=F)
    
    orc = Orchestrator(device="cpu")
    m = 1.0
    chi = 2.0
    
    ml_to_test = list(range(0, 101, 10))
    
    for ml in ml_to_test:
        params = [{'chi': chi, 'ml': ml, 'sigma3': 1, 'm': m, 'e': 1.0}]
        # Solve ODE
        num_res, _ = orc.backend.solve_batch(params, profile)
        
        # Solve background (B=0)
        from src.python.profiles import FieldProfile
        bg_profile = FieldProfile(rho)
        num_bg, _ = orc.backend.solve_batch(params, bg_profile)
        
        diff = num_res[0] - num_bg[0]
        integral = torch.sum(diff * (rho[1]-rho[0])).item()
        
        print(f"ml={ml:3d}: term_integral={integral:.4e}")

if __name__ == "__main__":
    debug_ml_sum()
