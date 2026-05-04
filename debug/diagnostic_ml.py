import torch
import numpy as np
import sys
import os

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import StepFunctionProfile

def diagnostic_ml_convergence():
    print("--- Spectral Sum Diagnostic: Sech2 Profile (B=0.5) ---")
    
    # Grid
    rho = torch.linspace(0.1, 5.0, 100, dtype=torch.float64)
    lambd = 2.0
    B_peak = 0.5
    from profiles import Sech2Profile
    profile = Sech2Profile(rho, B=B_peak, lambd=lambd)
    
    orc = Orchestrator(device="cpu")
    m = 1.0
    chi = 5.0 # Test a higher chi
    
    # Check specific ml values
    ml_to_test = [0, 10, 50, 100]
    
    for ml in ml_to_test:
        params = [{'chi': chi, 'ml': ml, 'sigma3': 1, 'm': m, 'e': 1.0}]
        num_results, w0 = orc.backend.solve_batch(params, profile)
        num_g0 = orc.renormalizer.compute_g0(torch.tensor([chi], dtype=torch.complex128), torch.tensor([ml]), m, rho)
        num_uv = orc.renormalizer.compute_uv_subtraction(torch.tensor([chi], dtype=torch.complex128), torch.tensor([ml]), m, rho, profile)
        
        diff_raw = num_results[0] - num_g0[0]
        diff_renorm = num_results[0] - num_g0[0] + num_uv[0]
        
        print(f"ml={ml}:")
        print(f"  num_results[mid]: {num_results[0, 50].item():.4e}")
        print(f"  num_g0[mid]:      {num_g0[0, 50].item():.4e}")
        print(f"  num_uv[mid]:      {num_uv[0, 50].item():.4e}")
        print(f"  Raw Diff:         {diff_raw[50].item():.4e}")
        print(f"  Renorm Diff:      {diff_renorm[50].item():.4e}")
        
        if torch.abs(diff_renorm[50]) > torch.abs(diff_raw[50]):
            print("  ⚠️ UV SUBTRACTION INCREASED THE DISCREPANCY!")

if __name__ == "__main__":
    diagnostic_ml_convergence()
