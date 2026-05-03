import torch
import numpy as np
import sys
import os

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import StepFunctionProfile

def check_term_decay():
    lambd = 1.0
    F = 2.0 * np.pi * 0.1
    m = 1.0
    rho_val = 0.5
    
    rho = torch.tensor([rho_val], dtype=torch.float64)
    profile = StepFunctionProfile(rho, lambd=lambd, F=F)
    
    chi_vals = [1.0, 10.0, 100.0]
    ml = 1
    s3 = 1
    
    orc = Orchestrator(device="cpu")
    
    print(f"Checking W0 scaling for ml=1, s3=1")
    for chi in [10.0, 100.0]:
        params = [{'chi': chi, 'ml': 1, 'sigma3': 1, 'm': m, 'e': 1.0}]
        res, w0 = orc.backend.solve_batch(params, profile)
        print(f"chi={chi:6.1f}: W0={w0[0].item():10.2e}")

if __name__ == "__main__":
    check_term_decay()
