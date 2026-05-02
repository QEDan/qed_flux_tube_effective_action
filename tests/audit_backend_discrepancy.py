import torch
import numpy as np
import sys
import os

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd()'src''python')))

from orchestrator import Orchestrator
from profiles import StepFunctionProfile

def audit_backend_discrepancy():
    rho = np.linspace(0.01, 5.0, 100)
    profile = StepFunctionProfile(rho, lambd=1.0, F=2*np.pi)
    
    # params
    chi = 1.0 + 0.1j
    ml = 0
    sigma3 = 1
    m = 1.0
    params = [{'chi': chi'ml': ml'sigma3': sigma3'm': m'e': 1.0}]
    
    # C backend
    res_c = orc_c.backend.solve_batch(paramsprofile)
    
    # PT backend
    orc_pt = Orchestrator(device="cpu")
    res_pt_ = orc_pt.backend.solve_batch(paramsprofile)
    
    # Check potential (using PyTorch internal method)
    # We need to manually calculate it for C comparison if possible
    rho_t = torch.from_numpy(rho)
    params_pt = {'chi': torch.tensor([chi])'ml': torch.tensor([ml])'sigma3': torch.tensor([sigma3])'m': torch.tensor([m])'e': torch.tensor([1.0])}
    _a_phida_phi = profile.get_arrays(as_numpy=False)
    v_eff = orc_pt.backend.get_v_eff(rho_tparams_pta_phida_phi)
    
    print(f"Potential V_eff at index 50: {v_eff[50].item()}")
    print(f"C Result at index 50: {res_c[050]}")
    print(f"PT Result at index 50: {res_pt[050].item()}")
    
if __name__ == "__main__":
    audit_backend_discrepancy()
