import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import Sech2Profile

def plot_integrand_stability():
    rho = torch.linspace(0.01, 10.0, 500, dtype=torch.float64)
    B = 0.5
    lambd = 1.0
    profile = Sech2Profile(rho, B=B, lambd=lambd)
    
    chi_vals = np.linspace(1.0, 20.0, 50)
    
    orc = Orchestrator(device="cpu")
    
    # We want to inspect the integrand at a specific chi
    # Integrand = sum_{ml, s3} (G - G0 - Gsub)
    
    integrands = []
    for chi in chi_vals:
        # Manually invoke orchestrator internals to get integrand
        params = [{'chi': chi + 0.01j, 'ml': 1, 'sigma3': 1, 'm': 1.0, 'e': 1.0}]
        results_num, _ = orc.backend.solve_batch(params, profile)
        
        num_chi = torch.tensor([chi + 0.01j], device=orc.device, dtype=torch.complex128)
        num_ml = torch.tensor([1], device=orc.device, dtype=torch.int32)
        
        g0 = orc.renormalizer.compute_g0(num_chi, num_ml, 1.0, rho)
        uv = orc.renormalizer.compute_uv_subtraction(num_chi, num_ml, 1.0, rho, profile)
        
        integrand = (results_num[0] - g0[0] - uv[0])
        # Integrate over rho
        total = torch.sum(rho * integrand).item()
        integrands.append(total)
        
    plt.figure(figsize=(10, 6))
    plt.plot(chi_vals, np.abs(integrands), label=r"Renormalized Integrand $|\mathcal{I}_{ren}(\chi)|$")
    plt.yscale('log')
    plt.title(r"Integrand Decay $\mathcal{I}_{ren}(\chi)$ vs $\chi$")
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\log(|\mathcal{I}_{ren}|)$")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/integrand_stability.png")
    print("Plot saved to results/integrand_stability.png")

if __name__ == "__main__":
    plot_integrand_stability()
