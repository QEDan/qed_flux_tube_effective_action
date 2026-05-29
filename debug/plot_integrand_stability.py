import torch
import numpy as np
import matplotlib.pyplot as plt
from src.python.orchestrator import Orchestrator
from src.python.profiles import Sech2Profile

def plot_integrand_stability():
    rho = torch.linspace(0.01, 10.0, 500, dtype=torch.float64)
    B = 0.5
    lambd = 1.0
    profile = Sech2Profile(rho, B=B, lambd=lambd)
    
    chi_vals = [1.0, 5.0, 10.0, 20.0]
    
    orc = Orchestrator(device="cpu")
    
    # We want to inspect the integrand at a specific chi
    # Integrand = sum_{ml, s3} (G - G0 - Gsub)
    
    integrands = []
    for chi in chi_vals:
        # Use orchestrator to get the total inner sum for this chi
        # Orchestrator.compute_effective_action performs the full sum
        ml_range = list(range(-30, 31))
        sigma3_vals = [1, -1]
        
        # Mock chi values for the orchestrator
        action = orc.compute_effective_action(profile, [chi + 0.01j], ml_range, sigma3_vals, m=1.0)
        
        # We want the total_inner_sum from the orchestrator.
        # Since orchestrator is a bit of a black box, let's just use its logic here.
        total_chi = 0.0
        for ml in ml_range:
            for s3 in sigma3_vals:
                params = [{'chi': chi + 0.01j, 'ml': ml, 'sigma3': s3, 'm': 1.0, 'e': 1.0}]
                results_num, _ = orc.backend.solve_batch(params, profile)
                num_chi = torch.tensor([chi + 0.01j], device=orc.device, dtype=torch.complex128)
                num_ml = torch.tensor([ml], device=orc.device, dtype=torch.int32)
                g0 = orc.renormalizer.compute_g0(num_chi, num_ml, 1.0, rho)
                uv = orc.renormalizer.compute_uv_subtraction(num_chi, num_ml, 1.0, rho, profile)
                
                # Path A matching: Renorm = G_num - G_0 + UV_sub
                integrand = (results_num[0] - g0[0] + uv[0])
                total_chi += torch.sum(rho * integrand).item()
        
        # Apply global UV subtraction
        _, a_p, da_p = profile.get_arrays(as_numpy=False)
        b_p = (a_p / (rho + 1e-15) + da_p)
        area_b2 = torch.sum(rho * b_p**2 * (rho[1]-rho[0])).real
        uv_global = area_b2 / (3.0 * chi**4) # Simplified for chi >> m
        
        total_chi_ren = total_chi + uv_global
        integrands.append(total_chi_ren)
        
        print(f"chi: {chi:.2f}, Full sum: {total_chi:.2e}, Renormalized: {total_chi_ren:.2e}")
        
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
