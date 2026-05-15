import torch
import numpy as np
import matplotlib.pyplot as plt
from src.python.periodic_profile import LatticeBumpProfile
from src.python.orchestrator import Orchestrator
from src.python.profiles import FieldProfile

def compute_lcf_density(B_vals, m=1.0, e=1.0, chi_max=100.0, n_chi=500):
    chi_vals = np.linspace(0.1, chi_max, n_chi)
    dchi = chi_vals[1] - chi_vals[0]
    density = np.zeros_like(B_vals)
    for i, B in enumerate(B_vals):
        if abs(B) < 1e-10:
            density[i] = 0.0
            continue
        T = 1.0 / (chi_vals**2)
        x = e * B * T
        
        integrand_core = np.zeros_like(x)
        small_mask = np.abs(x) < 1e-3
        large_mask = ~small_mask
        
        integrand_core[small_mask] = (7.0/360.0) * x[small_mask]**4
        # Use np.sinh and handle large values
        safe_x = x[large_mask]
        integrand_core[large_mask] = (safe_x / np.sinh(safe_x)) - 1.0 + (safe_x**2 / 6.0)
        
        integrand = chi_vals**3 * np.exp(-m**2 / chi_vals**2) * integrand_core
        density[i] = - (1.0 / np.pi) * np.sum(integrand * dchi)
    return density

def run_manual_validation():
    device = "cpu"
    orchestrator = Orchestrator(device=device, batch_size=512)
    
    a_val = np.sqrt(8.0)
    lambd = 0.6 * a_val
    F_val = 2.0 * np.pi
    
    rho_vals = torch.linspace(0.01, a_val/2.0, 500).to(device)
    profile = LatticeBumpProfile(rho_vals, a=a_val, lambd=lambd, F=F_val)
    B_vals = profile.B_vals.detach().numpy()
    
    # We'll use a linear chi grid to avoid weight complications for now
    chi_vals = np.linspace(0.1, 80.0, 30)
    dchi = chi_vals[1] - chi_vals[0]
    ml_vals = list(range(-200, 201))
    
    L_solver = np.zeros(len(rho_vals))
    
    vacuum_profile = FieldProfile(rho_vals)
    
    print("Starting Manual Solver Integration...")
    for chi in chi_vals:
        print(f"  chi = {chi:.1f}")
        batch = [{'chi': complex(chi), 'ml': ml, 'sigma3': 1, 'm': 1.0, 'e': 1.0} for ml in ml_vals]
        num_results, _ = orchestrator.backend.solve_batch(batch, profile)
        num_bg, _ = orchestrator.backend.solve_batch(batch, vacuum_profile)
        
        # Delta G sum
        sum_ml = torch.sum(num_results - num_bg, dim=0).real.detach().numpy()
        
        # Subtract B^2 term. 
        # For ScQED, the term in the proper time integral is +1/6 (eBT)^2.
        # In terms of chi, T=1/chi^2, so it is (eB)^2 / (6 chi^4).
        # Our solver computes sum_ml(Delta G).
        # Relation: L = -1/pi * Integral chi^3 dchi * [ sum_ml(Delta G) * (??) ]
        # From LCF match, sum_ml(Delta G) should be compared to (x/sinh x - 1).
        # So L_solver = -1/pi * Integral chi^3 dchi * [ sum_ml(Delta G) + (eB)^2 / (6 chi^4) ]
        
        uv_sub = (B_vals**2) / (6.0 * chi**4)
        renorm_sum = sum_ml + uv_sub
        
        # L(rho) = -1/pi * Integral chi^3 * renorm_sum
        L_solver += - (1.0 / np.pi) * (chi**3 * np.exp(-1.0 / chi**2) * renorm_sum * dchi)

    L_lcf = compute_lcf_density(B_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rho_vals.cpu().numpy(), L_solver, 'o-', label='Manual Solver (ScQED, UV Subtracted)')
    plt.plot(rho_vals.cpu().numpy(), L_lcf, '--', label='LCF Approximation')
    plt.axhline(0, color='black', lw=0.5)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\Delta \mathcal{L}(\rho)$')
    plt.title("Action Density: Manual Solver vs LCF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/periodic_manual_check.png")
    print("Plot saved to results/periodic_manual_check.png")
    
    print(f"Max Solver Density: {np.max(L_solver)}")
    print(f"Max LCF Density: {np.max(L_lcf)}")

if __name__ == "__main__":
    run_manual_validation()
