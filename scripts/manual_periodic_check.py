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
    # Expand mode range from 400 to 1000
    ml_vals = list(range(-500, 501))
    
    # We need to accumulate the density properly as per the Orchestrator's internal logic.
    # Total effective action density is integral over chi of (chi^3 * exp(-m^2/chi^2) * Renorm_Sum).
    
    # Let's perform the integration properly:
    # 1. Renorm_Sum is G_field - G_vacuum - UV_sub
    # 2. Integrate over modes (ml)
    # 3. Integrate over chi
    # 4. Multiply by chi^3 * exp(-m^2/chi^2) * -(1/pi)
    
    L_solver = np.zeros(len(rho_vals), dtype=np.complex128)
    
    vacuum_profile = FieldProfile(rho_vals)
    
    # Store per-chi densities to perform integral over chi
    L_solver_spatial = np.zeros(len(rho_vals), dtype=np.complex128)
    
    print("Starting Manual Solver Integration...")
    for i, chi in enumerate(chi_vals):
        print(f"  chi = {chi:.1f}")
        batch = [{'chi': complex(chi), 'ml': ml, 'sigma3': 1, 'm': 1.0, 'e': 1.0} for ml in ml_vals]
        num_results, _ = orchestrator.backend.solve_batch(batch, profile)
        num_bg, _ = orchestrator.backend.solve_batch(batch, vacuum_profile)
        
        renorm_sum = torch.sum(num_results - num_bg, dim=0).detach().cpu().numpy()
        print(f"DEBUG: Max renorm_sum value: {np.max(np.abs(renorm_sum))}")
        # Correct EH subtraction: B^2 / (6 * chi^4)
        b2_term = orchestrator.renormalizer.get_b2_term(profile, rho_vals).detach().cpu().numpy()
        uv_sub = b2_term / (6.0 * chi**4)
        
    # Theoretical integration measure from greensfunc.tex eqn:soln:
    # Action = -pi^2/2 * sum(ml) * Integral(chi^3 dchi) * Integral(rho^2 drho) * Renorm
    # For intensive density, we evaluate the local integrand and apply the chi-integral measure.
    
    # 1. Local intensive integrand per chi (renorm_sum is already local density)
    local_integrand = (renorm_sum - uv_sub)
    
    # The solver returns 'renorm_sum' which is G(rho, chi). 
    # To get effective action density, we integrate chi^3 * local_integrand * dchi.
    # Theoretical factor: -pi^2/2
    
    scaling_factor = - (np.pi**2 / 2.0)
    L_solver_spatial = np.zeros_like(rho_vals.cpu().numpy(), dtype=np.complex128)
    
    for i, chi in enumerate(chi_vals):
        # Calculate local integrand (inside loop where renorm_sum is available)
        # ... logic ...
        L_solver_spatial += (renorm_sum - uv_sub) * (chi**3 * dchi * scaling_factor)
    
    # Final scaling check: L_solver_spatial is now the effective action density
    # If this is still 10^4 too large, it implies our chi^3 * dchi measure is 
    # being applied as a global extensive measure rather than intensive density integration.
    L_solver_scaled = L_solver_spatial / np.sum(chi_vals**3 * dchi)


    # 1. Compare Solver vs LCF at a central constant field point
    # Select rho = lambd/2 (constant B region)
    idx = len(rho_vals) // 2
    solver_density = L_solver_spatial[idx] * (-1.0/np.pi)
    b_val = B_vals[idx]
    
    # EH density at this B_val
    chi_test = np.linspace(0.1, 80.0, 100)
    dchi_test = chi_test[1] - chi_test[0]
    # LCF formula for Euler-Heisenberg Lagrangian: L = -1/(8*pi^2) * Integral (ds/s^3) exp(-m^2*s) * (eB*s / sinh(eB*s) - 1 + eB^2*s^2/6)
    # Our code uses chi = 1/sqrt(s)
    # L = - (1/pi) * Integral chi^3 dchi * exp(-m^2/chi^2) * ( (eB/chi^2)/sinh(eB/chi^2) - 1 + (eB/chi^2)^2/6 )
    
    def eh_density_pt(B, chi_vals):
        T = 1.0 / (chi_vals**2)
        x = B * T
        core = (x / np.sinh(x)) - 1.0 + (x**2 / 6.0)
        integrand = chi_vals**3 * np.exp(-1.0 / chi_vals**2) * core
        return - (1.0 / np.pi) * np.sum(integrand * (chi_vals[1]-chi_vals[0]))
    
    theoretical_density = eh_density_pt(b_val, chi_test)
    print(f"\n--- Normalization Audit ---")
    print(f"B_val at center: {b_val:.4e}")
    print(f"Solver Density at center: {solver_density.real:.4e}")
    print(f"Theoretical EH Density at center: {theoretical_density:.4e}")
    print(f"Normalization Ratio: {theoretical_density / solver_density.real:.4e}")

    L_lcf = compute_lcf_density(B_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(rho_vals.cpu().numpy(), L_solver_scaled.real, 'o-', label='Manual Solver (Scaled)')
    plt.plot(rho_vals.cpu().numpy(), L_lcf, '--', label='LCF Approximation')
    plt.axhline(0, color='black', lw=0.5)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\rho \times \Delta \mathcal{L}(\rho)$')
    plt.title("Action Density: Manual Solver vs LCF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/periodic_manual_check.png")
    print("Plot saved to results/periodic_manual_check.png")
    
    print(f"Max Solver Density: {np.max(L_solver_scaled)}")
    print(f"Max LCF Density: {np.max(L_lcf)}")

if __name__ == "__main__":
    run_manual_validation()
