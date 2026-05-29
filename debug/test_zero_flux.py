import torch
import numpy as np
from src.python.orchestrator import Orchestrator
from src.python.profiles import ZeroFluxProfile

def exact_zero_flux_ea(B_peak, lambd, m):
    """
    Theoretical result for Zero Flux profile EH part.
    S = \int 2\pi \rho d\rho \mathcal{L}_{eff}(B(\rho))
    B(\rho) = B_peak * (1 - 2*rho^2/lambd^2) for rho < lambd.
    """
    e = 1.0
    rho_grid = np.linspace(0.0, lambd, 1000)
    B_profile = B_peak * (1.0 - 2.0 * rho_grid**2 / lambd**2)
    
    # EH term: (eB)^2 / 24pi^2 * ln(m^2/eB)
    B_safe = np.where(np.abs(B_profile) < 1e-10, 1e-10, B_profile)
    L_eff = (e * B_safe)**2 / (24.0 * np.pi**2) * np.log(m**2 / (e * np.abs(B_safe)))
    
    return 2.0 * np.pi * np.trapezoid(rho_grid * L_eff, rho_grid)

def test_zero_flux_matching():
    lambd = 1.0
    m = 1.0
    B = 0.1
    
    rho = torch.linspace(0.01, 5.0, 500, dtype=torch.float64)
    profile = ZeroFluxProfile(rho, B=B, lambd=lambd)
    
    # Converged grid
    chi_values = np.linspace(1.01, 10.0, 20).tolist()
    ml_values = list(range(-10, 11))
    sigma3_values = [1, -1]
    
    orc = Orchestrator(device="cpu")
    
    print("Computing Numerical EA for Zero Flux Profile...")
    action = orc.compute_effective_action(profile, chi_values, ml_values, sigma3_values, m=m)
    
    num_val = action.real.item()
    ana_val = exact_zero_flux_ea(B, lambd, m)
    
    print(f"Numerical Action: {num_val}")
    print(f"Analytic EH:     {ana_val}")
    
    if abs(num_val / ana_val - 1.0) < 0.2:
        print("✅ Test passed: Numerical result matches analytic expectation for zero flux profile.")
    else:
        print(f"❌ Test failed: Ratio is {num_val / ana_val if ana_val != 0 else 'inf'}")

if __name__ == "__main__":
    test_zero_flux_matching()
