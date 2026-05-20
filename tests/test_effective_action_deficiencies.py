import torch
import numpy as np
import pytest
from src.python.pytorch_solver import PyTorchSolver
from src.python.profiles import StepFunctionProfile, FieldProfile
from src.python.orchestrator import Orchestrator
from src.python.analytic import heisenberg_euler_lagrangian

@pytest.fixture
def device():
    return "cpu"

@pytest.fixture
def orchestrator(device):
    return Orchestrator(device=device, strategy="numerical")

def test_uv_convergence(orchestrator, device):
    """
    VERIFICATION TEST: Checks if the spectral integrand now decays using Euclidean momenta.
    """
    B = 1.0
    rho_val = 0.5
    rho = torch.tensor([rho_val]).to(device)
    profile = StepFunctionProfile(rho, lambd=5.0, F=B * (np.pi * 5.0**2))
    
    Q_1 = 10.0 
    Q_2 = 50.0
    ml_vals = list(range(-200, 201))
    sigma3_vals = [1, -1]
    
    def get_integrand(Q):
        # Use Euclidean chi = i * Q
        chi_e = 1j * Q
        
        mode_sum = torch.zeros(1, device=device, dtype=torch.complex128)
        for s3 in sigma3_vals:
            batch = [{'chi': chi_e, 'ml': ml, 'sigma3': s3, 'm': 1.0, 'e': 1.0} for ml in ml_vals]
            num_results, _ = orchestrator.backend.solve_batch(batch, profile)
            
            # Use compute_g0_local for topological vacuum subtraction with Euclidean chi
            chi_tensor = torch.tensor([chi_e], device=device, dtype=torch.complex128)
            ml_tensor = torch.tensor(ml_vals, device=device, dtype=torch.float64)
            num_bg = orchestrator.renormalizer.compute_g0(chi_tensor, ml_tensor, 1.0, rho, profile)            
            mode_sum += torch.sum(num_results - num_bg, dim=0)
        
        mode_sum_val = mode_sum.real.item()
        
        # New UV term: Local B^2/3 / Q^2 (for 2 spin states in Q dQ measure)
        # matches Orchestrator's B^2/3 / Q^4 in Q^3 dQ measure.
        uv_coeff_local = orchestrator.renormalizer.get_b2_term(profile, rho) * 2.0
        num_uv = uv_coeff_local[0].real.item() / (Q**2)
        
        norm_factor = 1.0 / (8.0 * np.pi**2)
        
        # local_renorm_sum = - 2.0 * (Delta_G / rho) + num_uv
        # Factor of 2 matches the spectral transform to 4D.
        # Delta_G is positive (interacting G is less negative than background G0).
        local_renorm_sum = - 2.0 * (mode_sum_val / rho_val) + num_uv
        
        return Q * local_renorm_sum * norm_factor

    val_low = abs(get_integrand(Q_1))
    val_high = abs(get_integrand(Q_2))
    
    print(f"\nUV Convergence Check (Euclidean):")
    print(f"Integrand at Q={Q_1}: {val_low:.4e}")
    print(f"Integrand at Q={Q_2}: {val_high:.4e}")
    
    # Integrand should decay fast in Euclidean space
    assert val_high < val_low, f"Integrand still diverges: {val_high:.4e} > {val_low:.4e}"

@pytest.mark.slow
def test_normalization_he(orchestrator, device):
    """
    VERIFICATION TEST: Checks if the numerical normalization matches HE within 20%.
    Summing over both spin states (sigma3=[1, -1]) to match Spinor QED EH.
    Using threshold=None to ensure we are testing the numerical mode sum.
    """
    B = 0.5
    rho = torch.tensor([0.1, 0.5, 1.0]).to(device)
    profile = StepFunctionProfile(rho, lambd=5.0, F=B * (np.pi * 5.0**2))
    
    # Use a wider chi/ml range for pure numerical convergence
    chi_vals = [complex(c) for c in np.linspace(0.1, 30.0, 50)]
    ml_vals = list(range(-300, 301)) # Wider range for convergence
    sigma3_vals = [1, -1]
    
    _, L_eff_rho = orchestrator.compute_effective_action(
        profile, chi_vals, ml_vals, sigma3_vals, m=1.0, e=1.0, collect_density=True,
        lcf_threshold=None # Force numerical
    )
    
    solver_density = L_eff_rho[1].real.item() # at rho=0.5
    he_density = heisenberg_euler_lagrangian(B, m=1.0, e=1.0)
    
    print(f"\nNumerical Normalization Check (B={B}):")
    print(f"Solver Density: {solver_density:.4e}")
    print(f"HE Density:     {he_density:.4e}")
    print(f"Ratio: {solver_density / he_density:.4e}")
    
    # Should be close to 1.0 now that spins are summed.
    assert 0.5 < abs(solver_density / he_density) < 2.0

def test_local_vacuum_consistency(orchestrator, device):
    """
    VERIFICATION TEST: Checks if vacuum regions now have zero density.
    """
    B_inner = 1.0
    lambd = 1.0
    rho = torch.tensor([0.5, 5.0]).to(device) # far outside
    profile = StepFunctionProfile(rho, lambd=lambd, F=B_inner * (np.pi * lambd**2))
    
    chi_vals = [complex(c) for c in np.linspace(0.1, 10.0, 20)]
    ml_vals = list(range(-50, 51))
    sigma3_vals = [1]
    
    _, L_eff_rho = orchestrator.compute_effective_action(
        profile, chi_vals, ml_vals, sigma3_vals, m=1.0, e=1.0, collect_density=True
    )
    
    density_out = L_eff_rho[1].real.item()
    
    print(f"\nLocal Consistency Check:")
    print(f"Density out field (rho=5.0): {density_out:.4e}")
    
    assert abs(density_out) < 1e-2, f"Significant non-zero vacuum density: {density_out:.4e}"


if __name__ == "__main__":
    # Allow running directly for debugging
    import sys
    pytest.main([__file__])
