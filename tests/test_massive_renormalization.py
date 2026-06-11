import torch
import numpy as np
import pytest
from src.python.orchestrator import Orchestrator
from src.python.profiles import StepFunctionProfile
from src.python import constants

def test_q_min_independence():
    """
    Verifies that the integrated effective action is independent of the 
    small-Q spectral cutoff when using the robust massive renormalization.
    """
    m = constants.ELECTRON_MASS
    e = constants.ELECTRON_CHARGE
    lambd = 2.0
    F_cal = 1.0
    F = (constants.TWO_PI * F_cal) / e
    
    n_rho = 20
    rho = torch.linspace(0.1, 1.2 * lambd, n_rho, dtype=torch.float64)
    profile = StepFunctionProfile(rho, lambd, F, e=e, smooth_width=0.2)
    
    orch = Orchestrator(strategy="numerical", device='cpu', batch_size=4096, global_mode=False)
    
    Q_max = 5.0
    n_Q = 20
    ml_values = [0, 1] # Small subset for speed
    sigma3_values = [1, -1]
    
    # Run with Q_min = 0.1
    chi_1 = torch.linspace(0.1, Q_max, n_Q, dtype=torch.complex128)
    action_1, _ = orch.compute_effective_action(profile, chi_1, ml_values, sigma3_values, m=m, e=e)
    
    # Run with Q_min = 0.01
    chi_2 = torch.linspace(0.01, Q_max, n_Q, dtype=torch.complex128)
    action_2, _ = orch.compute_effective_action(profile, chi_2, ml_values, sigma3_values, m=m, e=e)
    
    print(f"Action (Q_min=0.1):  {action_1.item():.6e}")
    print(f"Action (Q_min=0.01): {action_2.item():.6e}")
    
    # Relative difference should be very small (< 2%)
    rel_diff = abs(action_1.item() - action_2.item()) / (abs(action_1.item()) + 1e-20)
    assert rel_diff < 0.02

def test_q_zero_regularity():
    """
    Verifies that the UV subtraction term is finite at Q=0.
    """
    from src.python.renormalization import Renormalizer
    renorm = Renormalizer(device="cpu", strategy="analytic")
    
    m = 1.0
    rho = torch.tensor([1.0], dtype=torch.float64)
    profile = StepFunctionProfile(rho, lambd=2.0, F=1.0)
    
    chi_zero = torch.tensor([0.0], dtype=torch.complex128)
    ml = torch.tensor([0], dtype=torch.int32)
    e = constants.ELECTRON_CHARGE
    
    uv_sub = renorm.compute_uv_subtraction(chi_zero, ml, m, e, rho, profile)
    
    assert torch.isfinite(uv_sub)
    # Form: - b2 / (Q^2 + m^2)^2. At Q=0: - b2 / m^4
    b2 = renorm.get_b2_term(profile, rho, e=e)
    expected = - b2 / (m**4)
    assert torch.allclose(uv_sub.real, expected.real, rtol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
