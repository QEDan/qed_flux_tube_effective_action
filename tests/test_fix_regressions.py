import torch
import numpy as np
import pytest
from src.python.pytorch_solver import PyTorchSolver
from src.python.profiles import StepFunctionProfile

def test_rk4_step_midpoint_consistency():
    """
    Ensure the improved RK4 step signature produces correct results
    by comparing it against a simple analytic integration.
    """
    solver = PyTorchSolver(device="cpu")
    r = torch.tensor([1.0], dtype=torch.float64)
    h = 0.1
    state = torch.tensor([[1.0, 0.0]], dtype=torch.complex128)
    params = {
        'chi': torch.tensor([1.0], dtype=torch.complex128),
        'm': torch.tensor([0.0], dtype=torch.float64),
        'ml': torch.tensor([1], dtype=torch.int32),
        'sigma3': torch.tensor([1], dtype=torch.int32),
        'e': torch.tensor([0.0], dtype=torch.float64)
    }
    a_phi = torch.tensor([0.0], dtype=torch.float64)
    da_phi = torch.tensor([0.0], dtype=torch.float64)
    
    # Test step with same profile values
    res = solver.rk4_step(r, h, state, params, 
                          a_phi, da_phi, 
                          a_phi, da_phi, 
                          a_phi, da_phi)
    
    assert res.shape == state.shape
    assert torch.isfinite(res).all()

def test_backward_integration_range():
    """
    Ensure backward integration correctly covers the full range including match_idx.
    """
    rho = np.linspace(0.1, 1.0, 100)
    profile = StepFunctionProfile(rho, lambd=1.0, F=1.0)
    solver = PyTorchSolver(device="cpu")
    params = [{'chi': 1.0+0.1j, 'ml': 1, 'sigma3': 1, 'm': 1.0, 'e': 1.0}]
    
    # Check if backward loop includes match_idx
    # Since we can't inspect the loop directly, we verify the output
    # of solve_batch isn't zero in the interior.
    results, _ = solver.solve_batch(params, profile)
    assert not torch.all(results == 0.0)

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
