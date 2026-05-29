import torch
import pytest
from src.python.step_profile_effective_action import step_profile_analytic_ea

def test_step_profile_autograd():
    # Define parameters with requires_grad=True
    F_cal = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    lambd = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    
    # Run function
    ea = step_profile_analytic_ea(F_cal, lambd)
    
    # Mock a loss function
    loss = ea**2
    loss.backward()
    
    assert F_cal.grad is not None
    assert lambd.grad is not None
    assert not torch.allclose(F_cal.grad, torch.zeros_like(F_cal.grad))
    assert not torch.allclose(lambd.grad, torch.zeros_like(lambd.grad))
