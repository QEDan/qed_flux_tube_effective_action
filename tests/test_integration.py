import torch
import numpy as np
import pytest
from src.python.orchestrator import Orchestrator
from src.python.profiles import StepFunctionProfile

def test_full_orchestrator_pytorch():
    # Setup grid
    chi_values = [1.0, 2.0]
    ml_values = [0, 1]
    sigma3_values = [-1, 1]
    rho = torch.linspace(0.01, 2.0, 50, dtype=torch.float64)

    profile = StepFunctionProfile(rho, lambd=1.0, F=1.0)

    orch = Orchestrator(batch_size=4)
    # Run integration
    action, _ = orch.compute_effective_action(profile, chi_values, ml_values, sigma3_values)

    assert isinstance(action, torch.Tensor)
    assert torch.isfinite(action)

    print(f"Action: {action}")
