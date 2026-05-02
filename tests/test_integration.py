import torch
import numpy as np
import pytest
import sys
import os

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import StepFunctionProfile

def test_full_orchestrator_pytorch():
    # Setup grid
    chi_values = [1.0, 2.0]
    ml_values = [0, 1]
    sigma3_values = [-1, 1]
    rho = torch.linspace(0.01, 2.0, 50, dtype=torch.float64)
    
    profile = StepFunctionProfile(rho, lambd=1.0, F=1.0)
    
    orch = Orchestrator(batch_size=4)
    action = orch.compute_effective_action(profile, chi_values, ml_values, sigma3_values)
    
    assert action is not None
    assert isinstance(action, torch.Tensor)
    assert torch.isfinite(action)
    print(f"Action: {action}")

def test_full_orchestrator_asymptotic():
    # Setup grid with a very large chi
    chi_values = [1000.0]
    ml_values = [0]
    sigma3_values = [1]
    rho = torch.linspace(0.01, 2.0, 50, dtype=torch.float64)
    
    profile = StepFunctionProfile(rho, lambd=1.0, F=1.0)
    
    orch = Orchestrator(batch_size=4)
    # Threshold is 100.0 by default, so 1000.0 should trigger asymptotic regime
    action = orch.compute_effective_action(profile, chi_values, ml_values, sigma3_values, chi_threshold=100.0)
    
    # In asymptotic regime, we currently set renorm_g to 0, so action should be 0
    assert action == 0.0
    print(f"Asymptotic Action: {action}")
