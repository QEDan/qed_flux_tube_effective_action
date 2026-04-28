import sys
import os
import torch
import pytest

# Add src/python to path
sys.path.append(os.path.join(os.getcwd(), 'src/python'))

from orchestrator import Orchestrator
from profiles import StepFunctionProfile

def test_telemetry_density_integrand():
    """
    Verify that the orchestrator correctly returns the density integrand 
    when the collect_density flag is enabled.
    """
    rho = torch.linspace(0.1, 2.0, 50, dtype=torch.float64)
    profile = StepFunctionProfile(rho, lambd=1.0, F=1.0)
    
    chi_values = [1.0, 2.0]
    ml_values = [0]
    sigma3_values = [1]
    
    orch = Orchestrator(backend_type="pytorch", batch_size=4)
    
    # Run with telemetry
    action, density = orch.compute_effective_action(
        profile, chi_values, ml_values, sigma3_values, collect_density=True
    )
    
    assert isinstance(action, torch.Tensor)
    assert isinstance(density, torch.Tensor)
    assert density.shape == rho.shape
    assert torch.isfinite(density).all()
    
    # Run without telemetry
    action_no_telemetry = orch.compute_effective_action(
        profile, chi_values, ml_values, sigma3_values, collect_density=False
    )
    
    assert torch.allclose(action, action_no_telemetry)

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
