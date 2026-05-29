import torch
import pytest
from src.python.orchestrator import Orchestrator
from src.python.profiles import StepFunctionProfile

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
    
    orch = Orchestrator(batch_size=4)
    
    # Run integration
    action, density = orch.compute_effective_action(
        profile, chi_values, ml_values, sigma3_values, collect_density=True
    )    
    assert isinstance(action, torch.Tensor)
    assert isinstance(density, torch.Tensor)
    assert density.shape == rho.shape
    assert torch.isfinite(density).all()
    
    # Run again to verify stability
    action_v2, _ = orch.compute_effective_action(
        profile, chi_values, ml_values, sigma3_values
    )
    
    assert torch.allclose(action, action_v2)

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
