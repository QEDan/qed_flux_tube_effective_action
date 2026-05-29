import numpy as np
import torch
from src.python.orchestrator import Orchestrator
from src.python.orchestrator import generate_params_grid
from src.python.profiles import StepFunctionProfile

def test_pytorch_backend():
    """
    Validation of PyTorch backend implementation.
    """
    rho = np.linspace(0.01, 5.0, 500)
    profile = StepFunctionProfile(rho, lambd=1.0, F=2*np.pi)
    
    # Setup orchestrator
    orc = Orchestrator(device="cpu")
    
    # Generate parameters grid
    chi_values = [1.1, 2.1]
    ml_values = [0, 1]
    sigma3_values = [1, -1]
    params_grid = generate_params_grid(chi_values, ml_values, sigma3_values)
    
    results, w0 = orc.backend.solve_batch(params_grid, profile)
    
    if isinstance(results, torch.Tensor):
        results_np = results.detach().cpu().numpy()
    else:
        results_np = results
    
    assert results.shape == (len(params_grid), len(rho))
    assert not np.any(np.isnan(results_np))
    # Green's function should be non-zero
    assert np.max(np.abs(results_np)) > 0
    print("✅ PyTorch backend test passed!")
