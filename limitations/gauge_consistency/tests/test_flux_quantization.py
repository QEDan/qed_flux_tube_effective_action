import torch
import numpy as np
import pytest

from src.python.pytorch_solver import PyTorchSolver
from src.python.profiles import StepFunctionProfile

@pytest.mark.xfail(
    reason="AB mode mapping is not gauge-covariant in the current solver (see limitations/gauge_consistency/README.md)",
    strict=False,
)
def test_flux_quantization_cancellation():
    """
    Test that the Green's function exterior integral cancels for integer flux F.
    According to symbolic_validations/derive_flux_quantization.sage, for integer F,
    the sum over angular momentum modes should exhibit Aharonov-Bohm cancellation.
    """
    lambd = 0.1
    m = 1.0
    chi = 2.0  # Oscillatory regime: chi^2 > m^2
    e = 1.0
    
    rho_val = 10.0 # Evaluated far from the origin (exterior region)
    solver = PyTorchSolver(device="cpu")
    
    # The Aharonov-Bohm cancellation occurs when F_bar = e * F / (2 * pi) is an integer.
    # So F = 2 * pi for F_bar = 1.0
    F_int = 2.0 * np.pi
    
    # Check the mode mapping: G_ml(F=1) should equal G_{ml-1}(F=0)
    # This is the essence of the AB cancellation: the set of modes is shifted.
    ml_check_range = range(-5, 6)
    
    for ml in ml_check_range:
        # G_ml(F_int)
        profile_int = StepFunctionProfile(np.array([rho_val]), lambd=lambd, F=F_int, e=e)
        res_int = solver.solve_batch([{'chi': chi + 0j, 'ml': ml, 'sigma3': 1, 'm': m, 'e': e}], profile_int)[0]
        
        # G_{ml-1}(F=0)
        profile_zero = StepFunctionProfile(np.array([rho_val]), lambd=lambd, F=0.0, e=e)
        res_zero = solver.solve_batch([{'chi': chi + 0j, 'ml': ml - 1, 'sigma3': 1, 'm': m, 'e': e}], profile_zero)[0]
        
        diff = torch.abs(res_int - res_zero).item()
        print(f"Mode ml={ml}: G_ml(F=1)={res_int.item():.4f}, G_{ml-1}(F=0)={res_zero.item():.4f}, Diff={diff:.4f}")
        
        assert diff < 1e-1, f"Mode mapping failed for ml={ml}"
    
    print("✅ Flux Quantization mode-mapping check passed.")

if __name__ == "__main__":
    test_flux_quantization_cancellation()
