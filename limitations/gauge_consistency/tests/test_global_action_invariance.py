import torch
import numpy as np
from src.python.pytorch_solver import PyTorchSolver
from src.python.sech2_shell import Sech2ShellProfile

def test_global_effective_action_invariance():
    """
    Test that the global effective action (sum over modes) is invariant 
    under integer flux shifts, confirming the AB topological invariance.
    """
    m = 1.0
    chi = 2.0
    e = 1.0
    lambd = 0.5
    
    # B = 1.0 corresponds to a flux F_bar = 1.0 (integer)
    B_int = 1.0 / (2.0 * np.pi * lambd**2 * np.log(2.0))
    B_zero = 0.0
    
    solver = PyTorchSolver(device="cpu")
    ml_range = range(-10, 11)
    
    R = 2.0 * lambd

    def get_global_action(B_val):
        profile = Sech2ShellProfile(np.array([10.0]), R=R, B=B_val, lambd=lambd, e=e)
        params = [{'chi': chi + 0j, 'ml': ml, 'sigma3': 1, 'm': m, 'e': e} for ml in ml_range]
        # We compute the sum of Green's functions, which is proportional to the effective action
        results, _ = solver.solve_batch(params, profile)
        return torch.sum(results).item()

    action_int = get_global_action(B_int)
    action_zero = get_global_action(B_zero)
    
    print(f"Global Action (F_bar=1.0): {action_int:.4f}")
    print(f"Global Action (F_bar=0.0): {action_zero:.4f}")
    
    # The Aharonov-Bohm effect is topological; the vacuum energy should be invariant
    # for integer flux shifts in a gauge-consistent theory.
    # Note: We expect convergence within a tolerance for this numerical sum.
    assert np.abs(action_int - action_zero) < 1e-1, "Global effective action is not invariant under flux shift."
    
    print("✅ Global Effective Action invariance check passed.")

if __name__ == "__main__":
    test_global_effective_action_invariance()
