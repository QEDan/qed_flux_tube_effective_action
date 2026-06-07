import torch
import numpy as np
from src.python.orchestrator import Orchestrator
from src.python.profiles import StepFunctionProfile
from src.python import constants

def test_effective_action_gradient():
    print("Starting End-to-End Autodiff Verification...")
    device = 'cpu'
    
    # 1. Define differentiable parameters
    # Flux and Lambda
    F_cal = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    lambd = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    
    # 2. Setup grid and profile
    n_rho = 20
    rho = torch.linspace(0.01, 10.0, n_rho, dtype=torch.float64)
    
    # Use smoothed step profile to ensure potential is differentiable w.r.t lambd
    e = constants.ELECTRON_CHARGE
    F_phys = (constants.TWO_PI * F_cal) / e
    
    profile = StepFunctionProfile(rho, lambd, F_phys, e=e, smooth_width=0.5)
    
    # 3. Setup Orchestrator (Numerical strategy)
    orch = Orchestrator(strategy="numerical", device=device, batch_size=1024)
    
    # Small spectral grid for fast test
    n_Q = 5
    Q_vals = torch.linspace(0.5, 5.0, n_Q, dtype=torch.float64)
    ml_values = [0, 1]
    sigma3_values = [1, -1]
    m = constants.ELECTRON_MASS

    # 4. Compute Effective Action
    print("Computing effective action...")
    action, _ = orch.compute_effective_action(
        profile,
        Q_vals,
        ml_values,
        sigma3_values,
        m=m,
        e=e,
        lcf_threshold=None # Force numerical for all Q
    )
    
    print(f"Action: {action.item():.6e}")
    
    # 5. Compute Gradients
    print("Computing gradients...")
    action.backward()
    
    print(f"d(Action)/d(F_cal): {F_cal.grad.item():.6e}")
    print(f"d(Action)/d(lambd): {lambd.grad.item():.6e}")
    
    assert F_cal.grad is not None
    assert lambd.grad is not None
    assert not torch.isnan(F_cal.grad)
    assert not torch.isnan(lambd.grad)
    
    print("End-to-End Autodiff Verified successfully!")

if __name__ == "__main__":
    test_effective_action_gradient()
