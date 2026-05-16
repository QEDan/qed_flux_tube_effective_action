import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src/python'))

import torch
import numpy as np
from orchestrator import Orchestrator
from profiles import FieldProfile

class OptimizableProfile(FieldProfile):
    """
    A field profile where Aphi is a simple function with an optimizable parameter.
    Aphi(rho) = A0 * rho * exp(-rho^2 / lambda^2)
    """
    def __init__(self, rho, A0, lambd=1.0):
        super().__init__(rho)
        self.A0 = torch.tensor(A0, dtype=torch.float64, requires_grad=True)
        self.lambd = lambd
        self.update()
        
    def update(self):
        # Aphi = A0 * rho * exp(-rho^2 / lambda^2)
        # da_phi = A0 * exp(-rho^2/lambda^2) * (1 - 2*rho^2/lambda^2)
        exp_factor = torch.exp(-(self.rho**2) / (self.lambd**2))
        self.a_phi = self.A0 * self.rho * exp_factor
        self.da_phi = self.A0 * exp_factor * (1.0 - 2.0 * (self.rho**2) / (self.lambd**2))

def test_autodiff_action():
    print("Testing Auto-Diff of Effective Action...")
    
    rho = torch.linspace(0.1, 5.0, 100, dtype=torch.float64)
    profile = OptimizableProfile(rho, A0=0.1)
    
    orc = Orchestrator(device="cpu")
    
    chi_values = [1.0, 2.0]
    ml_values = [0, 1]
    sigma3_values = [1, -1]
    
    # Compute action
    action, _ = orc.compute_effective_action(profile, chi_values, ml_values, sigma3_values)
    print(f"Action value: {action.item()}")
    
    # Compute gradient with respect to A0
    # Since action is complex, we take the real part or absolute value to minimize
    loss = action.abs()
    loss.backward()
    
    grad = profile.A0.grad
    print(f"Gradient wrt A0: {grad}")
    
    assert grad is not None
    assert not torch.isnan(grad)
    print("✅ Auto-Diff successful!")

if __name__ == "__main__":
    test_autodiff_action()
