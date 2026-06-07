import torch
import torch.optim as optim
from src.python.orchestrator import Orchestrator
from src.python.profiles import StepFunctionProfile
from src.python import constants

def test_total_action_optimization():
    print("Starting Unified Framework Verification (Classical + 1-loop)...")
    device = 'cpu'
    
    # 1. Differentiable Parameters
    # Initial guess for lambda
    lambd = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    F_cal = torch.tensor(1.0, dtype=torch.float64) # Fixed flux
    
    # 2. Setup Orchestrator
    orch = Orchestrator(strategy="numerical", device=device)
    
    # 3. Optimization Loop
    optimizer = optim.Adam([lambd], lr=0.1)
    
    # Small grids for fast verification
    n_rho = 30
    rho = torch.linspace(0.01, 10.0, n_rho, dtype=torch.float64)
    Q_vals = torch.linspace(0.5, 5.0, 5, dtype=torch.float64)
    ml_values = [0, 1]
    sigma3_values = [1, -1]
    e = constants.ELECTRON_CHARGE
    m = constants.ELECTRON_MASS
    F_phys = (constants.TWO_PI * F_cal) / e

    print(f"Initial lambda: {lambd.item():.4f}")
    
    for step in range(3):
        optimizer.zero_grad()
        
        # Profile depends on lambd
        profile = StepFunctionProfile(rho, lambd, F_phys, e=e, smooth_width=0.5)
        
        # Total action = S_cl + Gamma_1loop
        total_action = orch.compute_total_action(
            profile,
            Q_vals,
            ml_values,
            sigma3_values,
            m=m,
            e=e,
            lcf_threshold=None
        )
        
        total_action.backward()
        optimizer.step()
        
        # Ensure lambda stays positive
        with torch.no_grad():
            lambd.clamp_(min=0.1)
            
        print(f"Step {step+1}: Action = {total_action.item():.6e}, Lambda = {lambd.item():.4f}, Grad = {lambd.grad.item():.6e}")

    assert lambd.grad is not None
    assert not torch.isnan(total_action)
    print("\nUnified Framework Verified successfully!")

if __name__ == "__main__":
    test_total_action_optimization()
