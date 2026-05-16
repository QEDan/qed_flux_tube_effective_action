"""
Refines the identified metastable profiles to improve numerical precision.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from src.python.field_profile_mlp import SplineProfile
from src.python.orchestrator import Orchestrator
from src.python.profiles import MLPProfile
from src.python.optimization_monitor import OptimizationMonitor

def refine_metastable_profile(checkpoint_path="checkpoints/fast_profile_1.pt", num_basis=20, steps=100, lr=0.2):
    torch.set_num_threads(1)
    print(f"--- Refining Profile: {checkpoint_path} with High Resolution ---")
    device = "cpu"
    # Larger batch size for high-res evaluations to reduce overhead
    orchestrator = Orchestrator(device=device, batch_size=2048)
    
    rho_vals = torch.linspace(0.01, 10.0, 150).to(device).to(torch.float64)
    dr = rho_vals[1] - rho_vals[0]
    target_flux = 2.0 * np.pi * 0.4
    
    # High-resolution evaluation grid
    chi_high = np.logspace(np.log10(0.1), np.log10(100.0), 60)
    chi_high = [complex(c) for c in chi_high]
    ml_high = list(range(-50, 51))

    # 1. Initialize refined model
    model = SplineProfile(num_basis=num_basis, total_flux=target_flux, rho_max=10.0)
    
    # 2. Load and interpolate weights from fast_profile_1
    if os.path.exists(checkpoint_path):
        old_state = torch.load(checkpoint_path)
        old_weights = old_state['weights']
        # Simple linear interpolation to go from 12 to 20 basis weights
        old_idx = np.linspace(0, 1, len(old_weights))
        new_idx = np.linspace(0, 1, num_basis)
        new_weights = np.interp(new_idx, old_idx, old_weights.numpy())
        model.weights.data = torch.from_numpy(new_weights).to(torch.float64)
        print(f"Interpolated weights from {len(old_weights)} to {num_basis} basis functions.")
    else:
        print(f"Warning: {checkpoint_path} not found. Starting from scratch.")

    monitor = OptimizationMonitor(model)
    optimizer = torch.optim.LBFGS(
        model.parameters(), 
        lr=lr, 
        max_iter=20, 
        history_size=10,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-8,
        tolerance_change=1e-10
    )
    
    pbar_steps = tqdm(range(steps), desc="Refining Steps", unit="step")
    for step in pbar_steps:
        def closure():
            optimizer.zero_grad()
            
            with torch.no_grad():
                model.weights.data.clamp_(min=-50.0, max=50.0)

            B_vals, a_phi = model(rho_vals)
            profile = MLPProfile(rho_vals, B_vals.squeeze(), a_phi.squeeze())
            
            # High-res action calculation
            action, _ = orchestrator.compute_effective_action(profile, chi_high, ml_high, [1, -1])
            
            # Reduced smoothness reg for refinement (let the action drive the shape)
            grad_B = torch.gradient(B_vals.squeeze(), spacing=dr.item())[0]
            reg_smooth = 1e-4 * torch.sum(grad_B**2) * dr
            
            # Log-scaling for stability if needed, but here we expect to be near the minimum
            action_real = action.real
            if action_real.abs() > 1e6:
                loss = torch.sign(action_real) * torch.log10(action_real.abs()) + reg_smooth
            else:
                loss = (action_real / 1e6) + reg_smooth
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            pbar_steps.set_postfix({"Act": f"{action_real.item():.2e}", "Loss": f"{loss.item():.4f}"})
            return loss

        try:
            optimizer.step(closure)
        except Exception as e:
            print(f"Refinement interrupted: {e}")
            break
            
    # Save refined profile
    os.makedirs("checkpoints", exist_ok=True)
    refined_path = "checkpoints/refined_metastable_profile.pt"
    torch.save(model.state_dict(), refined_path)
    print(f"Refined profile saved to {refined_path}")

if __name__ == "__main__":
    refine_metastable_profile()
