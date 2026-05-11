import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.python.field_profile_mlp import SplineProfile
from src.python.orchestrator import Orchestrator
from src.python.profiles import MLPProfile
from src.python.optimization_monitor import OptimizationMonitor

def discover_stationary_profiles(num_starts=3, steps=50, lr=0.1, log_dir="runs/stationary_discovery"):
    torch.set_num_threads(1)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"--- Starting Stationary Profile Discovery (Fast Discovery Mode) ---")
    
    device = "cpu"
    orchestrator = Orchestrator(device=device)
    
    rho_vals = torch.linspace(0.01, 10.0, 100).to(device)
    dr = rho_vals[1] - rho_vals[0]
    target_flux = 2.0 * np.pi * 0.4
    
    # Fast discovery grid: fewer points, smaller cutoff
    chi_fast = np.logspace(np.log10(0.1), np.log10(20.0), 15)
    chi_fast = [complex(c) for c in chi_fast]
    ml_fast = list(range(-15, 16))

    pbar_starts = tqdm(range(num_starts), desc="Discovery Starts", unit="start")
    for start_idx in pbar_starts:
        model = SplineProfile(num_basis=12, total_flux=target_flux, rho_max=10.0)
        
        # Perturb initial weights gently
        with torch.no_grad():
            model.weights.data *= (1.0 + 0.1 * torch.randn_like(model.weights.data))
        
        monitor = OptimizationMonitor(model)
        optimizer = torch.optim.LBFGS(
            model.parameters(), 
            lr=lr, 
            max_iter=10, 
            history_size=5,
            line_search_fn='strong_wolfe'
        )
        
        pbar_steps = tqdm(range(steps), desc=f"Start {start_idx} Steps", leave=False, unit="step")
        for step in pbar_steps:
            def closure():
                optimizer.zero_grad()
                
                # Prevent extreme weights
                with torch.no_grad():
                    model.weights.data.clamp_(min=-100.0, max=100.0)

                B_vals, a_phi = model(rho_vals)
                if not torch.isfinite(B_vals).all():
                    return torch.tensor(1e15, device=device, requires_grad=True)

                # Deterministic Action Calculation
                profile = MLPProfile(rho_vals, B_vals.squeeze(), a_phi.squeeze())
                action = orchestrator.compute_effective_action(profile, chi_fast, ml_fast, [1, -1])
                
                # Check for NaNs
                if not torch.isfinite(action).all():
                    return torch.tensor(1e15, device=device, requires_grad=True)

                # Smoothness Regularization
                grad_B = torch.gradient(B_vals.squeeze(), spacing=dr.item())[0]
                reg_smooth = 1e-2 * torch.sum(grad_B**2) * dr
                
                # Scale the loss if it's too large to keep gradients manageable
                # We use a soft-log scale for the action magnitude if it's > 1e6
                action_real = action.real
                if action_real.abs() > 1e6:
                    loss = torch.sign(action_real) * torch.log10(action_real.abs()) + reg_smooth
                else:
                    loss = (action_real / 1e6) + reg_smooth
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                pbar_steps.set_postfix({
                    "Act": f"{action_real.item():.1e}", 
                    "Loss": f"{loss.item():.2f}"
                })
                
                return loss

            try:
                loss_val = optimizer.step(closure)
                if step % 5 == 0:
                    writer.add_scalar(f'Action/Start_{start_idx}', loss_val.item(), step)
            except Exception as e:
                print(f"Instability at start {start_idx}, step {step}: {e}")
                break
        
        checkpoint_path = f"checkpoints/fast_profile_{start_idx}.pt"
        torch.save(model.state_dict(), checkpoint_path)
                
    writer.close()

if __name__ == "__main__":
    discover_stationary_profiles()
