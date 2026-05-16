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

def discover_stationary_profiles(num_starts=10, steps=50, lr=0.1, log_dir="runs/stationary_discovery"):
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
        
        # Simulated Annealing Parameters
        T_start = 1.0
        T_end = 0.01
        current_loss = float('inf')
        best_loss = float('inf')
        
        pbar_steps = tqdm(range(steps), desc=f"Start {start_idx} Annealing", leave=False, unit="step")
        for step in pbar_steps:
            T = T_start * (T_end / T_start) ** (step / steps)
            
            # 1. Perturb weights
            with torch.no_grad():
                old_weights = model.weights.data.clone()
                perturbation = torch.randn_like(model.weights.data) * T * 0.5
                model.weights.data += perturbation
                model.weights.data.clamp_(min=-50.0, max=50.0)

            # 2. Local gradient descent to "settle" into the nearest basin
            optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
            for _ in range(5): # Small number of refinement steps
                optimizer_adam.zero_grad()
                B_vals, a_phi = model(rho_vals)
                profile = MLPProfile(rho_vals, B_vals.squeeze(), a_phi.squeeze())
                action, _ = orchestrator.compute_effective_action(profile, chi_fast, ml_fast, [1, -1])
                grad_B = torch.gradient(B_vals.squeeze(), spacing=dr.item())[0]
                reg_smooth = 1e-2 * torch.sum(grad_B**2) * dr
                loss = action.real + reg_smooth
                loss.backward()
                optimizer_adam.step()
            
            # 3. Evaluate and Accept/Reject (Metropolis Criterion)
            with torch.no_grad():
                B_vals, a_phi = model(rho_vals)
                profile = MLPProfile(rho_vals, B_vals.squeeze(), a_phi.squeeze())
                action, _ = orchestrator.compute_effective_action(profile, chi_fast, ml_fast, [1, -1])
                grad_B = torch.gradient(B_vals.squeeze(), spacing=dr.item())[0]
                reg_smooth = 1e-2 * torch.sum(grad_B**2) * dr
                new_loss = (action.real + reg_smooth).item()
                
                delta_e = new_loss - current_loss
                if delta_e < 0 or np.random.rand() < np.exp(-delta_e / (T + 1e-10)):
                    current_loss = new_loss
                    if new_loss < best_loss:
                        best_loss = new_loss
                        best_state = model.state_dict().copy()
                else:
                    model.weights.data.copy_(old_weights)

            pbar_steps.set_postfix({
                "T": f"{T:.2f}",
                "Loss": f"{current_loss:.2f}",
                "Best": f"{best_loss:.2f}"
            })

        # Load best found state and do final high-precision refinement
        model.load_state_dict(best_state)
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=0.01, max_iter=20, line_search_fn='strong_wolfe'
        )
        def closure():
            optimizer_lbfgs.zero_grad()
            B_vals, a_phi = model(rho_vals)
            profile = MLPProfile(rho_vals, B_vals.squeeze(), a_phi.squeeze())
            action, _ = orchestrator.compute_effective_action(profile, chi_fast, ml_fast, [1, -1])
            grad_B = torch.gradient(B_vals.squeeze(), spacing=dr.item())[0]
            reg_smooth = 1e-2 * torch.sum(grad_B**2) * dr
            loss = action.real + reg_smooth
            loss.backward()
            return loss

        print(f"--- Start {start_idx} Final LBFGS Polish ---")
        try:
            optimizer_lbfgs.step(closure)
        except Exception as e:
            print(f"LBFGS refinement failed: {e}")
        
        checkpoint_path = f"checkpoints/fast_profile_{start_idx}.pt"
        torch.save(model.state_dict(), checkpoint_path)
                
    writer.close()

if __name__ == "__main__":
    discover_stationary_profiles()
