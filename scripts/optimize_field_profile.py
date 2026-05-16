"""
Optimizes the field profile parameters to minimize the effective action.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from src.python.field_profile_mlp import BasisProfile
from src.python.orchestrator import Orchestrator
from src.python.profiles import MLPProfile
from src.python.optimization_monitor import OptimizationMonitor

def optimize_field_profile(num_restarts=1, steps=200, lr=0.1, checkpoint_dir="checkpoints", log_dir="runs/field_opt_basis"):
    torch.set_num_threads(1)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"--- Starting Efficient Basis-Expansion Optimization (L-BFGS) ---")
    
    device = "cpu"
    orchestrator = Orchestrator(device=device)
    
    rho_vals = torch.linspace(0.01, 10.0, 100).to(device)
    target_flux = 2.0 * np.pi * 0.4
    
    chi_full = np.logspace(np.log10(0.1), np.log10(100.0), 100)
    chi_full = [complex(c) for c in chi_full]
    ml_full = list(range(-60, 61))
    chi_batch_size = 10 

    for i in range(num_restarts):
        print(f"\nRestart {i+1}/{num_restarts}...")
        model = BasisProfile(num_basis=15, total_flux=target_flux, rho_max=10.0)
        monitor = OptimizationMonitor(model)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20, history_size=10)
        
        history = []
        for step in range(steps):
            def closure():
                optimizer.zero_grad()
                B_vals, a_phi = model(rho_vals)
                
                # Monitor Health
                monitor.check_health()

                # Stochastic Action Estimation
                chi_indices = np.random.choice(len(chi_full), chi_batch_size, replace=False)
                chi_vals = [chi_full[idx] for idx in chi_indices]
                
                profile = MLPProfile(rho_vals, B_vals.squeeze(), a_phi.squeeze())
                action_sampled, _ = orchestrator.compute_effective_action(profile, chi_vals, ml_full, [1, -1])
                action_est = action_sampled * (len(chi_full) / chi_batch_size)
                
                # Smoothness Regularization
                grad_B = torch.gradient(B_vals.squeeze(), spacing=rho_vals[1]-rho_vals[0])[0]
                reg_smooth = 1e-4 * action_est.abs().detach() * torch.sum(grad_B**2)
                
                loss = action_est + reg_smooth
                loss.backward()
                return loss.real

            loss_val = optimizer.step(closure)
            history.append(loss_val.item())
            
            if step % 10 == 0:
                print(f"Step {step}, LossEst: {loss_val.item():.4e}")
                writer.add_scalar('Loss', loss_val.item(), step)
                
    writer.close()

if __name__ == "__main__":
    optimize_field_profile()
