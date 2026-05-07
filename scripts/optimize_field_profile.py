import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from src.python.field_profile_mlp import FluxConservingMLP
from src.python.orchestrator import Orchestrator
from src.python.profiles import MLPProfile

def optimize_field_profile(num_restarts=3, steps=100, lr=1.0, checkpoint_dir="checkpoints", log_dir="runs/field_opt"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"--- Starting Efficient Flux-Conserving Optimization (L-BFGS) ---")
    print(f"TensorBoard logs directed to: {log_dir}")
    
    device = "cpu"
    orchestrator = Orchestrator(device=device)
    
    rho_vals = torch.linspace(0.01, 5.0, 50).to(device)
    target_flux = 2.0 * np.pi * 0.4
    
    results = []
    
    for i in range(num_restarts):
        print(f"\nRestart {i+1}/{num_restarts}...")
        mlp = FluxConservingMLP(hidden_dim=32, num_layers=3, total_flux=target_flux)
        optimizer = torch.optim.LBFGS(mlp.parameters(), lr=lr, max_iter=5, history_size=10)
        
        start_step = 0
        ckpt_path = os.path.join(checkpoint_dir, f"restart_{i}_flux_cons.pt")
        
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path)
            mlp.load_state_dict(ckpt['mlp_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            start_step = ckpt['step']

        history = []
        
        def get_resolution(step):
            if step < 10:
                return [complex(c) for c in np.linspace(0.1, 10.0, 10)], list(range(-5, 6))
            elif step < 20:
                return [complex(c) for c in np.linspace(0.1, 20.0, 20)], list(range(-15, 16))
            else:
                return [complex(c) for c in np.linspace(0.1, 40.0, 40)], list(range(-30, 31))

        for step in range(start_step, steps):
            if step in [10, 20]:
                print(f"--- Transitioning to higher resolution at step {step} ---")
                optimizer = torch.optim.LBFGS(mlp.parameters(), lr=lr, max_iter=5, history_size=10)
            
            chi_vals, ml_vals = get_resolution(step)
            closure_stats = {}

            def closure():
                optimizer.zero_grad()
                B_vals, a_phi = mlp(rho_vals.view(-1, 1))
                B_vals = B_vals.squeeze()
                a_phi = a_phi.squeeze()
                
                if torch.any(torch.isnan(B_vals)):
                    return torch.tensor(0.0, requires_grad=True)

                profile = MLPProfile(rho_vals, B_vals, a_phi)
                action = orchestrator.compute_effective_action(profile, chi_vals, ml_vals, [1, -1])
                
                if torch.isnan(action):
                    return torch.tensor(0.0, requires_grad=True)

                grad_B = torch.gradient(B_vals, spacing=rho_vals[1]-rho_vals[0])[0]
                reg_smooth = 0.01 * action.abs().detach() * torch.sum(grad_B**2)
                
                loss = action + reg_smooth
                loss.backward()
                
                closure_stats['action'] = action.item()
                closure_stats['loss'] = loss.item()
                closure_stats['B_vals'] = B_vals.detach().clone()
                return loss.real

            loss_val = optimizer.step(closure)
            history.append(loss_val.item())
            
            # TensorBoard logging
            writer.add_scalar(f'Restart_{i}/Loss', float(closure_stats.get('loss', 0.0).real), step)
            writer.add_scalar(f'Restart_{i}/Action', float(closure_stats.get('action', 0.0).real), step)
            
            total_norm = 0.0
            for name, p in mlp.named_parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    writer.add_histogram(f'Restart_{i}/Gradients/{name}', p.grad, step)
            total_norm = total_norm ** 0.5
            writer.add_scalar(f'Restart_{i}/Gradient_Norm', total_norm, step)

            if 'B_vals' in closure_stats:
                fig_profile = plt.figure(figsize=(6, 4))
                plt.plot(rho_vals.numpy(), closure_stats['B_vals'].numpy())
                plt.title(f'B(rho) at step {step}')
                plt.xlabel('rho')
                plt.ylabel('B')
                writer.add_figure(f'Restart_{i}/Field_Profile', fig_profile, step)
                plt.close(fig_profile)

                rho_weights = torch.ones_like(rho_vals) * (rho_vals[1]-rho_vals[0])
                curr_flux = 2.0 * np.pi * torch.sum(rho_vals * closure_stats['B_vals'] * rho_weights)
                writer.add_scalar(f'Restart_{i}/Flux', curr_flux.item(), step)
            
            if step % 5 == 0:
                print(f"Step {step}, Loss: {loss_val.item():.4e}, GradNorm: {total_norm:.4e}")
                torch.save({'mlp_state': mlp.state_dict(), 'optimizer_state': optimizer.state_dict(), 'step': step + 1}, ckpt_path)
                
        results.append({'mlp': mlp, 'history': history, 'final_action': history[-1]})
    
    writer.close()
    
    # Final visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for res in results: plt.plot(res['history'])
    plt.xlabel('Step'); plt.ylabel('Loss'); plt.title('Convergence History')
    
    plt.subplot(1, 2, 2)
    rho_plot = torch.linspace(0.0, 5.0, 100).view(-1, 1)
    for res in results:
        B_p, _ = res['mlp'](rho_plot)
        plt.plot(rho_plot.numpy(), B_p.detach().numpy(), label=f"Act: {res['final_action']:.2f}")
    plt.xlabel('rho'); plt.ylabel('B(rho)'); plt.title('Discovered Field Profiles'); plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/field_profile_optimization.png")

if __name__ == "__main__":
    optimize_field_profile()
