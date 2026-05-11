import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.python.field_profile_mlp import SplineProfile

def visualize_discovered_profiles():
    print("--- Visualizing Discovered Profiles ---")
    device = "cpu"
    rho_vals = torch.linspace(0.0, 10.0, 200).to(device)
    target_flux = 2.0 * np.pi * 0.4
    
    checkpoint_dir = "checkpoints"
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("fast_profile_")]
    checkpoints.sort()
    
    if not checkpoints:
        print("No fast_profile_*.pt files found.")
        return

    plt.figure(figsize=(10, 6))
    
    for cp in checkpoints:
        path = os.path.join(checkpoint_dir, cp)
        # Assuming 12 basis as per the 'fast' script
        model = SplineProfile(num_basis=12, total_flux=target_flux, rho_max=10.0)
        try:
            model.load_state_dict(torch.load(path))
            model.eval()
            B, _ = model(rho_vals)
            plt.plot(rho_vals.numpy(), B.detach().numpy(), label=cp)
        except Exception as e:
            print(f"Could not load {cp}: {e}")

    plt.title("Comparison of Discovered Stationary Magnetic Field Profiles")
    plt.xlabel("Radius (rho)")
    plt.ylabel("B(rho)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/comparison_discovered_profiles.png")
    print("Comparison plot saved to results/comparison_discovered_profiles.png")

if __name__ == "__main__":
    visualize_discovered_profiles()
