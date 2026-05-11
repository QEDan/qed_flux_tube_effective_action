import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.python.field_profile_mlp import SplineProfile
from src.python.orchestrator import Orchestrator
from src.python.profiles import MLPProfile

def analyze_profile_stability(checkpoint_path):
    print(f"\n--- Analyzing Stability for {checkpoint_path} ---")
    device = "cpu"
    orchestrator = Orchestrator(device=device)
    
    rho_vals = torch.linspace(0.01, 10.0, 100).to(device)
    target_flux = 2.0 * np.pi * 0.4
    
    # Load Model - Use num_basis=12 to match Fast Discovery
    model = SplineProfile(num_basis=12, total_flux=target_flux, rho_max=10.0)
    try:
        model.load_state_dict(torch.load(checkpoint_path))
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"Skipping {checkpoint_path}: Basis size mismatch. (Expected 12, check your discovery settings)")
            return
        else:
            raise e
    model.eval()

    chi_vals = np.logspace(np.log10(0.1), np.log10(100.0), 20)
    chi_vals = [complex(c) for c in chi_vals]
    ml_vals = list(range(-20, 21))

    # Hessian calculation involves multiple calls to action_fn.
    # Since we can't easily hook into autograd's inner loop, we'll use a 
    # progress bar to monitor the checkpoint loop and a simple print for Hessian.
    def action_fn(weights):
        # Substitute weights temporarily
        original_weights = model.weights.data.clone()
        model.weights.data = weights
        
        B_vals, a_phi = model(rho_vals)
        profile = MLPProfile(rho_vals, B_vals.squeeze(), a_phi.squeeze())
        action = orchestrator.compute_effective_action(profile, chi_vals, ml_vals, [1, -1])
        
        model.weights.data = original_weights
        return action.real

    # Compute Hessian
    print("Computing Hessian (Autodiff)...")
    hessian = torch.autograd.functional.hessian(action_fn, model.weights)
    
    # Eigenvalue Analysis
    eigenvalues = torch.linalg.eigvals(hessian).real
    print(f"Eigenvalues: {eigenvalues}")
    
    stable_modes = torch.sum(eigenvalues > 0).item()
    unstable_modes = torch.sum(eigenvalues < 0).item()
    print(f"Stable Modes (>0): {stable_modes}")
    print(f"Unstable Modes (<0): {unstable_modes}")
    
    if unstable_modes == 0:
        print("Profile is a LOCAL MINIMUM (Stable Metastable State).")
    elif stable_modes == 0:
        print("Profile is a LOCAL MAXIMUM.")
    else:
        print(f"Profile is a SADDLE POINT with {unstable_modes} unstable directions.")

    # Plot Eigenvalues
    plt.figure()
    plt.bar(range(len(eigenvalues)), eigenvalues.detach().numpy())
    plt.axhline(0, color='black', lw=1)
    plt.title(f"Hessian Eigenvalues: {os.path.basename(checkpoint_path)}")
    plt.xlabel("Mode Index")
    plt.ylabel("Eigenvalue")
    plt.savefig(f"results/hessian_{os.path.basename(checkpoint_path)}.png")

if __name__ == "__main__":
    import os
    # Look for both naming conventions
    checkpoints = [f for f in os.listdir("checkpoints") if f.startswith("fast_profile_") or f.startswith("stationary_profile_")]
    
    if not checkpoints:
        print("No checkpoints found in 'checkpoints/' directory.")
    else:
        pbar_cp = tqdm(checkpoints, desc="Analyzing Checkpoints", unit="cp")
        for cp in pbar_cp:
            analyze_profile_stability(os.path.join("checkpoints", cp))
