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
        # Match SplineProfile.forward exactly using softplus
        basis = model._get_basis(rho_vals)
        w = torch.nn.functional.softplus(weights) if model.positivity_constraint else weights
        B_raw = torch.matmul(basis, w).view(-1, 1)
        
        # Renormalize to conserve flux
        dr = rho_vals[1] - rho_vals[0]
        raw_flux = 2.0 * np.pi * torch.sum(B_raw.squeeze() * rho_vals * dr)
        norm_factor = (model.Phi_over_2pi * 2.0 * np.pi) / (raw_flux + 1e-15)
        B_vals = B_raw * norm_factor
        
        # A_phi integration
        rho_integrand = B_vals.squeeze() * rho_vals
        dy = 0.5 * (rho_integrand[1:] + rho_integrand[:-1]) * (rho_vals[1:] - rho_vals[:-1])
        flux_integral = torch.zeros_like(rho_vals)
        flux_integral[1:] = torch.cumsum(dy, dim=0)
        r_safe = torch.where(rho_vals == 0, torch.tensor(1e-15, device=rho_vals.device), rho_vals)
        a_phi = (flux_integral / r_safe).view(-1, 1)

        profile = MLPProfile(rho_vals, B_vals.squeeze(), a_phi.squeeze())
        action = orchestrator.compute_effective_action(profile, chi_vals, ml_vals, [1, -1])
        
        return action.real

    # Compute Gradient to check stationarity
    print("Checking Stationarity (Gradient Norm)...")
    weights = model.weights.detach().clone().requires_grad_(True)
    loss = action_fn(weights)
    loss.backward()
    grad_norm = torch.norm(weights.grad).item()
    print(f"Gradient Norm: {grad_norm:.6e}")
    
    # Compute Hessian
    print("Computing Hessian (Autodiff)...")
    hessian = torch.autograd.functional.hessian(action_fn, model.weights)
    
    # Eigenvalue Analysis - Use eigvalsh for sorted real eigenvalues of symmetric Hessian
    eigenvalues = torch.linalg.eigvalsh(hessian)
    print(f"Eigenvalues: {eigenvalues.detach().numpy()}")
    
    stable_modes = torch.sum(eigenvalues > 1e-6).item()
    unstable_modes = torch.sum(eigenvalues < -1e-6).item()
    flat_modes = torch.sum(torch.abs(eigenvalues) <= 1e-6).item()
    
    print(f"Stable Modes (>0): {stable_modes}")
    print(f"Unstable Modes (<0): {unstable_modes}")
    print(f"Flat/Zero Modes: {flat_modes}")
    
    stationary_threshold = 1e-3
    is_stationary = grad_norm < stationary_threshold

    if not is_stationary:
        print(f"WARNING: Profile is NOT a stationary point (Grad Norm {grad_norm:.4e} > {stationary_threshold:.4e}).")
        print("Stability classification below may be invalid.")

    if unstable_modes == 0 and is_stationary:
        print("RESULT: Profile is a confirmed LOCAL MINIMUM.")
    elif stable_modes == 0 and is_stationary:
        print("RESULT: Profile is a confirmed LOCAL MAXIMUM.")
    elif is_stationary:
        print(f"RESULT: Profile is a confirmed SADDLE POINT with {unstable_modes} unstable directions.")
    else:
        print("RESULT: Optimization did not converge to a stationary point.")

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
