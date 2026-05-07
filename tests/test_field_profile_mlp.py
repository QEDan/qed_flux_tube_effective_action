import torch
import numpy as np
from src.python.field_profile_mlp import FluxConservingMLP

def test_field_profile_mlp():
    print("--- Testing FluxConservingMLP (Coordinate Mapping) ---")
    target_flux = 2.0 * np.pi * 0.4
    L = 2.0
    mlp = FluxConservingMLP(hidden_dim=32, num_layers=3, total_flux=target_flux, L=L)
    
    # Test Output Shapes over a wide range to see asymptotics
    rho_test = torch.linspace(0.01, 50.0, 500).reshape(-1, 1).requires_grad_(True)
    B_vals, a_phi = mlp(rho_test)
    
    # Test Flux Conservation (Identical by construction)
    # Total flux is Phi = 2*pi * lim_{rho->inf} (rho * a_phi)
    # Or integrate 2*pi*rho*B
    B_vals_np = B_vals.squeeze().detach().numpy()
    rho_np = rho_test.squeeze().detach().numpy()
    rho_weights = np.zeros_like(rho_np)
    rho_weights[1:] = rho_np[1:] - rho_np[:-1]
    integrated_flux = 2.0 * np.pi * np.sum(rho_np * B_vals_np * rho_weights)
    
    print(f"B(0): {B_vals_np[0]:.4f}")
    print(f"Asymptotic rho*A: { (rho_np[-1] * a_phi[-1].item()) * 2.0 * np.pi :.4f} (Target: {target_flux:.4f})")
    print(f"Integrated Flux: {integrated_flux:.4f} (Target: {target_flux:.4f})")
    
    # Test Differentiability
    loss = B_vals.sum()
    loss.backward()
    
    assert B_vals.shape == rho_test.shape
    # Check if asymptotic flux is very close
    asymptotic_flux = (rho_np[-1] * a_phi[-1].item()) * 2.0 * np.pi
    assert abs(asymptotic_flux - target_flux) < 0.05
    assert rho_test.grad is not None
    
    print("✅ FluxConservingMLP (Coordinate Mapping) test passed.")

if __name__ == "__main__":
    test_field_profile_mlp()

if __name__ == "__main__":
    test_field_profile_mlp()
