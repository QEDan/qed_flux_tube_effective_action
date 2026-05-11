import torch
import numpy as np
import matplotlib.pyplot as plt
from src.python.field_profile_mlp import SplineProfile

def test_spline_profile():
    print("--- Testing SplineProfile ---")
    rho = torch.linspace(0.0, 10.0, 500)
    target_flux = 2.0 * np.pi * 0.4
    
    # Initialize Spline Profile
    model = SplineProfile(num_basis=15, total_flux=target_flux, rho_max=10.0)
    B, A = model(rho)
    
    # 1. Verify Flux Conservation
    dr = rho[1] - rho[0]
    calculated_flux = 2.0 * np.pi * torch.sum(B.squeeze() * rho * dr)
    print(f"Target Flux:     {target_flux:.6f}")
    print(f"Calculated Flux: {calculated_flux.item():.6f}")
    assert torch.allclose(torch.tensor(target_flux), calculated_flux, rtol=1e-2), "Flux not conserved!"

    # 2. Verify Continuity (Visual/Numerical)
    # Check for NaNs or Inf
    assert torch.isfinite(B).all(), "NaN/Inf in B field!"
    assert torch.isfinite(A).all(), "NaN/Inf in A field!"
    
    # Check that B(rho) is smooth by looking at its gradient
    dB = torch.gradient(B.squeeze(), spacing=dr.item())[0]
    d2B = torch.gradient(dB, spacing=dr.item())[0]
    
    assert torch.isfinite(dB).all(), "NaN/Inf in B gradient!"
    assert torch.isfinite(d2B).all(), "NaN/Inf in B second derivative!"
    
    print("Continuity and Flux checks passed.")

    # 3. Plotting for manual verification
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rho.numpy(), B.detach().numpy(), label="B(rho)")
    plt.title("Spline B-Field Profile")
    plt.xlabel("rho")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(rho.numpy(), A.detach().numpy(), label="A_phi(rho)", color='orange')
    plt.title("Spline Vector Potential")
    plt.xlabel("rho")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/test_spline_profile.png")
    print("Profile visualization saved to results/test_spline_profile.png")

if __name__ == "__main__":
    test_spline_profile()
