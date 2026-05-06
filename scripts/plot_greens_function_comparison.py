import numpy as np
import torch
import matplotlib.pyplot as plt
from src.python.pytorch_solver import PyTorchSolver
from src.python.delta_shell import DeltaFunctionShellProfile

def plot_greens_function_agreement():
    print("--- Generating Green's Function Agreement Plot ---")
    
    # Grid
    rho = np.linspace(0.01, 15.0, 500)
    R = 5.0
    F = 1.0
    profile = DeltaFunctionShellProfile(rho, R=R, F=F)
    
    # Solve numerically
    solver = PyTorchSolver(device='cpu')
    
    # We choose a single chi, ml=0, sigma3=1 mode for comparison
    params = {'chi': complex(0.5, 0), 'ml': 0, 'sigma3': 1, 'm': 1.0, 'e': 1.0}
    # Solve for background (vacuum) and field
    g_num, _ = solver.solve_batch([params], profile)
    g_vac, _ = solver.solve_batch([params], DeltaFunctionShellProfile(rho, R=100.0, F=0.0))
    
    # Numerical G(rho, rho) is G(rho) along diagonal
    g_num_diag = g_num[0].detach().cpu().numpy()
    g_vac_diag = g_vac[0].detach().cpu().numpy()
    
    # Calculate residuals and imaginary parts
    residual = g_num_diag - g_vac_diag
    g_num_imag = g_num_diag.imag
    
    # Create 2-panel plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    axes[0].plot(rho, g_num_diag.real, label='Numerical G_num')
    axes[0].plot(rho, g_vac_diag.real, linestyle='--', label='Vacuum G_vac')
    axes[0].axvline(x=R, color='r', linestyle=':', label='Shell Radius R=5')
    axes[0].set_title("Numerical vs Analytic Green's Function (Diagonal)")
    axes[0].set_ylabel("Real Part")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(rho, residual.real, label='Residual (Re)')
    axes[1].set_title("Residuals (Numerical - Vacuum)")
    axes[1].set_xlabel("Radial Coordinate rho")
    axes[1].set_ylabel("Real Residual")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("results/greens_function_comparison.png")
    print("Enhanced 2-panel plot saved to results/greens_function_comparison.png")

if __name__ == "__main__":
    plot_greens_function_agreement()
