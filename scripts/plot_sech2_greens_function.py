"""
Generates Green's function comparison plots specifically for Sech2-shell configurations.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from src.python.pytorch_solver import PyTorchSolver
from src.python.sech2_shell import Sech2ShellProfile

def plot_sech2_greens_function_agreement():
    print("--- Generating Sech2-Shell Green's Function Comparison Plot ---")
    
    # Grid: Extend range to see long-distance behavior (up to rho=1000)
    R = 20.0
    lambd = 1.0
    rho = np.logspace(0, 3, 500) # Range from 1 to 1000
    profile = Sech2ShellProfile(rho, R=R, B=1.0, lambd=lambd)
    
    # Numerical solver
    solver = PyTorchSolver(device='cpu')
    params = {'chi': complex(0.5, 0), 'ml': 0, 'sigma3': 1, 'm': 1.0, 'e': 1.0}
    
    # Solve numerically
    g_num, _ = solver.solve_batch([params], profile)
    g_num_diag = g_num[0].detach().cpu().numpy()
    
    # Analytic Baseline (B=0): Use Vacuum Green's Function G^0
    g_vac, _ = solver.solve_batch([params], Sech2ShellProfile(rho, R=1000.0, B=0.0, lambd=lambd))
    g_vac_diag = g_vac[0].detach().cpu().numpy()
    
    # Residuals
    residual = np.abs(g_num_diag.real - g_vac_diag.real)
    
    # 3-panel plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    axes[0].plot(rho, g_num_diag.real, label='Numerical Real')
    axes[0].plot(rho, g_vac_diag.real, linestyle='--', label='Field-Free Background (B=0)')
    axes[0].axvline(x=R, color='r', linestyle=':', label='Shell Center R=20')
    axes[0].set_title("Numerical vs Field-Free Green's Function (Sech2-Shell)")
    axes[0].set_ylabel("Real Part")
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(rho, g_num_diag.imag, label='Numerical Imaginary', color='green')
    axes[1].plot(rho, g_vac_diag.imag, linestyle='--', label='Vacuum Imaginary', color='grey')
    axes[1].axvline(x=R, color='r', linestyle=':')
    axes[1].set_title("Imaginary Part Comparison")
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].semilogy(rho, residual, label='Residual Magnitude |Num - Background|', color='red')
    axes[2].axvline(x=R, color='r', linestyle=':')
    axes[2].set_title("Residual Magnitude (Difference from B=0)")
    axes[2].set_xlabel("Radial Coordinate rho (log scale)")
    axes[2].set_xscale('log')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("results/sech2_greens_function_comparison.png")
    print("Plot saved to results/sech2_greens_function_comparison.png")

if __name__ == "__main__":
    plot_sech2_greens_function_agreement()
