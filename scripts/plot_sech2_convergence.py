import numpy as np
import torch
import matplotlib.pyplot as plt
from src.python.sech2_shell import Sech2ShellProfile

def plot_sech2_convergence():
    print("--- Generating Sech2-Shell Convergence Plot ---")
    
    radii = [20.0, 40.0, 80.0]
    B = 1.0
    lambd = 1.0
    
    plt.figure(figsize=(10, 6))
    
    for R in radii:
        rho = np.linspace(R - 5.0, R + 5.0, 200)
        # Shift to local coordinate x = rho - R
        x = rho - R
        profile = Sech2ShellProfile(rho, R=R, B=B, lambd=lambd)
        _, _, b_field = profile.get_arrays()
        
        plt.plot(x, b_field, label=f'R={R}')
        
    plt.title("Magnetic Field Profiles for Sech2-Shells (Local Coordinates)")
    plt.xlabel("Local Radial coordinate x = rho - R")
    plt.ylabel("Magnetic field B(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/sech2_shell_comparison.png")
    print("Plot saved to results/sech2_shell_comparison.png")

if __name__ == "__main__":
    plot_sech2_convergence()
