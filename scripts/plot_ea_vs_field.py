"""
Plots the effective action as a function of the magnetic field strength.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import Sech2Profile

def plot_ea_vs_field():
    # Setup
    rho = torch.linspace(0.01, 10.0, 500, dtype=torch.float64)
    lambd = 1.0
    chi_values = [1.0, 2.0, 5.0]
    ml_values = [0, 1]
    sigma3_values = [1, -1]
    m = 1.0
    
    B_range = np.linspace(0.01, 0.5, 10)
    actions = []
    
    orc = Orchestrator(device="cpu")
    
    print("Computing Effective Action vs Field Strength...")
    for B in B_range:
        profile = Sech2Profile(rho, B=B, lambd=lambd)
        action, _ = orc.compute_effective_action(profile, chi_values, ml_values, sigma3_values, m=m)
        actions.append(action.real.item())
        print(f"B={B:.2f}, S_renorm={actions[-1]:.4e}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(B_range, actions, marker='o', label=r"Numerical $S_{renorm}$")
    plt.title(r"QED Effective Action vs Magnetic Field Strength $B$")
    plt.xlabel(r"Field Strength $B$")
    plt.ylabel(r"Renormalized Effective Action $S_{renorm}$")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/ea_vs_field.png")
    print("Plot saved to results/ea_vs_field.png")

if __name__ == "__main__":
    plot_ea_vs_field()
