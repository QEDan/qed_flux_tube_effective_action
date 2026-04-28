import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import StepFunctionProfile
from analytic import get_interior_solutions, get_analytic_wronskian
from visualization import plot_profile_comparison

def compare_analytic_vs_numerical():
    # Parameters
    lambd = 1.0
    F = 2.0 * np.pi * 1.0
    m = 1.0
    chi = 1.5 + 0.5j
    ml = 1
    sigma3 = 1
    
    # Grid
    rho = np.linspace(0.1, lambd - 0.05, 100)
    profile = StepFunctionProfile(rho, lambd=lambd, F=F)
    
    # Numerical results
    orc = Orchestrator(backend_type="pytorch", device="cpu")
    params = [{'chi': chi, 'ml': ml, 'sigma3': sigma3, 'm': m, 'e': 1.0}]
    results_num, _ = orc.backend.solve_batch(params, profile)
    results_num = results_num[0].detach().numpy()
    
    # Analytic results
    u0_ana, uinf_ana = get_interior_solutions(rho, chi, ml, sigma3, m, lambd, F)
    W0_ana = get_analytic_wronskian(chi, ml, sigma3, m, lambd, F)
    results_ana = (u0_ana * uinf_ana) / W0_ana
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(rho, results_num.real, label="Numerical G", linestyle='--')
    plt.plot(rho, results_ana.real, label="Analytic G")
    plt.title("Comparison: Numerical vs Analytic Green's Function (Real Part)")
    plt.xlabel("Radial coordinate rho")
    plt.ylabel("G(rho, rho)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/analytic_vs_numerical.png")
    print("Plot saved as results/analytic_vs_numerical.png")

if __name__ == "__main__":
    compare_analytic_vs_numerical()
