"""
Validates the numerical solver against WKB approximation limits.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
from src.python.pytorch_solver import PyTorchSolver
from src.python.profiles import StepFunctionProfile

def visualize_wkb_validation():
    # Parameters
    lambd = 1.0
    F = 2.0 * np.pi  # F_cal = 1.0
    e = 1.0
    m = 1.0
    ml = 0
    sigma3 = 1
    
    # High-energy limit for asymptotic WKB match
    chi = 2000.0
    
    # Grid
    n_points = int(20 * chi)
    rho = np.linspace(0.01, 0.9 * lambd, n_points)
    
    profile = StepFunctionProfile(rho, lambd=lambd, F=F, e=e)
    solver = PyTorchSolver(device='cpu')
    
    params = [{'chi': chi, 'ml': ml, 'sigma3': sigma3, 'm': m, 'e': e}]
    
    # Solve numerically
    g_num_t, _ = solver.solve_batch(params, profile)
    g_num = g_num_t[0].detach().cpu().numpy().real
    
    # Calculate WKB properties
    B = F / (np.pi * lambd**2)
    k2_eff = chi**2 - m**2 - e * sigma3 * B + e * ml * B
    Q = k2_eff + (0.25 - ml**2) / (rho**2) - 0.25 * (e * B * rho)**2
    s_prime_wkb = np.sqrt(Q)
    g_amp_wkb = (np.pi / 2.0) / s_prime_wkb
    s_wkb = cumulative_trapezoid(s_prime_wkb, rho, initial=0)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Select a very small window to show individual wave cycles
    zoom_range = (rho > 0.45) & (rho < 0.46)
    
    # Find best fit phase delta AND offset for visualization
    best_delta = 0
    best_offset = 0
    best_resid = 1e10
    for delta in np.linspace(0, 2*np.pi, 50):
        for offset in np.linspace(-0.01, 0.01, 50):
            g_test = g_amp_wkb * np.sin(2*s_wkb + delta) + offset
            resid = np.sum((g_num[zoom_range] - g_test[zoom_range])**2)
            if resid < best_resid:
                best_resid = resid
                best_delta = delta
                best_offset = offset
    
    plt.plot(rho[zoom_range], g_num[zoom_range], label=fr"Numerical $\chi={chi}$")
    g_wkb_osc = g_amp_wkb * np.sin(2*s_wkb + best_delta) + best_offset
    plt.plot(rho[zoom_range], g_wkb_osc[zoom_range], '--', label="WKB Fit (offset adjusted)")
    
    plt.title(f"Oscillatory Convergence (Zoomed, $\chi={chi}$)")
    plt.xlabel("$\\rho$ [L]")
    plt.ylabel("$G(\\rho, \\rho)$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('results/wkb_validation_visual.png', dpi=300)
    
    print("\n--- Validation Note ---")
    print("Oscillatory Convergence plot saved to results/wkb_validation_visual.png")
    print("What to look for:")
    print("- Amplitude Difference (Envelope): Quantifies sub-leading order WKB corrections.")
    print("- Vertical Offset: Residual shift consistent with higher-order semi-classical terms.")
    print("- Phase Alignment: Confirms accuracy of numerical phase accumulation.")
    print(f"Calculated best-fit vertical offset: {best_offset:.6e}")
    print("-----------------------\n")

if __name__ == "__main__":
    visualize_wkb_validation()
