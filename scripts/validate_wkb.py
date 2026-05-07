
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid

sys.path.append(os.path.join(os.getcwd(), 'src/python'))
from pytorch_solver import PyTorchSolver
from profiles import StepFunctionProfile

def visualize_wkb_validation():
    """
    Generates a high-quality visualization of the WKB limit validation.
    Shows phase matching, amplitude scaling, and convergence.
    """
    # Parameters
    lambd = 1.0
    F = 2.0 * np.pi  # F_cal = 1.0
    e = 1.0
    m = 1.0
    ml = 0
    sigma3 = 1
    
    chi_values = [50.0, 100.0, 200.0]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), constrained_layout=True)
    
    colors = ['r', 'g', 'b']
    
    for i, chi in enumerate(chi_values):
        # Grid: scale points with chi to maintain resolution
        n_points = int(25 * chi)
        rho = np.linspace(0.01, 1.0 * lambd, n_points)
        
        profile = StepFunctionProfile(rho, lambd=lambd, F=F, e=e)
        solver = PyTorchSolver(device='cpu')
        
        params = [{'chi': chi, 'ml': ml, 'sigma3': sigma3, 'm': m, 'e': e}]
        
        # Solve numerically
        g_num_t, _ = solver.solve_batch(params, profile)
        g_num = g_num_t[0].detach().cpu().numpy().real
        
        # Calculate Q(rho)
        B = F / (np.pi * lambd**2)
        k2_eff = chi**2 - m**2 - e * sigma3 * B + e * ml * B
        Q = k2_eff + (0.25 - ml**2) / (rho**2) - 0.25 * (e * B * rho)**2
        
        # Numerical Amplitude and Phase
        g_a = hilbert(g_num)
        g_amp_num = np.abs(g_a)
        phase_num = np.unwrap(np.angle(g_a))
        
        # WKB properties
        s_prime_wkb = np.sqrt(Q)
        g_amp_wkb = 1.0 / (2.0 * s_prime_wkb) 
        
        # Derivatives
        s_prime_num = 0.5 * np.abs(np.gradient(phase_num, rho))
        
        # Avoid edge effects
        valid = slice(n_points//20, -n_points//20)
        
        # Plot 1: Phase Derivative Comparison
        axes[0].plot(rho[valid], s_prime_num[valid], color=colors[i], alpha=0.5, label=fr"Numerical $\chi={chi}$")
        axes[0].plot(rho[valid], s_prime_wkb[valid], '--', color=colors[i], label=fr"WKB $\chi={chi}$")
        
        # Plot 2: Amplitude Comparison (G / G_wkb)
        # We plot the ratio to show it approaching 1.0
        ratio = g_amp_num / g_amp_wkb
        axes[1].plot(rho[valid], ratio[valid], color=colors[i], label=fr"$\chi={chi}$")
        
        # Plot 3: Oscillatory Match (Zoomed)
        # Select a small window to show the actual wave matching
        zoom_range = (rho > 0.4) & (rho < 0.5)
        if chi == chi_values[-1]: # Only plot for the highest chi to keep it clean
            axes[2].plot(rho[zoom_range], g_num[zoom_range], color=colors[i], label=fr"Numerical $\chi={chi}$")
            # WKB form: G ~ G_amp * sin(2*S + delta)
            s_wkb = cumulative_trapezoid(s_prime_wkb, rho, initial=0)
            # Find best fit phase delta for visualization
            best_delta = 0
            best_resid = 1e10
            for delta in np.linspace(0, 2*np.pi, 100):
                g_test = g_amp_wkb * np.sin(2*s_wkb + delta)
                resid = np.sum((g_num[zoom_range] - g_test[zoom_range])**2)
                if resid < best_resid:
                    best_resid = resid
                    best_delta = delta
            g_wkb_osc = g_amp_wkb * np.sin(2*s_wkb + best_delta)
            axes[2].plot(rho[zoom_range], g_wkb_osc[zoom_range], '--', color='k', label="WKB Fit")

    axes[0].set_title("WKB Phase Derivative Validation: $S'(\\rho) \\approx \\sqrt{Q(\\rho)}$", fontsize=14)
    axes[0].set_xlabel("$\\rho$ [L]", fontsize=12)
    axes[0].set_ylabel("$S'$ $[L^{-1}]$", fontsize=12)
    axes[0].legend(ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("WKB Amplitude Ratio: $G_{num} / G_{WKB}$ (including $\\pi$ factor)", fontsize=14)
    axes[1].set_xlabel("$\\rho$ [L]", fontsize=12)
    axes[1].set_ylabel("Ratio", fontsize=12)
    axes[1].axhline(1.0, color='k', linestyle=':', alpha=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.8, 1.2)

    axes[2].set_title("Oscillatory Convergence (Zoomed Region)", fontsize=14)
    axes[2].set_xlabel("$\\rho$ [L]", fontsize=12)
    axes[2].set_ylabel("$G(\\rho, \\rho)$", fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.savefig('results/wkb_validation_visual.png', dpi=300)
    print("Validation plot saved to results/wkb_validation_visual.png")

if __name__ == "__main__":
    visualize_wkb_validation()
