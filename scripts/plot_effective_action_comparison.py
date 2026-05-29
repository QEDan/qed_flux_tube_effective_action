"""
Compares numerical, analytic, and LCF effective actions for the step-function profile.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.python.orchestrator import Orchestrator
from src.python.profiles import StepFunctionProfile, FieldProfile
from src.python.locally_constant_field import const_field_heisenberg_euler_lagrangian as heisenberg_euler_lagrangian, get_full_analytic_solution

def plot_effective_action_comparison():
    print("--- Generating Effective Action Comparison Plot ---")
    
    device = 'cpu'
    rho = torch.linspace(0.01, 5.0, 100, device=device)
    rho_np = rho.numpy()
    lambd = 1.0
    F = 1.0
    profile = StepFunctionProfile(rho, lambd=lambd, F=F)
    orc = Orchestrator(device=device)

    # Parameters sweep: Varying magnetic field flux F
    F_values = np.linspace(0.1, 2.0, 10)
    
    # chi, ml, sigma3 grid (small for speed, but consistent)
    chi_values = (np.linspace(1.1, 10.0, 10) + 0.1j).tolist()
    ml_values = list(range(-15, 16))
    sigma3_values = [1, -1]
    m = 1.0

    actions_num = []
    actions_ana = []
    actions_lcf = []

    for f_val in F_values:
        print(f"Processing F={f_val:.2f}...")
        p = StepFunctionProfile(rho, lambd=lambd, F=f_val)
        
        # 1. Numerical Action
        action_n, _ = orc.compute_effective_action(p, chi_values, ml_values, sigma3_values, m=m)
        actions_num.append(action_n.item().real)
        
        # 2. Analytic Action (using get_full_analytic_solution)
        # We simulate the orchestrator loop using analytic solutions
        mode_sums_ana = np.zeros((len(chi_values), len(rho_np)), dtype=complex)
        for i, chi in enumerate(chi_values):
            # Sum over ml, s3
            for ml in ml_values:
                for s3 in sigma3_values:
                    # Whittaker params change with s3
                    # get_full_analytic_solution internally calls get_step_function_params
                    # which accounts for s3 correctly.
                    g_ana = get_full_analytic_solution(rho_np, chi, ml, s3, m, lambd, f_val)
                    # Vacuum solution
                    g_vac = get_full_analytic_solution(rho_np, chi, ml, s3, m, lambd, 0.0)
                    mode_sums_ana[i] += (g_ana - g_vac)

        # Spectral and Spatial integration for Analytic
        L_eff_ana = np.zeros_like(rho_np, dtype=complex)
        norm_factor = 1.0 / (16.0 * np.pi**4)
        
        # Weights
        chi_real = np.array([c.real for c in chi_values])
        chi_weights = np.zeros_like(chi_real)
        chi_weights[1:-1] = (chi_real[2:] - chi_real[:-2]) / 2.0
        chi_weights[0] = (chi_real[1] - chi_real[0]) / 2.0
        chi_weights[-1] = (chi_real[-1] - chi_real[-2]) / 2.0

        uv_coeff_global = orc.renormalizer.get_b2_term(p, rho)
        mean_uv = torch.mean(uv_coeff_global).item()

        for i, chi in enumerate(chi_real):
            num_uv = mean_uv / (chi**4)
            local_renorm_sum = (mode_sums_ana[i] / rho_np) + num_uv
            L_eff_ana += chi**3 * local_renorm_sum * chi_weights[i] * norm_factor

        rho_w = np.zeros_like(rho_np)
        rho_w[1:-1] = (rho_np[2:] - rho_np[:-2]) / 2.0
        rho_w[0] = (rho_np[1] - rho_np[0]) / 2.0
        rho_w[-1] = (rho_np[-1] - rho_np[-2]) / 2.0
        
        action_a = -1.0 * np.sum(L_eff_ana * rho_np * rho_w)
        actions_ana.append(action_a.real)

        # 3. LCF Action
        B_in = f_val / (np.pi * lambd**2)
        L_HE_in = heisenberg_euler_lagrangian(B_in)
        # Action_LCF = Volume_in * L_HE_in
        # In our Action convention: -Int rho drho L_eff
        # Inside volume: 0.5 * lambda^2 = 0.5
        action_l = -0.5 * (lambd**2) * L_HE_in
        actions_lcf.append(action_l)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(F_values, actions_num, 'o-', label="Numerical Solver")
    plt.plot(F_values, actions_ana, 's--', label="Analytic (Whittaker)")
    plt.plot(F_values, actions_lcf, 'x:', label="LCF Approximation")
    plt.xlabel("Magnetic Flux F")
    plt.ylabel("Effective Action (per unit time/length / 2pi?)")
    plt.title("Effective Action Comparison: Step-Function Profile")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/effective_action_comparison.png")
    print("Plot saved to results/effective_action_comparison.png")

if __name__ == "__main__":
    plot_effective_action_comparison()
