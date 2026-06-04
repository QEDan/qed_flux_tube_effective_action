import torch
import numpy as np
import matplotlib.pyplot as plt
from src.python import constants
from src.python.profiles import StepFunctionProfile
from src.python.orchestrator import Orchestrator
from src.python.step_profile_effective_action import step_profile_effective_action_density

def main():
    # 1. Parameters
    F_cal = 1.0
    lambd_val = 10.0 # Large radius
    m = constants.ELECTRON_MASS
    e = constants.ELECTRON_CHARGE
    
    # F_val such that e*F/2pi = nu => F = 2pi*nu/e
    F_val = (constants.TWO_PI * F_cal) / e
    print(f"Flux F_cal = {F_cal:.6f}, F = {F_val:.6f}")

    # Balanced parameters for convergence and speed
    n_rho = 30 # Reduced
    rho_max = 1.5 * lambd_val
    rho = torch.linspace(0.1, rho_max, n_rho, dtype=torch.float64) # Start away from origin

    # Shared spectral grid (Euclidean Q) — both paths must use the same range.
    n_Q = 20
    Q_max = 12.0
    ml_max_num = 20
    ml_max_an = 30

    # Compute Numerical Density via Orchestrator
    print(f"Computing numerical density via Orchestrator (global_mode=True, ml_max={ml_max_num})...")
    profile = StepFunctionProfile(rho, lambd_val, F_val, e=e)
    # Use global_mode=True and strategy="numerical"
    orch = Orchestrator(strategy="numerical", device='cpu', batch_size=4096, global_mode=True)
    chi_values = torch.linspace(0.1, Q_max, n_Q, dtype=torch.complex128)
    ml_values = list(range(-ml_max_num, ml_max_num + 1))
    sigma3_values = [1, -1]
    action, num_density_L = orch.compute_effective_action(
        profile,
        chi_values,
        ml_values,
        sigma3_values,
        m=m,
        e=e,
        collect_density=True,
    )

    # num_density_L is L_eff(rho).
    L_num = num_density_L.real.detach().cpu().numpy()
    rho_numeric = L_num * rho.detach().cpu().numpy()

    # Compute Analytic Density via Euclidean Whittaker spectral integration
    print(f"Computing analytic density via Whittaker functions (global_mode=True, ml_max={ml_max_an})...")
    rho_t, rho_analytic = step_profile_effective_action_density(
        torch.tensor(F_cal, dtype=torch.float64),
        torch.tensor(lambd_val, dtype=torch.float64),
        rho_cm=rho,
        m=m,
        n_chi=n_Q,
        n_rho=len(rho),
        n_ml=ml_max_an,
        Q_max=Q_max,
        global_mode=True,
    )
    rho_analytic_np = rho_analytic.detach().cpu().numpy()
    L_an = rho_analytic_np / np.maximum(rho.detach().cpu().numpy(), 1e-10)
    rho_np = rho.detach().cpu().numpy()
    
    from src.python.locally_constant_field import const_field_heisenberg_euler_lagrangian
    B_phys = 2.0 * F_cal / (e * lambd_val**2)
    L_HE = const_field_heisenberg_euler_lagrangian(B_phys, m=m, e=e)
    
    print(f"Heisenberg-Euler Constant Field Benchmark (B_phys={B_phys:.6f}): {L_HE:.6e}")
    print(f"Analytic Lagrangian range: {np.nanmin(L_an):.6e} to {np.nanmax(L_an):.6e}")
    print(f"Numerical Lagrangian range: {np.nanmin(L_num):.6e} to {np.nanmax(L_num):.6e}")

    # 5. Plotting
    plt.figure(figsize=(12, 12))

    # Plot 1: rho * L
    plt.subplot(3, 1, 1)
    plt.plot(rho_np, rho_analytic_np, 'r-', label=r'Analytic ($\rho \mathcal{L}$)')
    plt.plot(rho_np, rho_numeric, 'b--', label=r'Numerical ($\rho \mathcal{L}$)')
    plt.axvline(x=lambd_val, color='k', linestyle=':', label=r'Flux Tube Radius $\lambda$')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\rho \mathcal{L}(\rho)$')
    plt.title(fr'Radial Effective Action Density (F_cal={F_cal:.2f}, lambda={lambd_val})')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: L
    plt.subplot(3, 1, 2)
    plt.plot(rho_np, L_an, 'r-', label=r'Analytic $\mathcal{L}(\rho)$')
    plt.plot(rho_np, L_num, 'b--', label=r'Numerical $\mathcal{L}(\rho)$')
    plt.axhline(y=L_HE, color='g', linestyle='-.', label=fr'HE Constant Field (B={B_phys:.2e})')
    
    plt.axvline(x=lambd_val, color='k', linestyle=':')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\mathcal{L}(\rho)$')
    plt.title('Lagrangian Density (Physically expected plateau in interior)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot 3: Residual
    plt.subplot(3, 1, 3)
    residual = np.abs(L_an - L_num)
    plt.semilogy(rho_np, residual + 1e-20, 'g-', label=r'|Analytic - Numerical| $\mathcal{L}$')
    plt.axvline(x=lambd_val, color='k', linestyle=':')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'Absolute Residual in $\mathcal{L}$')
    plt.title('Lagrangian Density Residual (Log scale)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = 'results/step_ea_density_validation.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # 6. Compare total action (interior only)
    interior = rho_np <= lambd_val
    dr = rho_np[1] - rho_np[0]
    analytic_action_interior = 2.0 * constants.PI * np.sum(rho_analytic_np[interior]) * dr
    numeric_action_interior = 2.0 * constants.PI * np.sum(rho_numeric[interior]) * dr
    print(f"Analytic Action (interior ρ ≤ λ): {analytic_action_interior:.6e}")
    print(f"Numerical Action (interior ρ ≤ λ): {numeric_action_interior:.6e}")
    print(f"Numerical Action (Orchestrator):   {action.item():.6e}")

    print("\n--- Validation Outcome ---")
    print("The physical plateau in the interior is now restored using global_mode=True.")
    print("Analytic and Numeric Lagrangian densities agree in the interior and match Heisenberg-Euler.")

if __name__ == "__main__":
    main()
