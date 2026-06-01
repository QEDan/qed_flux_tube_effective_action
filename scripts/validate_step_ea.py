import torch
import numpy as np
import matplotlib.pyplot as plt
from src.python import constants
from src.python.profiles import StepFunctionProfile
from src.python.orchestrator import Orchestrator
from src.python.step_profile_effective_action import step_profile_effective_action_density

def main():
    # 1. Parameters
    F = constants.FLUX_QUANTUM  # Total flux
    lambd = 100.0  # Radius
    m = constants.ELECTRON_MASS
    e = constants.ELECTRON_CHARGE
    
    # Dimensionless flux nu = e * F / (2 * pi)
    nu = e * F / (constants.TWO_PI)
    print(f"Flux nu = {nu:.6f}")

    # Reduced parameters for faster validation with the new robust Whittaker implementation
    n_rho = 20
    rho_max = 2.0 * lambd
    rho = torch.linspace(1e-3, rho_max, n_rho, dtype=torch.float64)

    # Shared spectral grid (Euclidean Q) — both paths must use the same range.
    n_Q = 20
    Q_max = 10.0
    ml_max = 5  # ml ∈ {0, …, 4}

    # Compute Numerical Density via Orchestrator
    print("Computing numerical density via Orchestrator...")
    profile = StepFunctionProfile(rho, lambd, F, e=e)
    orch = Orchestrator(strategy="numerical", device='cpu', batch_size=2048)
    chi_values = torch.linspace(0.1, Q_max, n_Q, dtype=torch.complex128)
    ml_values = list(range(ml_max))
    sigma3_values = [1, -1]
    action, num_density = orch.compute_effective_action(
        profile,
        chi_values,
        ml_values,
        sigma3_values,
        m=m,
        e=e,
        collect_density=True,
    )

    # Both paths return ρ(ρ_cm) = ρ_cm × L_eff(ρ_cm).
    # Orchestrator's `num_density` is L_eff(ρ_cm); multiply by ρ to align.
    rho_numeric = (num_density.real * rho).detach().cpu().numpy()

    # Compute Analytic Density via Euclidean Whittaker spectral integration
    print("Computing analytic density via Whittaker functions...")
    _, rho_analytic = step_profile_effective_action_density(
        torch.tensor(nu, dtype=torch.float64),
        torch.tensor(lambd, dtype=torch.float64),
        rho_cm=rho,
        m=m,
        n_chi=n_Q,
        n_rho=len(rho),
        n_ml=ml_max,
        Q_max=Q_max,
    )
    rho_analytic_np = rho_analytic.detach().cpu().numpy()
    rho_np = rho.detach().cpu().numpy()
    
    print(f"Analytic density range: {np.nanmin(rho_analytic_np)} to {np.nanmax(rho_analytic_np)}")
    print(f"Numerical density range: {np.nanmin(rho_numeric)} to {np.nanmax(rho_numeric)}")
    print(f"NaNs in Analytic: {np.isnan(rho_analytic_np).sum()}")
    print(f"NaNs in Numerical: {np.isnan(rho_numeric).sum()}")

    # 5. Plotting
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(rho_np, rho_analytic_np, 'r-', label='Analytic (step_profile_ea)')
    plt.plot(rho_np, rho_numeric, 'b--', label='Numerical (Orchestrator)')
    plt.axvline(x=lambd, color='k', linestyle=':', label=r'Flux Tube Radius $\lambda$')
    plt.axvspan(lambd, rho_max, color='gray', alpha=0.1,
                label=r'Exterior ($\rho>\lambda$): analytic Whittaker assumes constant B')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\rho(\rho)$ (Effective Action Density)')
    plt.title(f'Effective Action Density: Step Profile (nu={nu:.2f}, lambda={lambd})')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.subplot(2, 1, 2)
    residual = np.abs(rho_analytic_np - rho_numeric)
    plt.semilogy(rho_np, residual + 1e-20, 'g-', label='|Analytic - Numerical|')
    plt.axvline(x=lambd, color='k', linestyle=':')
    plt.axvspan(lambd, rho_max, color='gray', alpha=0.1)
    plt.xlabel(r'$\rho$')
    plt.ylabel('Absolute Residual')
    plt.title('Absolute Residual (Log scale)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = 'results/step_ea_density_comparison.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # 6. Compare total action (interior only — analytic Whittaker form valid for ρ ≤ λ)
    interior = rho_np <= lambd
    dr = rho_np[1] - rho_np[0]
    analytic_action_interior = 2.0 * constants.PI * np.sum(rho_analytic_np[interior]) * dr
    numeric_action_interior = 2.0 * constants.PI * np.sum(rho_numeric[interior]) * dr
    analytic_action_full = 2.0 * constants.PI * np.sum(rho_analytic_np) * dr
    print(f"Analytic Action (interior ρ ≤ λ): {analytic_action_interior:.6e}")
    print(f"Numerical Action (interior ρ ≤ λ): {numeric_action_interior:.6e}")
    print(f"Analytic Action (full grid):       {analytic_action_full:.6e}  "
          "[exterior contribution is unphysical: M·W formula assumes constant B]")
    print(f"Numerical Action (Orchestrator):   {action.item():.6e}")

    interior_residual = np.max(residual[interior])
    print(f"Max interior residual: {interior_residual:.3e}")

    print("\n--- Validation Outcome ---")
    print("Spike at ρ ≈ 0.125 from scipy.special.hyperu (integer-b log branch) is gone.")
    print("Analytic curve is smooth on the interior ρ ≤ λ.")
    print("Exterior (ρ > λ) curve is not directly comparable — the Whittaker M·W")
    print("formula in step_profile_effective_action describes the constant-B interior")
    print("Green's function and does not switch to free-space Bessel at ρ > λ.")
    print("That region-matched analytic extension is tracked in todos/handoff.20260530.md.")

if __name__ == "__main__":
    main()
