import torch
import numpy as np
import matplotlib.pyplot as plt
from src.python import constants
from src.python.profiles import StepFunctionProfile
from src.python.orchestrator import Orchestrator

def main():
    # 1. Parameters
    F = constants.FLUX_QUANTUM
    lambd = 1.0
    m = constants.ELECTRON_MASS
    e = constants.ELECTRON_CHARGE
    
    # 2. Extend the radial grid to see long-range behavior
    n_rho = 300
    # Use log-spaced grid for visualization: 0.01 to 10000
    rho = torch.logspace(0, 6, n_rho, dtype=torch.float64)

    # 3. Compute density via Orchestrator
    print("Computing densities via Orchestrator...")
    profile = StepFunctionProfile(rho, lambd, F, e=e)
    orch = Orchestrator(strategy="analytic", device='cpu', batch_size=2048)
    
    chi_values = torch.linspace(0.1, 10.0, 50, dtype=torch.complex128)
    ml_values = list(range(10))
    sigma3_values = [1, -1]
    
    _, num_density = orch.compute_effective_action(
        profile,
        chi_values,
        ml_values,
        sigma3_values,
        m=m,
        e=e,
        collect_density=True
    )
    
    rho_numeric = (num_density.real * rho).detach().cpu().numpy()
    rho_np = rho.detach().cpu().numpy()
    
    # 4. Plotting on log-log axis (negative values only)
    plt.figure(figsize=(10, 6))
    
    # Filter for negative values
    mask = (rho_numeric < 0)
    plt.loglog(rho_np[mask], -rho_numeric[mask], 'b-', label=r'$-\rho(\rho)$')
    
    plt.axvline(x=lambd, color='k', linestyle=':', label='Flux Tube Radius ($\lambda$)')
    
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$-\rho(\rho)$ (for negative $\rho$)')
    plt.title(r'Negative Effective Action Density Decay (Log-Log scale)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plot_path = 'results/step_ea_density_log_rho.png'
    plt.savefig(plot_path)
    print(f"Log-log plot saved to {plot_path}")
    print("\n--- Validation Outcome ---")
    print("Check results/step_ea_density_log_rho.png.")
    print("Verify the long-range power-law decay of the negative density.")
    print("The density should approach zero as rho -> infinity.")

if __name__ == "__main__":
    main()
