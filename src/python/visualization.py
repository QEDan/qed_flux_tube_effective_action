import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Union, Any, List

def plot_energy_surface(lambda_values: List[float], action_values: List[float]) -> None:
    """
    Plots the effective action energy surface E = -Gamma vs flux tube width lambda.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, [-a for a in action_values], marker='o')
    plt.title("Quantum Corrected Energy Functional vs. Flux Tube Width")
    plt.xlabel("Width $\lambda$")
    plt.ylabel("Energy $E = -\Gamma$")
    plt.grid(True)
    plt.savefig("results/energy_surface.png")
    plt.close()

def plot_profile_comparison(rho: np.ndarray, profiles_dict: Any) -> None:
    """
    Compares classical (e.g., Step, London) and optimized profiles.
    profiles_dict: dict of profile_name: B_field_array
    """
    plt.figure(figsize=(8, 6))
    for name, b_field in profiles_dict.items():
        plt.plot(rho, b_field, label=name)
    plt.title("Magnetic Field Profile Comparison")
    plt.xlabel("Radial coordinate $\\rho$")
    plt.ylabel("Magnetic field $B(\\rho)$")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/profile_comparison.png")
    plt.close()

def plot_density_diffusion(rho: torch.Tensor, density_integrand: torch.Tensor, flux_f: float) -> None:
    """
    Visualizes the energy density integrand rho^2 * Delta_G.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(rho.detach().cpu().numpy(), density_integrand.detach().cpu().numpy().real)
    plt.title(f"Energy Density Integrand ($\mathcal{{F}}={flux_f}$)")
    plt.xlabel("Radial coordinate $\\rho$")
    plt.ylabel("Integrand $\\rho^2 \Delta G(\rho)$")
    plt.grid(True)
    plt.savefig(f"results/density_diffusion_f{flux_f}.png")
    plt.close()
