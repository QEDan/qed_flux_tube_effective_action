import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class BasisProfile(nn.Module):
    """
    Field profile defined as a sum of Gaussian basis functions:
    B(rho) = sum_i w_i * exp(-(rho - c_i)^2 / (2 * sigma^2))
    The total flux is conserved by re-normalizing the sum at each step.
    """
    def __init__(self, num_basis: int = 8, total_flux: float = 2.0*np.pi*0.4, rho_max: float = 10.0) -> None:
        super().__init__()
        self.Phi_over_2pi = total_flux / (2.0 * np.pi)
        self.num_basis = num_basis
        
        # Basis parameters
        self.centers = torch.linspace(0.0, rho_max, num_basis)
        self.sigma = rho_max / (num_basis * 2.0)
        
        # Initialize weights to approximate Sech2(rho) profile for jump-starting
        # B(rho) = sech^2(rho/lambd). We use a lambda of ~1.0
        lambd = 1.0
        target_B = 1.0 / (torch.cosh(self.centers / lambd)**2)
        
        # Solve least-squares to initialize weights: w = (Basis^T * Basis)^-1 * Basis^T * target_B
        rho_expanded = self.centers.view(-1, 1)
        basis = torch.exp(-(rho_expanded - self.centers)**2 / (2 * self.sigma**2))
        weights_init = torch.linalg.lstsq(basis, target_B).solution
        
        # Weights to optimize
        self.weights = nn.Parameter(torch.abs(weights_init))
        
    def forward(self, rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # B_raw(rho) = sum_i w_i * exp(-(rho - c_i)^2 / (2 * sigma^2))
        rho_expanded = rho.view(-1, 1)
        basis = torch.exp(-(rho_expanded - self.centers)**2 / (2 * self.sigma**2))
        
        B_raw = torch.matmul(basis, torch.abs(self.weights)) # Weights must be positive for B > 0
        
        # Renormalize to conserve flux: Phi = 2*pi * integral(rho * B) = total_flux
        # rho_weights calculated for integration
        dr = rho[1] - rho[0] if len(rho) > 1 else torch.tensor(0.1)
        rho_vals = rho.view(-1)
        raw_flux = 2.0 * np.pi * torch.sum(B_raw.squeeze() * rho_vals * dr)
        
        norm_factor = (self.Phi_over_2pi * 2.0 * np.pi) / (raw_flux + 1e-15)
        B_vals = B_raw * norm_factor
        
        # Clamp B_vals to prevent numerical divergence
        B_vals = torch.clamp(B_vals, min=-1e5, max=1e5)
        
        # Derive A_phi analytically using cumulative integration
        # Flux(rho) = 2*pi * integral_0^rho B * r * dr
        # a_phi = Flux(rho) / (2*pi*rho)
        rho_integrand = B_vals.squeeze() * rho_vals
        dy = 0.5 * (rho_integrand[1:] + rho_integrand[:-1]) * (rho_vals[1:] - rho_vals[:-1])
        flux_integral = torch.zeros_like(rho_vals)
        flux_integral[1:] = torch.cumsum(dy, dim=0)
        
        r_safe = torch.where(rho_vals == 0, torch.tensor(1e-15, device=rho.device), rho_vals)
        a_phi = flux_integral / r_safe
        
        return B_vals, a_phi.view(-1, 1)
