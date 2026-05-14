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
        self.centers = torch.linspace(0.0, rho_max, num_basis).to(torch.float64)
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
        self.weights = nn.Parameter(torch.abs(weights_init).to(torch.float64))
        
    def forward(self, rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # B_raw(rho) = sum_i w_i * exp(-(rho - c_i)^2 / (2 * sigma^2))
        rho_expanded = rho.view(-1, 1)
        basis = torch.exp(-(rho_expanded - self.centers)**2 / (2 * self.sigma**2))
        
        B_raw = torch.matmul(basis, torch.nn.functional.softplus(self.weights)) # Smooth positivity constraint
        
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

class SplineProfile(nn.Module):
    """
    Field profile defined using Cubic B-Spline basis functions.
    B(rho) = sum_i w_i * N_i(rho)
    Provides C2 continuity, local support, and a low-dimensional parameter space
    ideal for landscape mapping and Hessian analysis.
    """
    def __init__(self, num_basis: int = 15, total_flux: float = 2.0*np.pi*0.4, rho_max: float = 10.0, 
                 positivity_constraint: bool = True) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.total_flux = total_flux
        self.Phi_over_2pi = total_flux / (2.0 * np.pi)
        self.rho_max = rho_max
        self.positivity_constraint = positivity_constraint
        
        # Basis parameters: uniform spacing of spline centers
        self.centers = torch.linspace(0.0, rho_max, num_basis).to(torch.float64)
        self.h = rho_max / (num_basis - 1) if num_basis > 1 else 1.0
        
        # Initialize weights: Jump-start with a smooth decay
        # B(rho) ~ exp(-rho^2/2)
        target_B = torch.exp(-self.centers**2 / 2.0).to(torch.float64)
        
        # Solve least-squares to initialize weights: w = (Basis^T * Basis)^-1 * Basis^T * target_B
        basis_init = self._get_basis(self.centers)
        # Use pseudo-inverse for stability in least-squares
        weights_init = torch.linalg.lstsq(basis_init, target_B).solution
        
        # Weights to optimize
        self.weights = nn.Parameter(torch.abs(weights_init).to(torch.float64) if positivity_constraint else weights_init.to(torch.float64))

    def _get_basis(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the Cubic B-Spline basis functions at given rho values.
        Enforces B'(0) = 0 by mirroring basis functions: N_sym(rho) = N(rho) + N(-rho).
        """
        r = rho.view(-1, 1)
        c = self.centers.view(1, -1)
        
        # Cubic B-spline function
        def b_spline(z):
            z_abs = torch.abs(z)
            b1 = (2.0/3.0) - z_abs**2 + 0.5 * z_abs**3
            b2 = (1.0/6.0) * (2.0 - z_abs)**3
            return torch.where(z_abs < 1.0, b1, torch.where(z_abs < 2.0, b2, torch.zeros_like(z)))

        # N_sym(rho) = N((rho - c)/h) + N((-rho - c)/h)
        basis = b_spline((r - c) / self.h) + b_spline((-r - c) / self.h)
        return basis

    def forward(self, rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        basis = self._get_basis(rho)
        
        w = torch.nn.functional.softplus(self.weights) if self.positivity_constraint else self.weights
        B_raw = torch.matmul(basis, w).view(-1, 1)
        
        # Renormalize to conserve flux: Phi = 2*pi * integral(rho * B) = total_flux
        dr = rho[1] - rho[0] if len(rho) > 1 else torch.tensor(0.1)
        rho_vals = rho.view(-1)
        raw_flux = 2.0 * np.pi * torch.sum(B_raw.squeeze() * rho_vals * dr)
        
        norm_factor = (self.Phi_over_2pi * 2.0 * np.pi) / (raw_flux + 1e-15)
        B_vals = B_raw * norm_factor
        
        # Clamp B_vals to prevent numerical divergence in extreme regimes
        B_vals = torch.clamp(B_vals, min=-1e6, max=1e6)
        
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
