import torch
import numpy as np
from pytorch_solver import PyTorchSolver
from renormalization import Renormalizer
from profiles import StepFunctionProfile
from typing import List, Dict, Any, Optional, Tuple, Union


class Orchestrator:
    def __init__(self, device: Optional[str] = None, batch_size: int = 128) -> None:
        self.backend = PyTorchSolver(device=device)
        self.device = self.backend.device
        self.renormalizer = Renormalizer(device=self.device)
        self.batch_size = batch_size

    def compute_effective_action(self, field_profile: Any, chi_values: List[complex], ml_values: List[int], sigma3_values: List[int], m: float = 1.0, e: float = 1.0, chi_threshold: float = 100.0, collect_density: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the full effective action by integrating over chi and summing over ml.
        Implements batching to manage memory and UV renormalization.

        Dimensions:
        chi: [L^-1]
        m: [L^-1]
        e: [L^0]
        Action output: [L^0] (dimensionless vacuum potential energy density in natural units)
        """
        rho, _, _ = field_profile.get_arrays(as_numpy=False)
        rho = rho.to(self.device)
        n_points = len(rho)
        
        # Integration weights for rho (trapezoidal)
        rho_weights = torch.zeros_like(rho)
        rho_weights[1:-1] = (rho[2:] - rho[:-2]) / 2.0
        rho_weights[0] = (rho[1] - rho[0]) / 2.0
        rho_weights[-1] = (rho[-1] - rho[-2]) / 2.0
        rho_factor = rho_weights.to(torch.complex128)

        # Initialize density tracking if requested
        density_integrand = torch.zeros_like(rho, dtype=torch.complex128) if collect_density else None

        # Prepare parameters for batching
        all_params = []
        for chi in chi_values:
            for ml in ml_values:
                for s3 in sigma3_values:
                    all_params.append({'chi': chi, 'ml': ml, 'sigma3': s3, 'm': m, 'e': e})

        total_inner_sum = torch.zeros(len(chi_values), device=self.device, dtype=torch.complex128)

        # Mapping chi index for easy summation
        chi_map = {chi: i for i, chi in enumerate(chi_values)}

        # Process in batches
        for i in range(0, len(all_params), self.batch_size):
            batch = all_params[i : i + self.batch_size]

            # Split batch into numerical and asymptotic regimes
            numerical_batch = [p for p in batch if abs(p['chi']) <= chi_threshold]
            asymptotic_batch = [p for p in batch if abs(p['chi']) > chi_threshold]

            # Initialize renormalized_g for the whole batch
            renormalized_g = torch.zeros((len(batch), n_points), device=self.device, dtype=torch.complex128)

            if numerical_batch:
                # Solve ODE for numerical batch
                num_results, num_w0 = self.backend.solve_batch(numerical_batch, field_profile)

                # Get G0 and UV sub for numerical batch
                num_chi = torch.tensor([p['chi'] for p in numerical_batch], device=self.device, dtype=torch.complex128)
                num_ml = torch.tensor([p['ml'] for p in numerical_batch], device=self.device, dtype=torch.int32)

                num_g0 = self.renormalizer.compute_g0(num_chi, num_ml, m, rho)
                num_uv = self.renormalizer.compute_uv_subtraction(num_chi, num_ml, m, rho, field_profile)

                # Path A matching: G_num and G_0 are now both dimensionless.
                # Renorm = G_num - G_0 - UV_sub
                num_renorm = num_results - num_g0 - num_uv

                # Place into renormalized_g
                num_indices = [idx for idx, p in enumerate(batch) if abs(p['chi']) <= chi_threshold]
                renormalized_g[num_indices] = num_renorm

            if asymptotic_batch:
                asymp_indices = [idx for idx, p in enumerate(batch) if abs(p['chi']) > chi_threshold]
                renormalized_g[asymp_indices] = 0.0

            if collect_density:
                density_integrand += torch.sum(renormalized_g, dim=0)

            # Integration over rho: Eq 2.59 uses rho * (G_dim - G0_dim)
            inner_int = torch.sum(renormalized_g * rho * rho_factor, dim=-1) # (batch_size,)

            # Accumulate into total_inner_sum based on chi
            for idx, p in enumerate(batch):
                chi_idx = chi_map[p['chi']]
                total_inner_sum[chi_idx] += inner_int[idx]

        # Final integration over chi
        chi_tensor = torch.tensor([complex(c) for c in chi_values], device=self.device, dtype=torch.complex128)
        chi_real = chi_tensor.real
        chi_weights = torch.zeros_like(chi_real)
        if len(chi_real) > 1:
            chi_weights[1:-1] = (chi_real[2:] - chi_real[:-2]) / 2.0
            chi_weights[0] = (chi_real[1] - chi_real[0]) / 2.0
            chi_weights[-1] = (chi_real[-1] - chi_real[-2]) / 2.0
        else:
            chi_weights[0] = 1.0

        # Coefficient -1/(8*pi^2) accounts for 4D momentum measure (1/8pi^2) and spin degeneracy (2) 
        # and the -1/2 from the Tr ln expansion. Total = -1/(8*pi^2).
        action = -1.0 / (8.0 * np.pi**2) * torch.sum(chi_tensor**3 * total_inner_sum * chi_weights)

        if collect_density:
            return action, density_integrand
        return action



def generate_params_grid(chi_values: List[complex], ml_values: List[int], sigma3_values: List[int], m: float = 1.0, e: float = 1.0) -> List[Dict[str, Any]]:
    grid = []
    for chi in chi_values:
        for ml in ml_values:
            for s3 in sigma3_values:
                grid.append({
                    'chi': chi,
                    'ml': ml,
                    'sigma3': s3,
                    'm': m,
                    'e': e
                })
    return grid
