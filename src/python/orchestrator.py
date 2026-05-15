import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional

def generate_params_grid(chi_values, ml_values, sigma3_values, m=1.0, e=1.0):
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

class Orchestrator:
    def __init__(self, device: Optional[str] = 'cpu', batch_size: int = 1024) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        from pytorch_solver import PyTorchSolver
        from renormalization import Renormalizer
        self.backend = PyTorchSolver(device=device)
        self.renormalizer = Renormalizer(device=device)

    def compute_effective_action(self, field_profile: Any, chi_values: List[complex], ml_values: List[int], sigma3_values: List[int], m: float = 1.0, e: float = 1.0, chi_threshold: float = 100.0, collect_density: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        rho, _, _ = field_profile.get_arrays(as_numpy=False)
        rho = rho.to(self.device)
        n_points = len(rho)
        rho_weights = torch.zeros_like(rho)
        rho_weights[1:-1] = (rho[2:] - rho[:-2]) / 2.0
        rho_weights[0] = (rho[1] - rho[0]) / 2.0
        rho_weights[-1] = (rho[-1] - rho[-2]) / 2.0
        rho_factor = rho_weights.to(torch.complex128)
        
        density_integrand = torch.zeros_like(rho, dtype=torch.complex128) if collect_density else None
        
        all_params = []
        for chi in chi_values:
            for ml in ml_values:
                for s3 in sigma3_values:
                    all_params.append({'chi': chi, 'ml': ml, 'sigma3': s3, 'm': m, 'e': e})
        total_inner_sum = torch.zeros(len(chi_values), device=self.device, dtype=torch.complex128)
        chi_map = {chi: i for i, chi in enumerate(chi_values)}
        
        from profiles import FieldProfile
        vacuum_profile = FieldProfile(rho)
        chi_tensor = torch.tensor([complex(c) for c in chi_values], device=self.device, dtype=torch.complex128)
        
        for i in range(0, len(all_params), self.batch_size):
            batch = all_params[i : i + self.batch_size]
            num_results, _ = self.backend.solve_batch(batch, field_profile)
            num_bg, _ = self.backend.solve_batch(batch, vacuum_profile)
            num_chi = torch.tensor([p['chi'] for p in batch], device=self.device, dtype=torch.complex128)
            num_ml = torch.tensor([p['ml'] for p in batch], device=self.device, dtype=torch.int32)
            num_uv = self.renormalizer.compute_uv_subtraction(num_chi, num_ml, m, rho, field_profile)
            num_renorm = num_results - num_bg - num_uv
            
            if collect_density:
                density_integrand += torch.sum(num_renorm, dim=0)
            
            # Eq 2.50: spatial integrand is rho^2 * G_radial. 
            # Our num_renorm is rho * G_radial. So we need to multiply by rho.
            inner_int = torch.sum(num_renorm * rho.unsqueeze(0) * rho_weights.unsqueeze(0), dim=-1)
            for idx, p in enumerate(batch):
                chi_idx = chi_map[p['chi']]
                total_inner_sum[chi_idx] += inner_int[idx]
        ml_max = max(abs(min(ml_values)), abs(max(ml_values)))
        tail_corr_rho = self.renormalizer.compute_tail_correction(chi_tensor, m, rho, field_profile, ml_max)
        tail_corr = torch.sum(tail_corr_rho * rho_factor, dim=-1)
        total_inner_sum += tail_corr

        # Proper integration measure for 4D effective action
        chi_real = chi_tensor.real

        # Apply UV subtraction (B^2/6) per-mode-sum
        uv_coeff = self.renormalizer.get_b2_term(field_profile, rho)
        # sum over all modes is B^2/6. The subtraction must occur in the spectral integral.
        # Action = pi * Integral chi^3 dchi * ( Sum_ml(Delta G) - UV_sub )
        total_inner_sum -= torch.sum(uv_coeff * rho_weights, dim=-1)

        chi_weights = torch.zeros_like(chi_real)
        if len(chi_real) > 1:
            chi_weights[1:-1] = (chi_real[2:] - chi_real[:-2]) / 2.0
            chi_weights[0] = (chi_real[1] - chi_real[0]) / 2.0
            chi_weights[-1] = (chi_real[-1] - chi_real[-2]) / 2.0
        else:
            chi_weights[0] = 1.0

        action_integrand = np.pi * chi_real**3 * total_inner_sum
        action = torch.sum(action_integrand * chi_weights)

        print(f"DEBUG: Action={action.item():.6e}")

        if collect_density:
            # Return per-chi density [n_chi, n_points]
            # This is the integrand *before* summing over chi (matching telemetry expectations).
            return action, density_integrand
        return action

