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

    def compute_effective_action(self, field_profile: Any, chi_values: List[complex], ml_values: List[int], sigma3_values: List[int], m: float = 1.0, e: float = 1.0, collect_density: bool = False) -> Any:
        """
        Computes the 1-loop effective action (Gamma) and the intensive Lagrangian density L_eff(rho).
        Normalization aligns with Scalar QED HE (1/16pi^2) for each spin state.
        Total for fermions (sum over sigma3) recovers Spinor QED.
        """
        rho, _, _ = field_profile.get_arrays(as_numpy=False)
        rho = rho.to(self.device)
        n_points = len(rho)
        
        # 1. Prepare all parameters
        all_params = []
        for chi in chi_values:
            for ml in ml_values:
                for s3 in sigma3_values:
                    all_params.append({'chi': chi, 'ml': ml, 'sigma3': s3, 'm': m, 'e': e})
        
        # 2. Compute mode sums per chi
        mode_sums = torch.zeros((len(chi_values), n_points), device=self.device, dtype=torch.complex128)
        chi_map = {complex(c): i for i, c in enumerate(chi_values)}
        
        from profiles import FieldProfile
        vacuum_profile = FieldProfile(rho)
        
        for i in range(0, len(all_params), self.batch_size):
            batch = all_params[i : i + self.batch_size]
            num_results, _ = self.backend.solve_batch(batch, field_profile)
            num_bg, _ = self.backend.solve_batch(batch, vacuum_profile)
            
            for idx, p in enumerate(batch):
                chi_idx = chi_map[complex(p['chi'])]
                # Difference Delta_G = num_results - num_bg (units L)
                mode_sums[chi_idx] += (num_results[idx] - num_bg[idx])

        # Renormalization and Spectral Integration
        uv_coeff = self.renormalizer.get_b2_term(field_profile, rho)
        
        # Note: The intensive Lagrangian density requires dividing by the 4-volume trace factors.
        # Total intensive factor chosen to align with standard Scalar QED EH form.
        norm_factor = 1.0 / (16.0 * np.pi**4)
        
        r_safe = torch.where(rho == 0, torch.tensor(1e-15, device=rho.device), rho)
        
        chi_real = np.array([complex(c).real for c in chi_values])
        chi_weights = np.zeros_like(chi_real)
        if len(chi_real) > 1:
            chi_weights[1:-1] = (chi_real[2:] - chi_real[:-2]) / 2.0
            chi_weights[0] = (chi_real[1] - chi_real[0]) / 2.0
            chi_weights[-1] = (chi_real[-1] - chi_real[-2]) / 2.0
        else:
            chi_weights[0] = 1.0
            
        L_eff_rho = torch.zeros(n_points, device=self.device, dtype=torch.complex128)
        
        for i, chi in enumerate(chi_real):
            # UV subtraction: B^2 / (6 * chi^4)
            num_uv = uv_coeff / (chi**4)
            
            # Sum over modes Delta G / rho is dimensionless
            local_renorm_sum = (mode_sums[i] / r_safe) - num_uv
            
            L_eff_rho += chi**3 * local_renorm_sum * chi_weights[i] * norm_factor
        
        # Spatial Integration
        rho_weights = torch.zeros_like(rho)
        rho_weights[1:-1] = (rho[2:] - rho[:-2]) / 2.0
        rho_weights[0] = (rho[1] - rho[0]) / 2.0
        rho_weights[-1] = (rho[-1] - rho[-2]) / 2.0
        
        # Integrated action Gamma (Action per unit time and unit length)
        # EH Lagrangian density normalization: negative sign for Scalar QED correction
        action = -2.0 * np.pi * torch.sum(L_eff_rho * rho * rho_weights)
        
        print(f"DEBUG: Action={action.item():.6e}")
        
        if collect_density:
            return action, L_eff_rho
        return action, None
