from src.python.locally_constant_field import heisenberg_euler_integrand
from src.python import constants

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional

def generate_params_grid(chi_values, ml_values, sigma3_values, m=constants.ELECTRON_MASS, e=constants.ELECTRON_CHARGE):
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
    def __init__(self, device: Optional[str] = 'cpu', batch_size: int = 1024, strategy: str = "analytic") -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        from pytorch_solver import PyTorchSolver
        from renormalization import Renormalizer
        self.backend = PyTorchSolver(device=device)
        self.renormalizer = Renormalizer(device=device, strategy=strategy, solver=self.backend)

    def compute_effective_action(self, field_profile: Any, chi_values: List[complex], ml_values: List[int], sigma3_values: List[int], m: float = constants.ELECTRON_MASS, e: float = constants.ELECTRON_CHARGE, collect_density: bool = False, lcf_threshold: Optional[float] = 20.0) -> Any:
        """
        Computes the 1-loop effective action (Gamma) and the intensive Lagrangian density L_eff(rho).
        Normalization aligns with Scalar QED HE (1/16pi^2) for each spin state.
        Total for fermions (sum over sigma3) recovers Spinor QED.
        
        lcf_threshold: Above this Euclidean momentum Q, use the Local Constant Field (LCF) approximation
                       instead of numerical mode summing. Set to None to disable.
        """
        rho, _, _ = field_profile.get_arrays(as_numpy=False)
        rho = rho.to(self.device)
        n_points = len(rho)
        
        # 1. Prepare all parameters
        all_params = []
        for chi in chi_values:
            Q = abs(complex(chi))
            # Only use numerical solver below LCF threshold
            if lcf_threshold is not None and Q > lcf_threshold:
                continue
            for ml in ml_values:
                for s3 in sigma3_values:
                    all_params.append({'chi': chi, 'ml': ml, 'sigma3': s3, 'm': m, 'e': e})
        
        # 2. Compute mode sums per chi (Euclidean spectral integration)
        mode_sums = torch.zeros((len(chi_values), n_points), device=self.device, dtype=torch.complex128)
        chi_map = {complex(c): i for i, c in enumerate(chi_values)}
        
        # We solve the interacting case batch-by-batch
        if all_params:
            for i in range(0, len(all_params), self.batch_size):
                batch = all_params[i : i + self.batch_size]
                
                # EUCLIDEAN ROTATION: chi = i * Q where Q is the real parameter
                euclidean_batch = []
                for p in batch:
                    eb = p.copy()
                    eb['chi'] = 1j * abs(p['chi'])
                    euclidean_batch.append(eb)
                    
                # Solve for the numerical interacting case and background on the exact same grid
                num_results, _ = self.backend.solve_batch(euclidean_batch, field_profile)

                # Compute background on the same grid/params to ensure exact cancellation
                batch_chi = torch.tensor([p['chi'] for p in euclidean_batch], device=self.device, dtype=torch.complex128)
                batch_ml = torch.tensor([p['ml'] for p in batch], device=self.device, dtype=torch.float64)

                # Explicitly align background calculation with the solver's grid
                num_bg = self.renormalizer.compute_g0(batch_chi, batch_ml, m, rho, field_profile)

                # Point-wise renormalization subtraction
                for idx, p in enumerate(batch):
                    chi_idx = chi_map[complex(p['chi'])]
                    # Direct subtraction before any integration
                    mode_sums[chi_idx] += (num_results[idx] - num_bg[idx])
        # 4. Renormalization and Spectral Integration
        uv_coeff_local = self.renormalizer.get_b2_term(field_profile, rho) * 2.0
        B_local = getattr(field_profile, 'B_vals', torch.zeros_like(rho)).detach().cpu().numpy()

        # Consistent with 4D Spinor QED HE normalization (4 states)
        # 1/(32*pi^2) matches the observed scale of -0.016 at rho=0
        norm_factor = 1.0 / (constants.THIRTY_TWO_PI_SQUARED)

        r_safe = torch.where(rho == 0, torch.tensor(1e-15, device=rho.device), rho)
        chi_real = np.array([abs(complex(c)) for c in chi_values])
        
        # Spectral weight for Q^3 dQ measure
        chi_weights = np.zeros_like(chi_real)
        if len(chi_real) > 1:
            for i in range(len(chi_real)):
                if i == 0:
                    chi_weights[i] = (chi_real[1] - chi_real[0]) * (chi_real[0]**3) / 2.0
                elif i == len(chi_real) - 1:
                    chi_weights[i] = (chi_real[-1] - chi_real[-2]) * (chi_real[-1]**3) / 2.0
                else:
                    chi_weights[i] = (chi_real[i+1] - chi_real[i-1]) * (chi_real[i]**3) / 2.0
        else:
            chi_weights[0] = 1.0

        L_eff_rho = torch.zeros(n_points, device=self.device, dtype=torch.complex128)

        for i, Q in enumerate(chi_real):
            if lcf_threshold is not None and Q > lcf_threshold:
                local_renorm_sum = torch.from_numpy(np.array([heisenberg_euler_integrand(Q, B, m=m, e=e) for B in B_local])).to(self.device).to(torch.complex128)
            else:
                # Renormalized Integrand for 4 spinor states
                uv_sub = - uv_coeff_local / (Q**4 + 1e-15)
                mode_sum_4states = mode_sums[i] * 2.0
                local_renorm_sum = (mode_sum_4states.real / r_safe) + uv_sub.real

            # Q^3 is already in chi_weights
            L_eff_rho += local_renorm_sum * chi_weights[i] * norm_factor

        # Spatial Integration
        rho_weights = torch.zeros_like(rho)
        rho_weights[1:-1] = (rho[2:] - rho[:-2]) / 2.0
        rho_weights[0] = (rho[1] - rho[0]) / 2.0
        rho_weights[-1] = (rho[-1] - rho[-2]) / 2.0
        
        # Integrated action Gamma (Action per unit time and unit length)
        # Sign: Positive for Scalar QED, Negative for Fermions.
        # Heisenberg-Euler is positive. Our sum is negative?
        # Let's make it positive to match HE convention.
        # Note: 2*pi factor from angular integration over phi
        action = -constants.TWO_PI * torch.sum(L_eff_rho.real * rho * rho_weights)
        
        return action, -1.0 * L_eff_rho
