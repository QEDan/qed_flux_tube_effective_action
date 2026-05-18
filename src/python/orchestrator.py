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
        
        # 2. Compute mode sums per chi (Euclidean spectral integration)
        mode_sums = torch.zeros((len(chi_values), n_points), device=self.device, dtype=torch.complex128)
        chi_map = {complex(c): i for i, c in enumerate(chi_values)}
        
        # We solve the interacting case batch-by-batch
        for i in range(0, len(all_params), self.batch_size):
            batch = all_params[i : i + self.batch_size]
            
            # EUCLIDEAN ROTATION: chi = i * Q where Q is the real parameter
            euclidean_batch = []
            for p in batch:
                eb = p.copy()
                eb['chi'] = 1j * abs(p['chi']) # Ensure chi is purely imaginary
                euclidean_batch.append(eb)
                
            num_results, _ = self.backend.solve_batch(euclidean_batch, field_profile)
            
            # Topological vacuum subtraction using local flux matching
            batch_chi = torch.tensor([p['chi'] for p in euclidean_batch], device=self.device, dtype=torch.complex128)
            batch_ml = torch.tensor([p['ml'] for p in batch], device=self.device, dtype=torch.float64)
            num_bg = self.renormalizer.compute_g0_local(batch_chi, batch_ml, m, rho, field_profile)
            
            for idx, p in enumerate(batch):
                chi_idx = chi_map[complex(p['chi'])]
                mode_sums[chi_idx] += (num_results[idx] - num_bg[idx])
            
            if i == 0:
                print(f"DEBUG: Q={abs(batch[0]['chi']):.2f}, max_res={torch.max(torch.abs(num_results)):.2e}, max_bg={torch.max(torch.abs(num_bg)):.2e}, max_diff={torch.max(torch.abs(num_results-num_bg)):.2e}")

        # 4. Renormalization and Spectral Integration

        # Use the local field strength for point-wise UV subtraction
        uv_coeff_local = self.renormalizer.get_b2_term(field_profile, rho)

        # Correct Scalar QED EH normalization for chi dchi measure
        # L = (1/4pi^2) * Integral Q dQ * [ -Delta_G/rho - B^2/12Q^2 ]
        norm_factor = 1.0 / (4.0 * np.pi**2)

        r_safe = torch.where(rho == 0, torch.tensor(1e-15, device=rho.device), rho)

        # Treat chi_values as Euclidean momenta Q
        chi_real = np.array([abs(complex(c)) for c in chi_values])
        chi_weights = np.zeros_like(chi_real)
        if len(chi_real) > 1:
            chi_weights[1:-1] = (chi_real[2:] - chi_real[:-2]) / 2.0
            chi_weights[0] = (chi_real[1] - chi_real[0]) / 2.0
            chi_weights[-1] = (chi_real[-1] - chi_real[-2]) / 2.0
        else:
            chi_weights[0] = 1.0

        L_eff_rho = torch.zeros(n_points, device=self.device, dtype=torch.complex128)

        for i, Q in enumerate(chi_real):
            # UV subtraction: Local term (eB)^2 / 12 scaled for the spectral integral.
            # Q is the Euclidean momentum. The log divergence comes from Q * (1/Q^2) = 1/Q.
            num_uv = uv_coeff_local / (Q**2 + 1e-15)

            # Local density integrand: (Delta_G / rho) + UV_term
            # In Euclidean space, Delta_G is negative for positive potential.
            # We take the real part and flip sign to get positive L_eff.
            local_renorm_sum = - (mode_sums[i].real / r_safe) - num_uv.real

            L_eff_rho += Q * local_renorm_sum * chi_weights[i] * norm_factor

        
        # Spatial Integration
        rho_weights = torch.zeros_like(rho)
        rho_weights[1:-1] = (rho[2:] - rho[:-2]) / 2.0
        rho_weights[0] = (rho[1] - rho[0]) / 2.0
        rho_weights[-1] = (rho[-1] - rho[-2]) / 2.0
        
        # Integrated action Gamma (Action per unit time and unit length)
        # EH Lagrangian density normalization: positive for Scalar QED correction
        action = torch.sum(L_eff_rho.real * rho * rho_weights)
        
        print(f"DEBUG: Action={action.item():.6e}")
        
        if collect_density:
            return action, L_eff_rho
        return action, None
