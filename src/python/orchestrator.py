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
    def __init__(self, device: Optional[str] = 'cpu', batch_size: int = 1024, strategy: str = "analytic") -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        from pytorch_solver import PyTorchSolver
        from renormalization import Renormalizer
        self.backend = PyTorchSolver(device=device)
        self.renormalizer = Renormalizer(device=device, strategy=strategy, solver=self.backend)

    def compute_effective_action(self, field_profile: Any, chi_values: List[complex], ml_values: List[int], sigma3_values: List[int], m: float = 1.0, e: float = 1.0, collect_density: bool = False, lcf_threshold: Optional[float] = 20.0) -> Any:
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
                    
                num_results, _ = self.backend.solve_batch(euclidean_batch, field_profile)
                
                # Local topological vacuum subtraction: matching A_phi but with B=0
                # Using the actual field_profile ensures n = ml - e*A_phi*rho in compute_g0_local
                batch_chi = torch.tensor([p['chi'] for p in euclidean_batch], device=self.device, dtype=torch.complex128)
                batch_ml = torch.tensor([p['ml'] for p in batch], device=self.device, dtype=torch.float64)
                num_bg = self.renormalizer.compute_g0(batch_chi, batch_ml, m, rho, field_profile)
                
                for idx, p in enumerate(batch):
                    chi_idx = chi_map[complex(p['chi'])]
                    mode_sums[chi_idx] += (num_results[idx] - num_bg[idx])

        # 4. Renormalization and Spectral Integration
        uv_coeff_local = self.renormalizer.get_b2_term(field_profile, rho) * 2.0
        B_local = getattr(field_profile, 'B_vals', torch.zeros_like(rho)).detach().cpu().numpy()

        from analytic import heisenberg_euler_integrand

        # Consistent Scalar QED EH normalization for Q^3 dQ measure
        # L = (1/8pi^2) * Integral Q^3 dQ * [ Integrand ]
        norm_factor = 1.0 / (8.0 * np.pi**2)

        r_safe = torch.where(rho == 0, torch.tensor(1e-15, device=rho.device), rho)
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
            if lcf_threshold is not None and Q > lcf_threshold:
                lcf_integrand = np.array([heisenberg_euler_integrand(Q, B, m=m, e=e) for B in B_local])
                local_renorm_sum = torch.from_numpy(lcf_integrand).to(self.device).to(torch.complex128)
            else:
                # Renormalized Integrand = sum_{sigma3} (-2 * G_ren / r)
                # where G_ren = G_interacting - G0 - G_WKB_counterterm
                # The factor of 2 accounts for spin summation.
                # G_WKB is B^2/(6Q^4)

                # Counter-term for 2 states is B^2/3/Q^4
                uv_sub = uv_coeff_local / (Q**4 + 1e-15)

                # Mode sum contribution (summed over spin)
                mode_sum_i = mode_sums[i]

                # Integrand for L_eff = - 2 * ( (mode_sum - G0_sum)/r - uv_sub )
                # Mode_sum (already contains G_int - G0).
                # Factor of 2 matches the spectral transform from 2D mode sum to 4D EH.

                local_renorm_sum = - 2.0 * (mode_sums[i].real / r_safe) + uv_sub.real

            # Integral over Q
            L_eff_rho += Q**3 * local_renorm_sum * chi_weights[i] * norm_factor

        # Spatial Integration
        rho_weights = torch.zeros_like(rho)
        rho_weights[1:-1] = (rho[2:] - rho[:-2]) / 2.0
        rho_weights[0] = (rho[1] - rho[0]) / 2.0
        rho_weights[-1] = (rho[-1] - rho[-2]) / 2.0
        
        # Integrated action Gamma (Action per unit time and unit length)
        # EH Lagrangian density normalization: positive for Scalar QED correction
        action = torch.sum(L_eff_rho.real * rho * rho_weights)
        
        return action, L_eff_rho
