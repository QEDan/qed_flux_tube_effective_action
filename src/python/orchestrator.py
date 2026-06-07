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
    def __init__(self, device: Optional[str] = 'cpu', batch_size: int = 1024, strategy: str = "analytic", global_mode: bool = False) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.global_mode = global_mode
        from src.python.pytorch_solver import PyTorchSolver
        from src.python.renormalization import Renormalizer
        self.backend = PyTorchSolver(device=device)
        self.renormalizer = Renormalizer(device=device, strategy=strategy, solver=self.backend, global_mode=global_mode)

    def compute_classical_action(self, field_profile: Any) -> torch.Tensor:
        """
        Computes the classical action (Action per unit time and unit length):
        S_cl = 2*pi * \int rho d_rho  (1/2) * B(rho)^2
        Units: Natural Units (hbar=c=1). B has dimension [L^-2], so S_cl is dimensionless.
        """
        rho, _, _ = field_profile.get_arrays(as_numpy=False)
        rho = rho.to(self.device)
        B = getattr(field_profile, 'B_vals', torch.zeros_like(rho))
        
        # Integration weights (dr)
        rho_weights = torch.zeros_like(rho)
        if len(rho) > 1:
            dr = rho[1:] - rho[:-1]
            rho_weights[1:-1] = (rho[2:] - rho[:-2]) / 2.0
            rho_weights[0] = dr[0] / 2.0
            rho_weights[-1] = dr[-1] / 2.0
        else:
            rho_weights[0] = 1.0
            
        # S_cl = 2*pi * sum(rho * 0.5 * B^2 * dr)
        s_cl = constants.TWO_PI * torch.sum(rho * 0.5 * (B**2) * rho_weights)
        return s_cl

    def compute_total_action(self, field_profile: Any, chi_values: Union[List[complex], torch.Tensor], ml_values: List[int], sigma3_values: List[int], m: float = constants.ELECTRON_MASS, e: float = constants.ELECTRON_CHARGE, lcf_threshold: Optional[float] = 20.0) -> torch.Tensor:
        """
        Computes the total effective action S_tot = S_cl + Gamma_1loop.
        """
        s_cl = self.compute_classical_action(field_profile)
        gamma_1loop, _ = self.compute_effective_action(field_profile, chi_values, ml_values, sigma3_values, m=m, e=e, lcf_threshold=lcf_threshold)
        return s_cl + gamma_1loop

    def compute_effective_action(self, field_profile: Any, chi_values: Union[List[complex], torch.Tensor], ml_values: List[int], sigma3_values: List[int], m: float = constants.ELECTRON_MASS, e: float = constants.ELECTRON_CHARGE, collect_density: bool = False, lcf_threshold: Optional[float] = 20.0) -> Any:
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

        # 1. Prepare chi (Euclidean momentum Q) as a tensor
        if isinstance(chi_values, list):
            # If chi_values are complex, extract Q = |chi|
            chi_t = torch.tensor([abs(complex(c)) for c in chi_values], device=self.device, dtype=torch.float64)
        else:
            chi_t = chi_values.to(self.device).to(torch.float64)

        # 2. Prepare mode sums per chi (Euclidean spectral integration)
        mode_sums = torch.zeros((len(chi_t), n_points), device=self.device, dtype=torch.complex128)

        # We solve the interacting case batch-by-batch
        # Create all_params but keep them as tensors where possible
        ml_t = torch.tensor(ml_values, device=self.device, dtype=torch.float64)
        s3_t = torch.tensor(sigma3_values, device=self.device, dtype=torch.float64)

        for i, Q in enumerate(chi_t):
            if lcf_threshold is not None and Q > lcf_threshold:
                continue

            # Prepare batch for this Q
            # For now, keep the loop over ml/sigma3 if batch_size allows, but ensure differentiability
            # Actually, batching is better for performance.
            batch_params = []
            for ml in ml_values:
                for s3 in sigma3_values:
                    batch_params.append({'chi': 1j * Q, 'ml': ml, 'sigma3': s3, 'm': m, 'e': e})

            if batch_params:
                # Solve interacting case
                num_results, _ = self.backend.solve_batch(batch_params, field_profile)

                # Compute background
                # batch_chi and batch_ml must be tensors for compute_g0
                batch_chi_t = (1j * Q).expand(len(batch_params))
                batch_ml_t = torch.tensor([p['ml'] for p in batch_params], device=self.device, dtype=torch.float64)

                num_bg = self.renormalizer.compute_g0(batch_chi_t, batch_ml_t, m, e, rho, field_profile)

                # Sum results for this Q
                # (num_results - num_bg) shape: (n_batch, n_rho)
                diff = (num_results - num_bg)
                diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
                mode_sums[i] = diff.sum(dim=0)

        # 4. Renormalization and Spectral Integration
        norm_factor = constants.HE_NORMALIZATION_FACTOR
        r_safe = torch.where(rho == 0, torch.tensor(1e-15, device=rho.device), rho)

        # Compute spectral weights (Q^3 dQ) differentiably
        chi_weights = torch.zeros_like(chi_t)
        if len(chi_t) > 1:
            dq = chi_t[1:] - chi_t[:-1]
            # Trapezoidal-like or midpoint weights
            chi_weights[1:-1] = (chi_t[2:] - chi_t[:-2]) / 2.0
            chi_weights[0] = dq[0] / 2.0
            chi_weights[-1] = dq[-1] / 2.0
        else:
            chi_weights[0] = 1.0

        chi_measure = chi_weights * (chi_t ** 3)

        L_eff_rho = torch.zeros(n_points, device=self.device, dtype=torch.complex128)

        for i, Q in enumerate(chi_t):
            if lcf_threshold is not None and Q > lcf_threshold:
                local_renorm_sum = torch.zeros(n_points, device=self.device, dtype=torch.complex128)
            else:
                # Renormalizer returns 2 states (scalar QED term). Double for 4 spinor states.
                # Compute UV subtraction differentiably
                Q_complex = 1j * Q
                uv_sub = self.renormalizer.compute_uv_subtraction(Q_complex.unsqueeze(0), torch.tensor([0.0], device=self.device), m, rho, field_profile).squeeze(0)

                uv_sub_4states = uv_sub * 2.0
                mode_sum_4states = mode_sums[i] * 2.0
                local_renorm_sum = (mode_sum_4states.real / r_safe) + uv_sub_4states.real

            L_eff_rho += local_renorm_sum * chi_measure[i] * norm_factor

        # Spatial Integration weights (2*pi * rho * dr)
        rho_weights = torch.zeros_like(rho)
        if len(rho) > 1:
            dr = rho[1:] - rho[:-1]
            rho_weights[1:-1] = (rho[2:] - rho[:-2]) / 2.0
            rho_weights[0] = dr[0] / 2.0
            rho_weights[-1] = dr[-1] / 2.0
        else:
            rho_weights[0] = 1.0

        # Integrated action Gamma
        action = -constants.TWO_PI * torch.sum(L_eff_rho.real * rho * rho_weights)

        return action, -1.0 * L_eff_rho

