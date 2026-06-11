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
        Normalization aligns with the 4D spectral density for 4 spinor states.
        """
        rho, _, _ = field_profile.get_arrays(as_numpy=False)
        rho = rho.to(self.device)
        n_points = len(rho)

        # 1. Prepare chi (Euclidean momentum Q) as a tensor
        if isinstance(chi_values, list):
            chi_t = torch.tensor([abs(complex(c)) for c in chi_values], device=self.device, dtype=torch.float64)
        else:
            chi_t = chi_values.to(self.device).to(torch.float64)

        # 2. Prepare mode sums per chi
        mode_sums = torch.zeros((len(chi_t), n_points), device=self.device, dtype=torch.complex128)
        r_safe = torch.where(rho == 0, torch.tensor(1e-15, device=rho.device), rho)

        for i, Q in enumerate(chi_t):
            if lcf_threshold is not None and Q > lcf_threshold:
                continue

            batch_params = []
            for ml in ml_values:
                for s3 in sigma3_values:
                    batch_params.append({'chi': 1j * Q, 'ml': ml, 'sigma3': s3, 'm': m, 'e': e})

            if batch_params:
                num_results, _ = self.backend.solve_batch(batch_params, field_profile)
                batch_chi_t = (1j * Q).expand(len(batch_params))
                batch_ml_t = torch.tensor([float(p['ml']) for p in batch_params], device=self.device, dtype=torch.float64)
                num_bg = self.renormalizer.compute_g0(batch_chi_t, batch_ml_t, m, e, rho, field_profile)

                diff = (num_results - num_bg)
                diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
                mode_sums[i] = diff.sum(dim=0)

        # 4. Renormalization and Spectral Integration
        # norm_factor consistent with 4D HE density (4 states)
        norm_factor = 1.0 / (4.0 * constants.PI**2)
        
        # Spectral measure: (S^3 - m^2 * S) dS = Q_parallel^3 dQ_parallel
        S_t = torch.sqrt(chi_t**2 + m**2)
        S_weights = torch.zeros_like(S_t)
        if len(S_t) > 1:
            ds = S_t[1:] - S_t[:-1]
            S_weights[1:-1] = (S_t[2:] - S_t[:-2]) / 2.0
            S_weights[0] = ds[0] / 2.0
            S_weights[-1] = ds[-1] / 2.0
        else:
            S_weights[0] = 1.0
        chi_measure = S_weights * (S_t**3 - (m**2) * S_t)

        L_eff_rho = torch.zeros(n_points, device=self.device, dtype=torch.complex128)

        for i, Q in enumerate(chi_t):
            S = S_t[i]
            if lcf_threshold is not None and S > lcf_threshold:
                # Use LCF (Heisenberg-Euler) spectral density for the tail
                _, _, _ = field_profile.get_arrays(as_numpy=False)
                r_safe_val = torch.where(rho == 0, torch.tensor(1e-15, device=rho.device), rho)
                B_phys = (field_profile.a_phi / r_safe_val + field_profile.da_phi)
                
                # HE expansion part (f_he is positive for fermions)
                eBS2 = e * B_phys / (S**2)
                f_he = torch.where(torch.abs(eBS2) < 1e-3,
                                   (eBS2**4) / 45.0, # Positive!
                                   1.0 + (eBS2**2 / 3.0) - eBS2 / torch.tanh(eBS2)) # Negated to match LCF
                
                # Wait! Standard HE f(x) = x coth x - 1 - x2/3 is POSITIVE for fermions?
                # No, f(x) = x/tanh x - 1 - x2/3 is NEGATIVE for small x!
                # So -f(x) is POSITIVE.
                # Let's match the analytic benchmark which uses -h.real.
                local_renorm_sum = f_he.to(torch.complex128)
            else:
                Q_complex = 1j * Q
                uv_sub = self.renormalizer.compute_uv_subtraction(Q_complex.unsqueeze(0), torch.tensor([0.0], device=self.device), m, e, rho, field_profile).squeeze(0)
                
                # mode_sums has 2 spins. HE is for 4 states. Double it.
                # Subtract to get the finite part (G - G0 + uv)
                # If mode_sums (diff) is positive, and uv is negative, they cancel.
                # For fermions, G - G0 + uv should be NEGATIVE? 
                # Analytic benchmark uses -h.real. So h.real must be negative.
                # mode_sums (diff) = results - bg. 
                # If V > 0, G is less negative, so G - G0 is positive.
                # So mode_sums is positive.
                # Then L ~ mode_sums * norm.
                # To match HE > 0, we need a minus sign?
                # Wait, dissertation Eq 2.45: L = 1/pi * Integral Q^3 mode_sum.
                # So if mode_sum is positive, L is positive.
                
                local_renorm_sum = (mode_sums[i].real / r_safe) + uv_sub.real

            L_eff_rho += local_renorm_sum * chi_measure[i] * norm_factor

        # Spatial Integration weights
        rho_weights = torch.zeros_like(rho)
        if len(rho) > 1:
            dr = rho[1:] - rho[:-1]
            rho_weights[1:-1] = (rho[2:] - rho[:-2]) / 2.0
            rho_weights[0] = dr[0] / 2.0
            rho_weights[-1] = dr[-1] / 2.0
        else:
            rho_weights[0] = 1.0

        # L_eff_rho is the intensive Lagrangian density.
        # Negate to match convention (positive for fermions, matching analytic benchmark)
        L_eff_rho = -1.0 * L_eff_rho

        # action = 2*pi * integral(rho * L)
        action = constants.TWO_PI * torch.sum(L_eff_rho.real * rho * rho_weights)

        return action, L_eff_rho
