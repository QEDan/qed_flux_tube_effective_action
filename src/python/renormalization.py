import torch
import numpy as np
from scipy.special import jv, yv
from typing import Union, Dict, Tuple, Any, Optional, List

class Renormalizer:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._g0_cache: Dict[Tuple[complex, int], torch.Tensor] = {}

    def compute_g0(self, chi: torch.Tensor, ml: torch.Tensor, m: float, rho: torch.Tensor) -> torch.Tensor:
        """
        Computes the background Green's function G0 = -pi/2 * J_ml(k*rho) * Y_ml(k*rho)
        where k = sqrt(chi^2 - m^2).
        Dimensionless.
        """
        k2 = chi*chi - m*m
        k2 = torch.where(torch.abs(k2) < 1e-12, torch.tensor(1e-12, dtype=torch.complex128), k2)
        k = torch.sqrt(k2).to(torch.complex128)

        n_points = len(rho)
        n_batch = len(chi)
        
        g0 = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.complex128)
        
        # Identify unique (chi, ml) pairs to compute
        to_compute_indices: List[int] = []
        for i in range(n_batch):
            c_val = complex(chi[i].item())
            m_val = int(ml[i].item())
            key = (c_val, m_val)
            if key in self._g0_cache:
                g0[i] = self._g0_cache[key]
            else:
                to_compute_indices.append(i)
                
        if to_compute_indices:
            idx = torch.tensor(to_compute_indices, device=self.device)
            sub_k = k[idx]
            sub_ml = ml[idx]
            
            sub_k_rho = sub_k.unsqueeze(-1) * rho.unsqueeze(0)
            sub_k_rho_np = sub_k_rho.detach().cpu().numpy()
            sub_ml_np = sub_ml.detach().cpu().numpy()
            
            res_j = np.zeros_like(sub_k_rho_np, dtype=np.complex128)
            res_y = np.zeros_like(sub_k_rho_np, dtype=np.complex128)
            
            for i, m_val in enumerate(sub_ml_np):
                res_j[i] = jv(float(m_val), sub_k_rho_np[i])
                res_y[i] = yv(float(m_val), sub_k_rho_np[i])
                
            sub_g0_np = -0.5 * np.pi * rho.detach().cpu().numpy() * res_j * res_y
            sub_g0 = torch.from_numpy(sub_g0_np).to(self.device).to(torch.complex128)
            
            # Update cache and result
            for i, actual_idx in enumerate(to_compute_indices):
                g0[actual_idx] = sub_g0[i]
                c_val = complex(chi[actual_idx].item())
                m_val = int(ml[actual_idx].item())
                self._g0_cache[(c_val, m_val)] = sub_g0[i]
                
        return g0

    def compute_uv_subtraction(self, chi: torch.Tensor, ml: torch.Tensor, m: float, rho: torch.Tensor, field_profile: Any) -> torch.Tensor:
        """
        Computes the WKB-based UV subtraction term based on Eq 2.59 of greensfunc.tex.
        """
        k2 = chi*chi - m*m
        k2 = torch.where(torch.abs(k2) < 1e-12, torch.tensor(1e-12, dtype=torch.complex128), k2)
        k = torch.sqrt(k2)
        
        _, a_phi, da_phi = field_profile.get_arrays(as_numpy=False)
        B = (a_phi / (rho + 1e-15) + da_phi)
        
        theta = 2*k.unsqueeze(-1)*rho - (0.25 - ml.unsqueeze(-1)**2)/(k.unsqueeze(-1)*rho + 1e-15)
        
        # B^2/k^2 is the field-dependent scaling factor.
        # Theoretical UV term should be proportional to B^2 (or equivalent F^2).
        # We use B as proxy for the field strength.
        uv_sub = (B.unsqueeze(0)**2) * ( (rho**3 / (2*k2.unsqueeze(-1))) * torch.sin(theta) + (rho**2 / (6*k.unsqueeze(-1)**3)) * torch.cos(theta) )
        
        return uv_sub.to(torch.complex128)

    def compute_whittaker_benchmark(self, chi: float, ml: int, sigma3: int, m: float, lambd: float, F: float, rho: torch.Tensor) -> torch.Tensor:
        """
        Analytic solution for Step Function profile using Whittaker functions.
        Used for verification as per implementation_strategy.md Section 6.
        """
        e = 1.0
        F_dim = (e * F) / (2 * np.pi)
        k2 = chi*chi + m*m - (2.0 * F_dim / (lambd*lambd)) * (sigma3 - ml)
        
        rho_np = rho.detach().cpu().numpy()
        res = np.zeros_like(rho_np, dtype=np.complex128)
        
        # Interior region (rho < lambd)
        # Placeholder for exact Whittaker implementation
        return torch.from_numpy(res).to(self.device)
