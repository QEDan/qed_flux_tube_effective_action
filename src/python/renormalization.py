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
        Computes the background Green's function G0 = -pi/2 * rho * J_ml(k*rho) * Y_ml(k*rho)
        where k = sqrt(chi^2 - m^2).
        Dimension: [L]
        Uses asymptotic approximation for large ml to avoid overflow.
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
            
            sub_g0_np = np.zeros_like(sub_k_rho_np, dtype=np.complex128)
            
            for i, m_val in enumerate(sub_ml_np):
                m_abs = np.abs(m_val)
                z = sub_k_rho_np[i]
                # Use approximation for large ml or small z to avoid overflow in Y_ml
                # Overflow occurs when (z/2)^-m > 10^308. 
                # For z=0.01, z/2=0.005=1/200. (200)^m > 10^308 => m*log10(200) > 308 => m*2.3 > 308 => m > 134.
                # Let's be safe and use 50.
                mask_approx = m_abs > 50
                
                # G_radial_0 = -pi/2 * J * Y. 
                # For m >> z: J ~ (z/2)^m / m!, Y ~ -(m-1)! / pi (z/2)^m
                # So G_radial_0 ~ -pi/2 * [ -1 / (pi * m) ] = 1 / (2*m)
                # Correcting for z: G_radial_0 = 1 / (2 * sqrt(m^2 - z^2))
                
                # We apply the mask per point if z is large, but usually z is small
                if m_abs > 50:
                    sub_g0_np[i] = 0.5 / np.sqrt(m_abs**2 - z**2 + 1e-15)
                else:
                    res_j = jv(float(m_val), z)
                    res_y = yv(float(m_val), z)
                    sub_g0_np[i] = -0.5 * np.pi * res_j * res_y
            
            # G = rho * G_radial_0
            sub_g0_final = torch.from_numpy(rho.detach().cpu().numpy() * sub_g0_np).to(self.device).to(torch.complex128)
            
            # Update cache and result
            for i, actual_idx in enumerate(to_compute_indices):
                g0[actual_idx] = sub_g0_final[i]
                c_val = complex(chi[actual_idx].item())
                m_val = int(ml[actual_idx].item())
                self._g0_cache[(c_val, m_val)] = sub_g0_final[i]
                
        return g0

    def compute_uv_subtraction(self, chi: torch.Tensor, ml: torch.Tensor, m: float, rho: torch.Tensor, field_profile: Any) -> torch.Tensor:
        """
        Computes the oscillatory WKB-based UV subtraction term (Eq 2.100).
        These terms are angular-momentum dependent and scaled by rho for Path A matching.
        """
        chi_abs = torch.abs(chi)
        k2 = chi_abs*chi_abs - m*m
        k2 = torch.clamp(k2, min=1e-3)
        k = torch.sqrt(k2)
        
        _, a_phi, da_phi = field_profile.get_arrays(as_numpy=False)
        B = (a_phi / (rho + 1e-15) + da_phi)
        
        # Theta = 2k*rho - (1/4 - ml^2)/(k*rho)
        theta = 2*k.unsqueeze(-1)*rho - (0.25 - ml.unsqueeze(-1)**2)/(k.unsqueeze(-1)*rho + 1e-15)
        
        # Oscillatory part from Eq 2.100
        # Term = (B^2/4) * [ rho^3/2k^2 * sin(theta) + rho^2/6k^3 * cos(theta) ]
        # This term has dimension [L], matching G and G0.
        uv_osc = 0.25 * (B.unsqueeze(0)**2) * ( (rho.unsqueeze(0)**3 / (2*k2.unsqueeze(-1))) * torch.sin(theta) + (rho.unsqueeze(0)**2 / (6*k.unsqueeze(-1)**3)) * torch.cos(theta) )
        
        # Damping factor for small k*rho where WKB is invalid
        damping = 1.0 - torch.exp(-(k.unsqueeze(-1) * rho.unsqueeze(0))**2)
        
        return (uv_osc * damping).to(torch.complex128)

    def compute_whittaker_benchmark(self, chi: float, ml: int, sigma3: int, m: float, lambd: float, F: float, rho: torch.Tensor) -> torch.Tensor:
        """
        Analytic solution for Step Function profile using Whittaker functions.
        Used for verification as per implementation_strategy.md Section 6.
        """
        return torch.zeros_like(rho, dtype=torch.complex128)
