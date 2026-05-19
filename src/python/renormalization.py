import torch
import numpy as np
from scipy.special import jv, yv
from typing import Union, Dict, Tuple, Any, Optional, List

class Renormalizer:
    """
    Handles vacuum subtraction and UV renormalization terms for the effective action.
    This class provides the background Green's function (G0) and the counter-terms
    needed to ensure the convergence of the spectral integral over chi.
    """
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._g0_cache: Dict[Tuple[complex, int], torch.Tensor] = {}

    def compute_g0_local(self, chi: torch.Tensor, ml: torch.Tensor, m: float, rho: torch.Tensor, field_profile: Any) -> torch.Tensor:
        """
        Computes the topological vacuum Green's function G0_ml(rho, rho) 
        matching the local vector potential A(rho).
        """
        _, a_phi, _ = field_profile.get_arrays(as_numpy=False)
        e = 1.0
        
        # k = sqrt(chi^2 - m^2)
        k2 = chi*chi - m*m
        
        n_batch = len(chi)
        n_points = len(rho)
        
        # n_local: (n_batch, n_points)
        n_local = ml.unsqueeze(-1) - e * (a_phi * rho).unsqueeze(0)
        
        # Convert to numpy for scipy
        n_np = n_local.detach().cpu().numpy()
        k2_np = k2.detach().cpu().numpy()
        rho_np = rho.detach().cpu().numpy()
        
        res_np = np.zeros_like(n_np, dtype=np.complex128)
        
        from scipy.special import iv, kv, ive, kve, jv, yv
        
        # Vectorized calculation
        for i in range(n_batch):
            order = n_np[i]
            k2_val = k2_np[i]
            
            if k2_val.real < 0:
                # EUCLIDEAN: k = i * kappa
                kappa = np.sqrt(-k2_val.real)
                z = kappa * rho_np
                
                # Use modified Bessel I, K for all regions.
                # Exponentially scaled versions (ive, kve) are numerically stable for large order/z.
                res_np[i] = - ive(order, z) * kve(order, z)
            else:
                # MINKOWSKI: k^2 > 0
                k = np.sqrt(k2_val)
                z = k * rho_np
                
                # Minkowski: G_radial = - 0.5 * pi * i * H_n^{(1)}(z) * J_n(z)
                # Correct sign for matching interacting Green's function (positive).
                from scipy.special import hankel1
                
                res_np[i] = - 0.5 * np.pi * 1j * jv(order, z) * hankel1(order, z)
                
                # Mask out singularities
                mask_sing = (z < 1e-5)
                if np.any(mask_sing):
                    # Asymptotic for n=0: G ~ log(z) (divergent)
                    # For n>0: G ~ -1/(2*n)
                    # For simplicity, avoid singularity by masking if rho is too small
                    # or returning 0 at r=0.
                    res_np[i][mask_sing] = 0.0

        # G = rho * G_radial
        g0 = torch.from_numpy(rho_np * res_np).to(self.device).to(torch.complex128)
        return g0
    def compute_g0(self, chi: torch.Tensor, ml: torch.Tensor, m: float, rho: torch.Tensor, field_profile: Optional[Any] = None) -> torch.Tensor:
        """Alias for compute_g0_local for backward compatibility."""
        from src.python.profiles import FieldProfile
        if field_profile is None:
            field_profile = FieldProfile(rho)
        return self.compute_g0_local(chi, ml, m, rho, field_profile)

    def compute_uv_subtraction(self, chi: torch.Tensor, ml: torch.Tensor, m: float, rho: torch.Tensor, field_profile: Any) -> torch.Tensor:
        """
        Computes the UV subtraction term. 
        """
        _, a_phi, da_phi = field_profile.get_arrays(as_numpy=False)
        r_safe = torch.where(rho == 0, torch.tensor(1e-15, device=rho.device), rho)
        B = (a_phi / r_safe + da_phi)
        return torch.zeros((len(chi), len(rho)), device=self.device, dtype=torch.complex128)

    def get_b2_term(self, field_profile: Any, rho: torch.Tensor) -> torch.Tensor:
        """
        Returns (eB)^2 / 6.0 density.
        Matches Scalar QED field strength renormalization term.
        """
        _, a_phi, da_phi = field_profile.get_arrays(as_numpy=False)
        r_safe = torch.where(rho == 0, torch.tensor(1e-15, device=rho.device), rho)
        B = (a_phi / r_safe + da_phi)
        return (B**2 / 6.0).to(torch.complex128)

    def compute_whittaker_benchmark(self, chi: complex, ml: int, sigma3: int, m: float, lambd: float, F: float, rho: torch.Tensor) -> torch.Tensor:
        """
        Analytic solution for Step Function profile using Whittaker functions.
        Matches Eq 2.76 / 2.100 of greensfunc.tex.
        Dimension: [L] (rho-scaled Green's function)
        """
        from analytic import M_whittaker, W_whittaker
        import mpmath
        
        e = 1.0
        F_cal = (e * F) / (2.0 * np.pi)
        k2 = chi*chi - m*m
        
        # kappa = lambda^2 * k^2 / (4 * F_cal) + (ml - sigma3)/2
        # mu = ml / 2
        # z = (F_cal / lambda^2) * rho^2
        kappa = (lambd**2 * k2) / (4.0 * F_cal)
        mu = abs(float(ml)) / 2.0
        
        rho_np = rho.detach().cpu().numpy()
        z = (F_cal / lambd**2) * (rho_np**2)
        
        # G_radial = (lambda^2 / 2*F_cal) * (Gamma(1/2 + mu - kappa) / ml!) * W * M
        # We need to handle the Gamma function carefully for poles
        gamma_factor = complex(mpmath.gamma(0.5 + mu - kappa))
        denom = float(mpmath.factorial(abs(ml)))
        
        # Solve for W and M
        W_vals = W_whittaker(z, kappa, mu)
        M_vals = M_whittaker(z, kappa, mu)
        
        g_radial = - (lambd**2 / (2.0 * F_cal)) * (gamma_factor / denom) * W_vals * M_vals
        
        # Result is rho * g_radial
        return torch.from_numpy(rho_np * g_radial).to(self.device).to(torch.complex128)
