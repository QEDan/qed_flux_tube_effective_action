from src.python import constants
import torch
import numpy as np
from scipy.special import jv, yv
from typing import Union, Dict, Tuple, Any, Optional, List

import torch
import numpy as np
from scipy.special import jv, yv
from typing import Union, Dict, Tuple, Any, Optional, List
from abc import ABC, abstractmethod

class BackgroundStrategy(ABC):
    @abstractmethod
    def compute_g0(self, chi: torch.Tensor, ml: torch.Tensor, m: float, e: float, rho: torch.Tensor, field_profile: Any, global_mode: bool = False) -> torch.Tensor:
        pass

class AnalyticBackgroundStrategy(BackgroundStrategy):
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

    def compute_g0(self, chi: torch.Tensor, ml: torch.Tensor, m: float, e: float, rho: torch.Tensor, field_profile: Any, global_mode: bool = False) -> torch.Tensor:
        from src.python import torch_special
        _, a_phi, _ = field_profile.get_arrays(as_numpy=False)
        k2 = chi*chi - m*m
        n_batch = len(ml)
        n_points = len(rho)
        
        # Ensure k2 is broadcasted to (n_batch,)
        if k2.ndim == 0:
            k2 = k2.expand(n_batch)
        elif len(k2) == 1 and n_batch > 1:
            k2 = k2.expand(n_batch)
        
        if global_mode:
            # Match only asymptotic flux (standard vacuum subtraction)
            F = getattr(field_profile, 'F', 0.0)
            F_cal = e * F / (constants.TWO_PI)
            lambd = getattr(field_profile, 'lambd', rho[-1])
            nu_asym = torch.where(rho < lambd, torch.zeros_like(rho), torch.ones_like(rho) * F_cal)
            n_local = torch.abs(ml.unsqueeze(-1) - nu_asym.unsqueeze(0))
        else:
            # Match local A_phi exactly (Topological vacuum subtraction)
            n_local = torch.abs(ml.unsqueeze(-1) - e * (a_phi * rho).unsqueeze(0))

        # Vectorized Euclidean Path (k2 < 0)
        is_eucl = (k2.real <= 0)
        # Use safe k2 for each branch to avoid NaN gradients from inactive branch
        k2_eucl = torch.where(is_eucl, k2, torch.tensor(-1.0, dtype=k2.dtype, device=k2.device))
        kappa = torch.sqrt(-k2_eucl.real)
        z = kappa.unsqueeze(-1) * rho.unsqueeze(0)
        
        ik_prod = torch_special.bessel_i_k_product(n_local, z)
        res_euclidean = -ik_prod

        # Vectorized Minkowski Path (k2 > 0)
        k2_mink = torch.where(~is_eucl, k2, torch.tensor(1.0, dtype=k2.dtype, device=k2.device))
        k = torch.sqrt(k2_mink.real)
        zm = k.unsqueeze(-1) * rho.unsqueeze(0)
        
        j_nu = torch_special.bessel_jv(n_local, zm)
        y_nu = torch_special.bessel_yv(n_local, zm)
        
        res_minkowski = 0.5 * constants.PI * j_nu * y_nu - 0.5 * constants.PI * 1j * j_nu**2
        
        # Select based on sign of k2
        res = torch.where(is_eucl.unsqueeze(-1), res_euclidean.to(torch.complex128), res_minkowski.to(torch.complex128))
        
        # Multiply by rho and ensure origin is zeroed
        final_res = rho.unsqueeze(0) * res
        if rho[0] == 0:
            final_res[:, 0] = 0.0
            
        return final_res

class NumericalBackgroundStrategy(BackgroundStrategy):
    def __init__(self, solver: Any, device: str = "cpu") -> None:
        self.solver = solver
        self.device = torch.device(device)

    def compute_g0(self, chi: torch.Tensor, ml: torch.Tensor, m: float, e: float, rho: torch.Tensor, field_profile: Any, global_mode: bool = False) -> torch.Tensor:
        from src.python.profiles import LocalBackgroundProfile, PureGaugeProfile
        if global_mode:
            # Match only asymptotic flux
            F = getattr(field_profile, 'F', 0.0)
            lambd = getattr(field_profile, 'lambd', rho[-1])
            # Create a profile that is zero in interior and pure gauge in exterior
            # For simplicity, we use PureGaugeProfile and zero it out in interior
            bg_profile = PureGaugeProfile(rho, F, e=e)
            inner = rho < lambd
            bg_profile.a_phi[inner] = 0.0
            bg_profile.da_phi[inner] = 0.0
        else:
            # Use LocalBackgroundProfile to match local A_phi exactly
            bg_profile = LocalBackgroundProfile(field_profile)
        
        batch = []
        for i in range(len(chi)):
            # Background is spin-independent (B=0), but we must provide a sigma3 for the solver
            batch.append({'chi': chi[i].item(), 'ml': ml[i].item(), 'sigma3': 1, 'm': m, 'e': e})
        results, _ = self.solver.solve_batch(batch, bg_profile)
        return results

class Renormalizer:
    def __init__(self, device: str = "cpu", strategy: str = "analytic", solver: Optional[Any] = None, global_mode: bool = False) -> None:
        self.device = torch.device(device)
        self.global_mode = global_mode
        self._g0_cache: Dict[Tuple[complex, int], torch.Tensor] = {}
        
        if strategy == "numerical":
            if solver is None:
                 raise ValueError("Solver required for numerical renormalization strategy.")
            self.strategy = NumericalBackgroundStrategy(solver, device=device)
        else:
            self.strategy = AnalyticBackgroundStrategy(device=device)

    def compute_g0(self, chi: torch.Tensor, ml: torch.Tensor, m: float, e: float, rho: torch.Tensor, field_profile: Any) -> torch.Tensor:
        return self.strategy.compute_g0(chi, ml, m, e, rho, field_profile, global_mode=self.global_mode)



    def compute_uv_subtraction(self, chi: torch.Tensor, ml: torch.Tensor, m: float, rho: torch.Tensor, field_profile: Any) -> torch.Tensor:
        """
        Computes the robust massive UV subtraction term: b2 / (Q^2 + m^2)^2.
        This matches the proper-time structure and is finite at Q=0.
        Returns tensor of shape (n_batch, n_rho).
        """
        n_batch = len(ml)
        n_rho = len(rho)
        b2 = self.get_b2_term(field_profile, rho, e=constants.ELECTRON_CHARGE) # (n_rho,)
        
        # Q is i*chi? No, Orchestrator passes chi which is already Euclidean iQ in some places?
        # Actually Orchestrator uses chi_real = abs(complex(c)). 
        # Let's assume chi is real Euclidean Q here.
        Q = torch.abs(chi).to(self.device).to(torch.float64) # (n_batch,)
        
        # Subtraction term: - b2 / (Q^2 + m^2)^2
        # Use broadcasting: b2 is (n_rho), Q is (n_batch)
        denom = (Q.unsqueeze(-1)**2 + m**2)**2
        res = - b2.unsqueeze(0) / denom
        return res.to(torch.complex128)

    def get_b2_term(self, field_profile: Any, rho: torch.Tensor, e: float = constants.ELECTRON_CHARGE) -> torch.Tensor:
        """
        Returns (eB)^2 / 6.0 density.
        Matches Scalar QED field strength renormalization term.
        """
        _, a_phi, da_phi = field_profile.get_arrays(as_numpy=False)
        r_safe = torch.where(rho == 0, torch.tensor(1e-15, device=rho.device), rho)
        B_phys = (a_phi / r_safe + da_phi)
        eB = e * B_phys
        return (eB**2 / 6.0).to(torch.complex128)

    def compute_whittaker_benchmark(self, chi: complex, ml: int, sigma3: int, m: float, lambd: float, F: float, rho: torch.Tensor) -> torch.Tensor:
        """
        Analytic solution for Step Function profile using vectorized PyTorch Whittaker functions.
        Dimension: [L] (rho-scaled Green's function)
        """
        from src.python import torch_special
        
        chi_t = torch.as_tensor(chi, dtype=torch.complex128, device=self.device)
        e = 1.0
        F_cal = (e * F) / (constants.TWO_PI)
        k2 = chi_t*chi_t - m*m
        
        kappa = (lambd**2 * k2) / (4.0 * F_cal)
        mu = abs(float(ml)) / 2.0
        
        z = (F_cal / lambd**2) * (rho**2)
        
        # G_radial = - (lambd^2 / 2*F_cal) * (Gamma(1/2 + mu - kappa) / ml!) * W * M
        log_abs_M, sign_M = torch_special.whittaker_m_log(kappa, mu, z)
        log_abs_W, sign_W = torch_special.whittaker_w_log(kappa, mu, z)
        
        # log(Gamma(0.5 + mu - kappa))
        gamma_arg = 0.5 + mu - kappa
        log_gamma_c = torch_special._lgamma_torch(gamma_arg)
        
        # Normalization factor
        log_pre = torch.log(torch.as_tensor(lambd**2 / (2.0 * F_cal), device=self.device, dtype=torch.float64))
        log_denom = torch.lgamma(torch.as_tensor(float(abs(ml) + 1), device=self.device, dtype=torch.float64))
        
        # Total log Green's function (complex)
        # G = - exp(log_pre + log_gamma_c - log_denom + log_abs_M + log_abs_W) * sign_M * sign_W
        log_G_mag = log_pre + log_gamma_c.real - log_denom + log_abs_M + log_abs_W
        phase_G = log_gamma_c.imag + torch.where(sign_M * sign_W < 0, torch.pi, 0.0) + torch.pi # +pi for the minus sign
        
        G_c = torch.polar(torch.exp(log_G_mag), phase_G)
        
        return rho * G_c
