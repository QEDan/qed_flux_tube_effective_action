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
    def compute_g0(self, chi: torch.Tensor, ml: torch.Tensor, m: float, rho: torch.Tensor, field_profile: Any) -> torch.Tensor:
        pass

class AnalyticBackgroundStrategy(BackgroundStrategy):
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

    def compute_g0(self, chi: torch.Tensor, ml: torch.Tensor, m: float, rho: torch.Tensor, field_profile: Any) -> torch.Tensor:
        _, a_phi, _ = field_profile.get_arrays(as_numpy=False)
        e = 1.0
        k2 = chi*chi - m*m
        n_batch = len(ml)
        n_points = len(rho)
        # Centrifugal order must be absolute value: sqrt((ml - e*A*rho)^2)
        # Handle broadcasting: if chi has length 1, repeat it to match ml
        if len(chi) == 1 and n_batch > 1:
            k2 = k2.repeat(n_batch)
        
        n_local = torch.abs(ml.unsqueeze(-1) - e * (a_phi * rho).unsqueeze(0))
        n_np = n_local.detach().cpu().numpy()
        k2_np = k2.detach().cpu().numpy()
        rho_np = rho.detach().cpu().numpy()
        res_np = np.zeros_like(n_np, dtype=np.complex128)
        from scipy.special import iv, kv, ive, kve, jv, yv, hankel1
        for i in range(n_batch):
            order = n_np[i]
            k2_val = k2_np[i]
            if k2_val.real < 0:
                kappa = np.sqrt(-k2_val.real)
                z = kappa * rho_np
                denom = np.sqrt(order**2 + z**2 + 1e-15)
                res_np[i] = - 0.5 / denom
                mask_asym = (order**2 + z**2 > 100.0)
                mask_reg = ~mask_asym
                if np.any(mask_reg):
                    res_np[i][mask_reg] = - ive(order[mask_reg], z[mask_reg]) * kve(order[mask_reg], z[mask_reg]) * np.exp(z[mask_reg]) * np.exp(-z[mask_reg])
                    # Corrected to -I_nu * K_nu
                    res_np[i][mask_reg] = - iv(order[mask_reg], z[mask_reg]) * kv(order[mask_reg], z[mask_reg])
                
                # Zero out singular points at r=0 for order > 0
                mask_zero = (z < 1e-15) & (order > 0)
                if np.any(mask_zero):
                    res_np[i][mask_zero] = 0.0
            else:
                k = np.sqrt(k2_val)
                z = k * rho_np
                # G_bg = - 0.5 * pi * 1j * J_nu(z) * H_nu^{(1)}(z)
                # = - 0.5 * pi * 1j * J_nu(z) * (J_nu(z) + 1j * Y_nu(z))
                # = - 0.5 * pi * 1j * J_nu^2(z) + 0.5 * pi * J_nu(z) * Y_nu(z)
                # The real part is 0.5 * pi * J_nu(z) * Y_nu(z)
                # The imaginary part is -0.5 * pi * J_nu^2(z)
                res_np[i] = 0.5 * constants.PI * jv(order, z) * yv(order, z) - 0.5 * constants.PI * 1j * jv(order, z)**2
                mask_sing = (z < 1e-15)
                if np.any(mask_sing):
                    res_np[i][mask_sing] = 0.0
        
        # Multiply by rho and ensure origin is zeroed to avoid 0 * inf = NaN
        final_res_np = rho_np * res_np
        if rho_np[0] == 0:
            final_res_np[:, 0] = 0.0
        return torch.from_numpy(final_res_np).to(self.device).to(torch.complex128)

class NumericalBackgroundStrategy(BackgroundStrategy):
    def __init__(self, solver: Any, device: str = "cpu") -> None:
        self.solver = solver
        self.device = torch.device(device)

    def compute_g0(self, chi: torch.Tensor, ml: torch.Tensor, m: float, rho: torch.Tensor, field_profile: Any) -> torch.Tensor:
        from src.python.profiles import LocalBackgroundProfile
        # Use LocalBackgroundProfile to match local A_phi exactly
        bg_profile = LocalBackgroundProfile(field_profile)
        batch = []
        for i in range(len(chi)):
            # Background is spin-independent (B=0), but we must provide a sigma3 for the solver
            batch.append({'chi': chi[i].item(), 'ml': ml[i].item(), 'sigma3': 1, 'm': m, 'e': 1.0})
        results, _ = self.solver.solve_batch(batch, bg_profile)
        return results

class Renormalizer:
    def __init__(self, device: str = "cpu", strategy: str = "analytic", solver: Optional[Any] = None) -> None:
        self.device = torch.device(device)
        self._g0_cache: Dict[Tuple[complex, int], torch.Tensor] = {}
        
        if strategy == "numerical":
            if solver is None:
                 raise ValueError("Solver required for numerical renormalization strategy.")
            self.strategy = NumericalBackgroundStrategy(solver, device=device)
        else:
            self.strategy = AnalyticBackgroundStrategy(device=device)

    def compute_g0(self, chi: torch.Tensor, ml: torch.Tensor, m: float, rho: torch.Tensor, field_profile: Any) -> torch.Tensor:
        return self.strategy.compute_g0(chi, ml, m, rho, field_profile)


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
        from src.python.analytic_step_profile import M_whittaker, W_whittaker
        import mpmath
        
        e = 1.0
        F_cal = (e * F) / (constants.TWO_PI)
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
