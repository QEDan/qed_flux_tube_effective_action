import numpy as np
import torch
from typing import Union, Tuple, Optional, Any, List

class Discontinuity:
    """
    Represents a singular point in the effective potential (e.g., a delta function).
    The solver uses this to apply jump conditions to the wave function derivative.
    """
    def __init__(self, location: float, magnitude: float, is_sigma3_dependent: bool = False):
        self.location = location
        self.magnitude = magnitude
        self.is_sigma3_dependent = is_sigma3_dependent

class FieldProfile:
    def __init__(self, rho: Union[np.ndarray, torch.Tensor]) -> None:
        if isinstance(rho, np.ndarray):
            self.rho: torch.Tensor = torch.from_numpy(rho).to(torch.float64)
        else:
            self.rho = rho
        self.a_phi: torch.Tensor = torch.zeros_like(self.rho)
        self.da_phi: torch.Tensor = torch.zeros_like(self.rho)

    def get_arrays(self, as_numpy: bool = True) -> Tuple[Any, Any, Any]:
        if as_numpy:
            return self.rho.detach().cpu().numpy(), \
                   self.a_phi.detach().cpu().numpy(), \
                   self.da_phi.detach().cpu().numpy()
        return self.rho, self.a_phi, self.da_phi

    def get_discontinuities(self) -> List[Discontinuity]:
        """
        Returns a list of points where the potential contains singular contributions 
        (e.g., delta functions) requiring jump conditions in the ODE solver.
        The solver defaults to assuming the distribution is continuous if this returns an empty list.
        """
        return []

class StepFunctionProfile(FieldProfile):
    def __init__(self, rho: Union[np.ndarray, torch.Tensor], lambd: float, F: float, e: float = 1.0, smooth_width: Optional[float] = None) -> None:
        """
        Magnetic field B = (F / (pi * lambd^2)) * theta(lambd - rho)
        A_phi = (F / (2*pi)) * f_lambda(rho) / rho
        f_lambda(rho) = (rho^2/lambd^2) * theta(lambd - rho) + theta(rho - lambd)
        """
        super().__init__(rho)
        self.lambd = lambd
        self.F = F
        self.e = e
        self.smooth_width = smooth_width
        self.update()

    def update(self) -> None:
        pre = self.F / (2.0 * np.pi)
        
        if self.smooth_width is None:
            # Sharp step function
            # To handle boundary point lambda, use a small eps to avoid discontinuity
            eps = 1e-10
            inner = self.rho < (self.lambd - eps)
            at_boundary = torch.abs(self.rho - self.lambd) < eps
            
            # f_lambda(rho)
            f_lambd = torch.where(inner, (self.rho**2) / (self.lambd**2), torch.ones_like(self.rho))
            
            # A_phi = pre * f_lambda / rho
            self.a_phi = pre * f_lambd / self.rho
            
            # B = pre * 2 / lambd^2 for rho < lambd, else 0. 
            # At boundary, we use the average B_in/2
            b_field = torch.where(inner, 2.0 * pre / (self.lambd**2), 
                                  torch.where(at_boundary, pre / (self.lambd**2), torch.zeros_like(self.rho)))
            
            # da_phi = B - a_phi / rho
            self.da_phi = b_field - self.a_phi / self.rho
        else:
            # Smoothed step function using sigmoid for theta(lambd - rho)
            arg = (self.lambd - self.rho) / self.smooth_width
            theta_smooth = 0.5 * (1.0 + torch.tanh(arg))
            
            # f_lambda_smooth = (rho^2/lambd^2) * theta_smooth + (1 - theta_smooth)
            f_lambd = (self.rho**2 / self.lambd**2) * theta_smooth + (1.0 - theta_smooth)
            
            self.a_phi = pre * f_lambd / self.rho
            
            # Derivative of f_lambda
            d_theta = -0.5 * (1.0 - torch.tanh(arg)**2) / self.smooth_width
            
            df_lambd = (2.0 * self.rho / self.lambd**2) * theta_smooth + \
                       (self.rho**2 / self.lambd**2) * d_theta - d_theta
            
            # da_phi = pre * (df_lambd / rho - f_lambd / rho^2)
            self.da_phi = pre * (df_lambd / self.rho - f_lambd / (self.rho**2))

    def get_discontinuities(self) -> List[Discontinuity]:
        """
        For a sharp step function, there is a delta-function spike in the potential 
        at rho = lambda, leading to a jump in u'.
        """
        if self.smooth_width is not None or self.lambd <= 0 or abs(self.F) < 1e-12:
            return []
        # Jump = -e * F / (pi * lambda^2)
        return [Discontinuity(location=self.lambd, magnitude=-self.e * self.F / (np.pi * self.lambd**2))]

class ZeroFluxProfile(FieldProfile):
    def __init__(self, rho: Union[np.ndarray, torch.Tensor], B: float, lambd: float, e: float = 1.0) -> None:
        """
        Magnetic field B(rho) = B * (1 - 2*rho^2/lambd^2) for rho < lambd, else 0.
        A_phi(rho) = (B*rho/2) * (1 - rho^2/lambd^2) for rho < lambd, else 0.
        Total flux F = 0.
        """
        super().__init__(rho)
        self.B = B
        self.lambd = lambd
        self.e = e
        self.update()

    def update(self) -> None:
        r_safe = torch.where(self.rho == 0, torch.tensor(1e-15, device=self.rho.device), self.rho)
        inner = self.rho < self.lambd
        
        # B_field(rho)
        b_field = torch.where(inner, self.B * (1.0 - 2.0 * self.rho**2 / self.lambd**2), torch.zeros_like(self.rho))
        
        # A_phi(rho)
        self.a_phi = torch.where(inner, (self.B * self.rho / 2.0) * (1.0 - self.rho**2 / self.lambd**2), torch.zeros_like(self.rho))
        
        # da_phi = B - A_phi/rho
        self.da_phi = b_field - self.a_phi / r_safe

class SuperGaussianProfile(FieldProfile):
    def __init__(self, rho: Union[np.ndarray, torch.Tensor], B0: float, lambd: float, e: float = 1.0) -> None:
        """
        Magnetic field B(rho) = B0 * exp(-(rho/lambd)^4).
        A_phi(rho) = (sqrt(pi) * B0 * lambd^2 / (4 * rho)) * erf((rho/lambd)^2).
        """
        super().__init__(rho)
        self.B0 = B0
        self.lambd = lambd
        self.e = e
        self.update()

    def update(self) -> None:
        r_safe = torch.where(self.rho == 0, torch.tensor(1e-15, device=self.rho.device), self.rho)
        
        # B_field(rho)
        b_field = self.B0 * torch.exp(-(self.rho / self.lambd)**4)
        
        # A_phi(rho) using torch.erf
        # Integral_0^rho r * exp(-(r/lambda)^4) dr = (lambda^2 * sqrt(pi) / 4) * erf((rho/lambda)^2)
        pre = (np.sqrt(np.pi) * self.B0 * self.lambd**2) / 4.0
        self.a_phi = pre * torch.erf((self.rho / self.lambd)**2) / r_safe
        
        # da_phi = B - A_phi/rho
        self.da_phi = b_field - self.a_phi / r_safe

class MLPProfile(FieldProfile):
    def __init__(self, rho: torch.Tensor, B_vals: torch.Tensor, a_phi: torch.Tensor, e: float = 1.0) -> None:
        super().__init__(rho)
        self.B_vals = B_vals
        self.a_phi = a_phi
        self.e = e
        self.update()

    def update(self) -> None:
        r_safe = torch.where(self.rho == 0, torch.tensor(1e-15, device=self.rho.device), self.rho)
        # da_phi = B - a_phi / rho
        self.da_phi = self.B_vals - self.a_phi / r_safe
