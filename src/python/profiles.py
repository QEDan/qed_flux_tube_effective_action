import numpy as np
import torch
from typing import Union, Tuple, Optional, Any

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
            inner = self.rho < self.lambd
            
            # f_lambda(rho)
            f_lambd = torch.where(inner, (self.rho**2) / (self.lambd**2), torch.ones_like(self.rho))
            
            # A_phi = pre * f_lambda / rho
            self.a_phi = pre * f_lambd / self.rho
            
            # B = pre * 2 / lambd^2 for rho < lambd, else 0
            b_field = torch.where(inner, 2.0 * pre / (self.lambd**2), torch.zeros_like(self.rho))
            
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

class Sech2Profile(FieldProfile):
    def __init__(self, rho: Union[np.ndarray, torch.Tensor], B: float, lambd: float, e: float = 1.0) -> None:
        """
        Magnetic field B(rho) = B * sech^2(rho/lambd)
        A_phi(rho) = B*lambd*tanh(rho/lambd) - (B*lambd^2/rho)*ln(cosh(rho/lambd))
        """
        super().__init__(rho)
        self.B = B
        self.lambd = lambd
        self.e = e
        self.update()

    def update(self) -> None:
        # Avoid division by zero at rho=0
        r_safe = torch.where(self.rho == 0, torch.tensor(1e-15, device=self.rho.device), self.rho)
        
        arg = self.rho / self.lambd
        
        # A_phi(rho)
        self.a_phi = self.B * self.lambd * torch.tanh(arg) - (self.B * self.lambd**2 / r_safe) * torch.log(torch.cosh(arg))
        
        # B_field(rho)
        b_field = self.B / (torch.cosh(arg)**2)
        
        # da_phi = B - A_phi/rho
        self.da_phi = b_field - self.a_phi / r_safe
