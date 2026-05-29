import numpy as np
import torch
from typing import Union
from src.python import constants
from src.python.profiles import FieldProfile

class Sech2ShellProfile(FieldProfile):
    def __init__(self, rho: Union[np.ndarray, torch.Tensor], R: float, B: float, lambd: float, e: float = constants.ELECTRON_CHARGE) -> None:
        """
        Displaced sech2 flux tube:
        B(rho) = B * sech^2((rho - R) / lambd)
        A_phi(rho) approx (R/rho) * integral(B(r'), r', -inf, rho)
        """
        super().__init__(rho)
        self.R = R
        self.B = B
        self.lambd = lambd
        self.e = e
        self.update()

    def update(self) -> None:
        r_safe = torch.where(self.rho == 0, torch.tensor(1e-15, device=self.rho.device), self.rho)
        arg = (self.rho - self.R) / self.lambd
        
        # B_field(rho) = B * sech^2(arg)
        # sech(x) = 1/cosh(x)
        cosh_arg = torch.cosh(arg)
        # Handle potential overflow in cosh for very large arg
        b_field = self.B / (cosh_arg**2 + 1e-300)
        
        # A_phi(rho) exact integral for cylindrically symmetric B(rho) = B0 * sech^2((rho-R)/lambda)
        # Phi(rho) = 2*pi * integral_0^rho { r' * B(r') dr' }
        # Let x = (r'-R)/lambda
        # Phi(rho) = 2*pi * B0 * lambda * [ R*tanh(x) + lambda*(x*tanh(x) - log(cosh(x))) ] |_{-R/lambda}^{(rho-R)/lambda}
        
        def stable_log_cosh(x):
            abs_x = torch.abs(x)
            return abs_x - torch.log(torch.tensor(2.0, device=x.device)) + torch.log1p(torch.exp(-2.0 * abs_x))

        def f_func(x):
            return self.R * torch.tanh(x) + self.lambd * (x * torch.tanh(x) - stable_log_cosh(x))

        phi = 2.0 * np.pi * self.B * self.lambd * (f_func(arg) - f_func(torch.tensor(-self.R / self.lambd, device=self.rho.device)))
        self.a_phi = phi / (2.0 * np.pi * r_safe)
        
        # da_phi = B - A_phi/rho
        self.da_phi = b_field - self.a_phi / r_safe
