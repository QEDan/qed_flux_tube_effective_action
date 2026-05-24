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
        b_field = self.B / (torch.cosh(arg)**2)
        
        # A_phi(rho) approximation for R >> lambd
        # A_phi approx (R/rho) * B * lambd * (tanh(arg) + 1)
        self.a_phi = (self.R / r_safe) * self.B * self.lambd * (torch.tanh(arg) + 1.0)
        
        # da_phi = B - A_phi/rho
        self.da_phi = b_field - self.a_phi / r_safe
