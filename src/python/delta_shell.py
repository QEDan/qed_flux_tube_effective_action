import numpy as np
import torch
from typing import Union
from src.python.profiles import FieldProfile

class DeltaFunctionShellProfile(FieldProfile):
    def __init__(self, rho: Union[np.ndarray, torch.Tensor], R: float, F: float, e: float = 1.0) -> None:
        """
        Delta-function shell magnetic field profile:
        B(rho) = (F / (2*pi*R)) * delta(rho - R)
        This corresponds to a flux tube at radius R.
        """
        super().__init__(rho)
        self.R = R
        self.F = F
        self.e = e
        self.update()

    def update(self) -> None:
        # For a numerical solver, we represent the Delta function as a sharp peak 
        # (normalized to the flux F) on a fine grid.
        # However, it is mathematically more robust to define the A_phi directly:
        # A_phi(rho) = (F / 2*pi) * (1/rho) * theta(rho - R)
        
        pre = self.F / (2.0 * np.pi)
        
        # A_phi(rho) = (pre / rho) for rho >= R, else 0
        self.a_phi = torch.where(self.rho >= self.R, pre / self.rho, torch.zeros_like(self.rho))
        
        # B(rho) is zero everywhere except at R, where it is a delta.
        # da_phi = B - A_phi/rho. Since B=0 for rho != R, da_phi = -A_phi/rho
        r_safe = torch.where(self.rho == 0, torch.tensor(1e-15, device=self.rho.device), self.rho)
        self.da_phi = -self.a_phi / r_safe
        
        # Note: The solver must be configured to handle the boundary jump at R.
