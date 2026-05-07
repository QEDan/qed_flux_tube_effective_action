import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List

class FluxConservingMLP(nn.Module):
    """
    An MLP architecture that enforces total flux conservation identically
    via the asymptotic boundary conditions of the gauge field A_phi.
    
    u(rho) = rho * A_phi(rho)
    B(rho) = (1/rho) * du/drho
    
    We map rho to x in [0, 1] via x = rho / (rho + L).
    We define u(rho) = (Phi/2pi) * g(x) where g(x) = x^2 * [1 + (1-x) * MLP(x)].
    This ensures u(0) = 0, u(inf) = Phi/2pi.
    """
    def __init__(self, hidden_dim: int = 32, num_layers: int = 3, total_flux: float = 2.0*np.pi*0.4, L: float = 1.0) -> None:
        super().__init__()
        self.Phi_over_2pi = total_flux / (2.0 * np.pi)
        self.L = L
        
        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes B(rho) and A_phi(rho) such that flux is conserved.
        """
        rho.requires_grad_(True)
        
        # 1. Coordinate mapping: rho -> x in [0, 1]
        x = rho / (rho + self.L)
        
        # 2. Enclosed flux shape function g(x)
        # We need g(0)=0, g(1)=1, g'(0)=0.
        # Let's use g(x) = x^2 * (3 - 2x) + x^2 * (1 - x)^2 * MLP(x)
        # The first part is the standard cubic Hermite that goes smoothly from 0 to 1.
        # The second part is a flexible deviation that vanishes at both 0 and 1.
        
        mlp_out = self.net(x)
        # Smoothed cubic base
        g_base = (x**2) * (3.0 - 2.0 * x)
        # Flexible part
        g_flex = (x**2) * ((1.0 - x)**2) * mlp_out
        
        g_x = g_base + g_flex
        u_rho = self.Phi_over_2pi * g_x
        
        # 3. Derive B(rho) = (1/rho) * du/drho
        du_drho = torch.autograd.grad(u_rho, rho, 
                                      grad_outputs=torch.ones_like(u_rho),
                                      create_graph=True, retain_graph=True)[0]
        
        r_safe = torch.where(rho == 0, torch.tensor(1e-15, device=rho.device), rho)
        B_vals = du_drho / r_safe
        
        # 4. Derive A_phi = u / rho
        a_phi = u_rho / r_safe
        
        return B_vals, a_phi
