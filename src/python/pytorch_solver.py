import torch
import numpy as np
from typing import Union, Dict, List, Any, Optional

class PyTorchSolver:
    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
    def get_v_eff(self, r: torch.Tensor, params: Dict[str, torch.Tensor], a_phi: torch.Tensor, da_phi: torch.Tensor) -> torch.Tensor:
        """
        Computes the effective potential for a batch of parameters.
        Matches Eq 2.50 in greensfunc.tex.
        """
        e = params['e']
        ml = params['ml'].to(torch.float64)
        s3 = params['sigma3'].to(torch.float64)
        chi = params['chi']
        m = params['m']
        
        # Add epsilon to r to avoid division by zero
        r_eps = r + 1e-15
        
        # V_ml(rho) = e*sigma3 * (Aphi/rho + dAphi/drho) + (ml^2-1)/rho^2 + e^2*Aphi^2 - 2*e*ml*Aphi/rho
        v_ml = e * s3 * (a_phi / r_eps + da_phi) + (ml*ml - 1.0) / (r_eps*r_eps) + (e * a_phi)*(e * a_phi) - 2.0 * e * ml * a_phi / r_eps
        
        # ODE: u'' + 1/rho * u' - [V_ml + 1/rho^2 - (chi^2 - m^2)] u = 0
        return v_ml + 1.0 / (r_eps*r_eps) - (chi*chi - m*m)


    def solve_batch(self, params_list: List[Dict[str, Any]], field_profile: Any) -> torch.Tensor:
        """
        Solves the ODE using full-domain integration for both regular solutions.
        Includes handling of delta-function jump conditions at profile boundaries.
        """
        # Get tensors
        rho, a_phi, da_phi = field_profile.get_arrays(as_numpy=False)
        rho = rho.to(self.device)
        a_phi = a_phi.to(self.device)
        da_phi = da_phi.to(self.device)
        n_batch = len(params_list)
        n_points = len(rho)

        # Check if profile has a jump (like StepFunctionProfile)
        lambd = getattr(field_profile, 'lambd', None)
        F = getattr(field_profile, 'F', None)

        # Batch parameters
        params = {
            'chi': torch.tensor([p['chi'] for p in params_list], device=self.device, dtype=torch.complex128),
            'ml': torch.tensor([p['ml'] for p in params_list], device=self.device, dtype=torch.int32),
            'sigma3': torch.tensor([p['sigma3'] for p in params_list], device=self.device, dtype=torch.int32),
            'm': torch.tensor([p['m'] for p in params_list], device=self.device, dtype=torch.float64),
            'e': torch.tensor([p['e'] for p in params_list], device=self.device, dtype=torch.float64),
        }

        # 1. Integrate forward from rho[0] (Regular at origin)
        u0 = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.complex128)
        abs_ml = torch.abs(params['ml']).to(torch.float64)
        u0[:, 0] = torch.where(abs_ml == 0, 1.0+0j, torch.pow(rho[0], abs_ml) + 0j)
        du0 = torch.where(abs_ml == 0, 0.0+0j, abs_ml * torch.pow(rho[0], abs_ml - 1) + 0j)
        state_u0 = torch.stack([u0[:, 0], du0], dim=1)

        for i in range(n_points - 1):
            h = rho[i+1] - rho[i]
            state_u0 = self.rk4_step(rho[i], h, state_u0, params, 
                                     a_phi[i], da_phi[i],
                                     0.5*(a_phi[i]+a_phi[i+1]), 0.5*(da_phi[i]+da_phi[i+1]),
                                     a_phi[i+1], da_phi[i+1])
            u0[:, i+1] = state_u0[:, 0]

            # Apply delta jump
            if lambd is not None and F is not None and rho[i] < lambd <= rho[i+1]:
                F_cal = params['e'] * F / (2.0 * np.pi)
                jump_coeff = -2.0 * F_cal / (lambd**2)
                state_u0[:, 1] += jump_coeff * state_u0[:, 0]

        # 2. Integrate backward from rho[-1] (Regular at infinity)
        uinf = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.complex128)
        k2_ext = params['chi']*params['chi'] - params['m']*params['m']
        k2_ext = torch.where(torch.abs(k2_ext) < 1e-12, torch.tensor(1e-12, dtype=torch.complex128), k2_ext)
        
        u_inf_init = torch.zeros(n_batch, device=self.device, dtype=torch.complex128)
        du_inf_init = torch.zeros(n_batch, device=self.device, dtype=torch.complex128)
        
        from scipy.special import yv, yvp, kv, kvp
        rho_max = rho[-1].item()
        F_ext = F if F is not None else 0.0
        F_cal_ext = params['e'] * F_ext / (2.0 * np.pi)
        n_order = params['ml'].to(torch.float64) - F_cal_ext

        for b in range(n_batch):
            k2 = k2_ext[b].item()
            ord_val = n_order[b].item()
            if k2.real > 0:
                k = np.sqrt(k2)
                u_inf_init[b] = yv(ord_val, k * rho_max)
                du_inf_init[b] = k * yvp(ord_val, k * rho_max)
            else:
                kappa = np.sqrt(-k2)
                u_inf_init[b] = kv(ord_val, kappa * rho_max)
                du_inf_init[b] = kappa * kvp(ord_val, kappa * rho_max)

        state_uinf = torch.stack([u_inf_init, du_inf_init], dim=1)
        uinf[:, -1] = state_uinf[:, 0]

        for i in range(n_points - 1, 0, -1):
            h = rho[i-1] - rho[i]
            state_uinf = self.rk4_step(rho[i], h, state_uinf, params, 
                                       a_phi[i], da_phi[i],
                                       0.5*(a_phi[i]+a_phi[i-1]), 0.5*(da_phi[i]+da_phi[i-1]),
                                       a_phi[i-1], da_phi[i-1])
            uinf[:, i-1] = state_uinf[:, 0]
            if lambd is not None and F is not None and rho[i] > lambd >= rho[i-1]:
                F_cal = params['e'] * F / (2.0 * np.pi)
                jump_coeff = 2.0 * F_cal / (lambd**2)
                state_uinf[:, 1] += jump_coeff * state_uinf[:, 0]

        # 3. Wronskian at rho_max
        # W0 = rho * (u0' * uinf - u0 * uinf')
        W0 = rho[-1] * (state_u0[:, 1] * u_inf_init - state_u0[:, 0] * du_inf_init)

        results = (rho.unsqueeze(0) * u0 * uinf) / W0.unsqueeze(1)
        self.last_u0 = u0
        self.last_uinf = uinf
        return results, W0


    def f(self, r: torch.Tensor, state: torch.Tensor, params: Dict[str, torch.Tensor], a_phi: torch.Tensor, da_phi: torch.Tensor) -> torch.Tensor:
        u = state[:, 0]
        du = state[:, 1]
        v_eff = self.get_v_eff(r, params, a_phi, da_phi)
        
        d_u = du
        d_du = -1.0/r * du + v_eff * u
        return torch.stack([d_u, d_du], dim=1)

    def rk4_step(self, r: torch.Tensor, h: torch.Tensor, state: torch.Tensor, params: Dict[str, torch.Tensor], 
                 a_p_start: torch.Tensor, da_p_start: torch.Tensor, 
                 a_p_mid: torch.Tensor, da_p_mid: torch.Tensor, 
                 a_p_end: torch.Tensor, da_p_end: torch.Tensor) -> torch.Tensor:
        k1 = self.f(r, state, params, a_p_start, da_p_start)
        k2 = self.f(r + 0.5*h, state + 0.5*h*k1, params, a_p_mid, da_p_mid)
        k3 = self.f(r + 0.5*h, state + 0.5*h*k2, params, a_p_mid, da_p_mid)
        k4 = self.f(r + h, state + h*k3, params, a_p_end, da_p_end)
        return state + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
