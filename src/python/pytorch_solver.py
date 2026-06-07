from src.python import constants
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional

class PyTorchSolver:
    def __init__(self, device: Optional[str] = 'cpu') -> None:  # 'cpu' device because it benchmarked faster than gpu
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    def get_v_eff(self, r: torch.Tensor, params: Dict[str, torch.Tensor], a_phi: torch.Tensor, da_phi: torch.Tensor) -> torch.Tensor:
        """
        Computes the effective potential V_eff(r) for the radial Schrodinger-like equation.
        """
        r_safe = torch.max(r, torch.tensor(1e-15, device=self.device))
        b_field = a_phi/r_safe + da_phi
        # Interaction terms from -Pi^2 + e*sigma3*B
        v_ml = params['e']**2 * a_phi**2 + params['e']*params['sigma3']*b_field - 2*params['e']*params['ml']*a_phi/r_safe
        
        # Centrifugal term and spectral shift
        r2_eps = r_safe * r_safe
        res = v_ml + (params['ml'].to(torch.float64)**2) / r2_eps - (params['chi']**2 - params['m']**2)
        return res

    def rk4_step(self, r: torch.Tensor, h: torch.Tensor, state: torch.Tensor, params: Dict[str, torch.Tensor], 
                 a_p_start: torch.Tensor, da_p_start: torch.Tensor, 
                 a_p_mid: torch.Tensor, da_p_mid: torch.Tensor, 
                 a_p_end: torch.Tensor, da_p_end: torch.Tensor) -> torch.Tensor:
        def f(r_val, state_val, a_p, da_p):
            u, du = state_val[:, 0], state_val[:, 1]
            v_eff = self.get_v_eff(r_val, params, a_p, da_p)
            r_safe = torch.max(r_val, torch.tensor(1e-15, device=self.device))
            d2u = v_eff * u - (1.0/r_safe) * du
            return torch.stack([du, d2u], dim=1)

        k1 = f(r, state, a_p_start, da_p_start)
        k2 = f(r + 0.5*h, state + 0.5*h*k1, a_p_mid, da_p_mid)
        k3 = f(r + 0.5*h, state + 0.5*h*k2, a_p_mid, da_p_mid)
        k4 = f(r + h, state + h*k3, a_p_end, da_p_end)
        res = state + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        return res

    def solve_batch(self, params_list: List[Dict[str, Any]], field_profile: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solves the radial ODE for a batch of parameters.
        The solver assumes the potential is continuous unless the field_profile 
        returns a list of discontinuities via get_discontinuities().
        """
        rho, a_phi, da_phi = field_profile.get_arrays(as_numpy=False)
        rho = rho.to(self.device)
        a_phi = a_phi.to(self.device)
        da_phi = da_phi.to(self.device)
        n_batch = len(params_list)
        n_points = len(rho)
        
        # Get discontinuities from profile for jump handling
        discontinuities = field_profile.get_discontinuities()

        # Prepare parameters as tensors to maintain gradient flow
        params = {
            'chi': torch.stack([p['chi'] if torch.is_tensor(p['chi']) else torch.tensor(p['chi'], device=self.device, dtype=torch.complex128) for p in params_list]),
            'ml': torch.stack([p['ml'] if torch.is_tensor(p['ml']) else torch.tensor(p['ml'], device=self.device, dtype=torch.float64) for p in params_list]),
            'sigma3': torch.stack([p['sigma3'] if torch.is_tensor(p['sigma3']) else torch.tensor(p['sigma3'], device=self.device, dtype=torch.float64) for p in params_list]),
            'm': torch.tensor([p['m'] for p in params_list], device=self.device, dtype=torch.float64),
            'e': torch.tensor([p['e'] for p in params_list], device=self.device, dtype=torch.float64),
        }

        # 1. Forward integration (u0)
        u0 = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.complex128)
        log_scale_u0 = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.float64)
        
        abs_ml = torch.abs(params['ml'])
        v_eff_start = self.get_v_eff(rho[0], params, a_phi[0], da_phi[0])
        k_eff2 = -(v_eff_start - (abs_ml*abs_ml) / (rho[0]*rho[0] + 1e-30))
        
        u0_start_norm = 1.0 - k_eff2 * rho[0]*rho[0] / (4.0 * (abs_ml + 1.0))
        # Initialize derivative, ensuring stability for small rho
        rho_val = torch.max(rho[0], torch.tensor(1e-15, device=self.device))
        du0_start_norm = (abs_ml / rho_val) * u0_start_norm - (k_eff2 * rho_val / (2.0 * (abs_ml + 1.0)))
        
        state_u0 = torch.stack([u0_start_norm + 0j, du0_start_norm + 0j], dim=1)
        norm0 = torch.norm(state_u0, dim=1, keepdim=True) + 1e-100
        state_u0 = state_u0 / norm0
        log_acc_u0 = abs_ml * torch.log(rho_val) + torch.log(norm0.squeeze(1))
        
        u0[:, 0] = state_u0[:, 0]
        log_scale_u0[:, 0] = log_acc_u0

        for i in range(n_points - 1):
            h = rho[i+1] - rho[i]
            
            # Apply jump conditions if any registered discontinuity is crossed
            for disc in discontinuities:
                if rho[i] < disc.location <= rho[i+1]:
                    mag = torch.as_tensor(disc.magnitude, device=self.device, dtype=torch.complex128)
                    if disc.is_sigma3_dependent:
                        mag = mag * params['sigma3'].to(torch.complex128)
                    state_u0[:, 1] += mag * state_u0[:, 0]

            state_u0 = self.rk4_step(rho[i], h, state_u0, params, 
                                     a_phi[i], da_phi[i],
                                     0.5*(a_phi[i]+a_phi[i+1]), 0.5*(da_phi[i]+da_phi[i+1]),
                                     a_phi[i+1], da_phi[i+1])
            
            # Normalize at each step for maximum stability and consistent scaling
            norm = torch.norm(state_u0, dim=1, keepdim=True) + 1e-100
            state_u0 = state_u0 / norm
            log_acc_u0 = log_acc_u0 + torch.log(norm.squeeze(1))
            
            u0[:, i+1] = state_u0[:, 0]
            log_scale_u0[:, i+1] = log_acc_u0

        # 2. Backward integration (uinf)
        uinf = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.complex128)
        log_scale_uinf = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.float64)
        
        from src.python import torch_special
        k2_ext = params['chi']*params['chi'] - params['m']*params['m']
        F_val = field_profile.F if hasattr(field_profile, 'F') else 0.0
        n_order = params['ml'].to(torch.float64) - (params['e'] * (F_val) / (constants.TWO_PI))
        
        # Euclidean initialization (kappa^2 = -k2_ext)
        # Use full complex k2_ext
        is_eucl = (k2_ext.real <= 0)
        # We need a safe k2 for each branch to avoid NaN gradients from the un-masked branch
        # But we must use the complex value for the masked branch too, just a safe one.
        k2_eucl = torch.where(is_eucl, k2_ext, torch.tensor(-1.0 + 0j, dtype=k2_ext.dtype, device=k2_ext.device))
        kappa = torch.sqrt(-k2_eucl)
        zm = kappa * rho[-1]
        u_init_eucl = torch_special.bessel_kv(n_order, zm)
        du_init_eucl = -0.5 * (torch_special.bessel_kv(n_order - 1, zm) + torch_special.bessel_kv(n_order + 1, zm)) * kappa
        
        # Minkowski initialization (k^2 = k2_ext)
        k2_mink = torch.where(~is_eucl, k2_ext, torch.tensor(1.0 + 0j, dtype=k2_ext.dtype, device=k2_ext.device))
        k_min = torch.sqrt(k2_mink)
        zm_min = k_min * rho[-1]
        u_init_mink = torch_special.bessel_yv(n_order, zm_min)
        # dY/dz = 0.5*(Y_{nu-1} - Y_{nu+1})
        du_init_mink = 0.5 * (torch_special.bessel_yv(n_order - 1, zm_min) - torch_special.bessel_yv(n_order + 1, zm_min)) * k_min
        
        u_inf_init = torch.where(is_eucl, u_init_eucl, u_init_mink)
        du_inf_init = torch.where(is_eucl, du_init_eucl, du_init_mink)
        
        state_uinf = torch.stack([u_inf_init, du_inf_init], dim=1)
        norm_inf = torch.norm(state_uinf, dim=1, keepdim=True) + 1e-100
        state_uinf = state_uinf / norm_inf
        log_acc_uinf = torch.log(norm_inf.squeeze(1))
        
        # Capture the normalized state at rho[-1] for Wronskian calculation
        state_uinf_at_R = state_uinf.clone()
        log_acc_uinf_at_R = log_acc_uinf.clone()
        
        uinf[:, -1] = state_uinf[:, 0]
        log_scale_uinf[:, -1] = log_acc_uinf

        for i in range(n_points - 1, 0, -1):
            h = rho[i-1] - rho[i]

            for disc in discontinuities:
                if rho[i] > disc.location >= rho[i-1]:
                    mag = torch.as_tensor(disc.magnitude, device=self.device, dtype=torch.complex128)
                    if disc.is_sigma3_dependent:
                        mag = mag * params['sigma3'].to(torch.complex128)
                    state_uinf[:, 1] -= mag * state_uinf[:, 0]

            state_uinf = self.rk4_step(rho[i], h, state_uinf, params, 
                                       a_phi[i], da_phi[i],
                                       0.5*(a_phi[i]+a_phi[i-1]), 0.5*(da_phi[i]+da_phi[i-1]),
                                       a_phi[i-1], da_phi[i-1])
            
            # Normalize at each step for maximum stability and consistent scaling
            norm = torch.norm(state_uinf, dim=1, keepdim=True) + 1e-100
            state_uinf = state_uinf / norm
            log_acc_uinf = log_acc_uinf + torch.log(norm.squeeze(1))
                
            uinf[:, i-1] = state_uinf[:, 0]
            log_scale_uinf[:, i-1] = log_acc_uinf

        # W0 = rho * (u0' * uinf - u0 * uinf')
        # We evaluate at rho[-1] where state_u0 and state_uinf_at_R are both available and normalized.
        W0_norm = rho[-1] * (state_u0[:, 1] * state_uinf_at_R[:, 0] - state_u0[:, 0] * state_uinf_at_R[:, 1])
        W0_stable = torch.where(torch.abs(W0_norm) < 1e-12, torch.tensor(1e-12, device=self.device, dtype=W0_norm.dtype), W0_norm)
        
        # log_diff = (L0(r) + Linf(r)) - (L0(R) + Linf(R))
        log_W0_acc = log_scale_u0[:, -1] + log_acc_uinf_at_R
        log_diff = (log_scale_u0 + log_scale_uinf) - log_W0_acc.unsqueeze(1)
        
        # Green's function: G = - r * u0_norm * uinf_norm / W
        # W = r(u0 uinf' - u0' uinf)
        # W0_norm = r(u0' uinf - u0 uinf') = -W
        res = - (rho.unsqueeze(0) * u0 * uinf) / (W0_stable.unsqueeze(1)) * torch.exp(log_diff)
        
        if rho[0] == 0:
            res[:, 0] = 0.0
        return res, W0_stable
