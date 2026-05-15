import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional

class PyTorchSolver:
    def __init__(self, device: Optional[str] = 'cpu') -> None:  # 'cpu' device because it benchmarked faster than gpu
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    def get_v_eff(self, r: torch.Tensor, params: Dict[str, torch.Tensor], a_phi: torch.Tensor, da_phi: torch.Tensor) -> torch.Tensor:
        # Standard Pauli equation potential in cylindrical coordinates
        # V = e^2 A^2 + e sigma3 B - 2 e ml A / r + ml^2 / r^2 - (chi^2 - m^2)
        # Note: We keep the ml^2/r^2 term here and the 1/r du term in the ODE.
        # This matches the standard Bessel equation: u'' + 1/r u' + (k^2 - ml^2/r^2) u = 0
        b_field = a_phi/r + da_phi
        v_ml = params['e']**2 * a_phi**2 + params['e']*params['sigma3']*b_field - 2*params['e']*params['ml']*a_phi/r
        r_eps = torch.where(torch.abs(r) < 1e-15, torch.tensor(1e-15, device=self.device), r)
        res = v_ml + (params['ml'].to(torch.float64)**2) / (r_eps*r_eps) - (params['chi']**2 - params['m']**2)
        return res

    def rk4_step(self, r: torch.Tensor, h: torch.Tensor, state: torch.Tensor, params: Dict[str, torch.Tensor], 
                 a_p_start: torch.Tensor, da_p_start: torch.Tensor, 
                 a_p_mid: torch.Tensor, da_p_mid: torch.Tensor, 
                 a_p_end: torch.Tensor, da_p_end: torch.Tensor) -> torch.Tensor:
        def f(r_val, state_val, a_p, da_p):
            u, du = state_val[:, 0], state_val[:, 1]
            v_eff = self.get_v_eff(r_val, params, a_p, da_p)
            d2u = v_eff * u - (1.0/r_val) * du
            return torch.stack([du, d2u], dim=1)

        k1 = f(r, state, a_p_start, da_p_start)
        k2 = f(r + 0.5*h, state + 0.5*h*k1, a_p_mid, da_p_mid)
        k3 = f(r + 0.5*h, state + 0.5*h*k2, a_p_mid, da_p_mid)
        k4 = f(r + h, state + h*k3, a_p_end, da_p_end)
        res = state + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        return res

    def rk4_step(self, r: torch.Tensor, h: torch.Tensor, state: torch.Tensor, params: Dict[str, torch.Tensor], 
                 a_p_start: torch.Tensor, da_p_start: torch.Tensor, 
                 a_p_mid: torch.Tensor, da_p_mid: torch.Tensor, 
                 a_p_end: torch.Tensor, da_p_end: torch.Tensor) -> torch.Tensor:
        def f(r_val, state_val, a_p, da_p):
            u, du = state_val[:, 0], state_val[:, 1]
            v_eff = self.get_v_eff(r_val, params, a_p, da_p)
            d2u = v_eff * u - (1.0/r_val) * du
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
        F = getattr(field_profile, 'F', None)

        params = {
            'chi': torch.tensor([p['chi'] for p in params_list], device=self.device, dtype=torch.complex128),
            'ml': torch.tensor([p['ml'] for p in params_list], device=self.device, dtype=torch.int32),
            'sigma3': torch.tensor([p['sigma3'] for p in params_list], device=self.device, dtype=torch.int32),
            'm': torch.tensor([p['m'] for p in params_list], device=self.device, dtype=torch.float64),
            'e': torch.tensor([p['e'] for p in params_list], device=self.device, dtype=torch.float64),
        }

        # 1. Forward integration
        u0 = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.complex128)
        log_scale_u0 = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.complex128)
        
        abs_ml = torch.abs(params['ml']).to(torch.float64)
        v_eff_start = self.get_v_eff(rho[0], params, a_phi[0], da_phi[0])
        k_eff2 = -(v_eff_start - (abs_ml*abs_ml) / (rho[0]*rho[0]))
        
        u0_start_norm = 1.0 - k_eff2 * rho[0]*rho[0] / (4.0 * (abs_ml + 1.0))
        du0_start_norm = (abs_ml / rho[0]) * u0_start_norm - (k_eff2 * rho[0] / (2.0 * (abs_ml + 1.0)))
        
        state_u0 = torch.stack([u0_start_norm + 0j, du0_start_norm + 0j], dim=1)
        norm0 = torch.norm(state_u0, dim=1, keepdim=True) + 1e-100
        state_u0 = state_u0 / norm0
        log_acc_u0 = abs_ml * torch.log(rho[0]) + torch.log(norm0.squeeze(1))
        
        u0[:, 0] = state_u0[:, 0]
        log_scale_u0[:, 0] = log_acc_u0

        for i in range(n_points - 1):
            h = rho[i+1] - rho[i]
            
            # Apply jump conditions if any registered discontinuity is crossed
            for disc in discontinuities:
                if rho[i] < disc.location <= rho[i+1]:
                    mag = disc.magnitude
                    if disc.is_sigma3_dependent:
                        mag = mag * params['sigma3'].to(torch.float64)
                    state_u0[:, 1] += mag * state_u0[:, 0]

            state_u0 = self.rk4_step(rho[i], h, state_u0, params, 
                                     a_phi[i], da_phi[i],
                                     0.5*(a_phi[i]+a_phi[i+1]), 0.5*(da_phi[i]+da_phi[i+1]),
                                     a_phi[i+1], da_phi[i+1])
            
            # Normalize whenever the state grows too large
            norm = torch.norm(state_u0, dim=1, keepdim=True)
            mask = norm.squeeze(1) > 1e10
            if torch.any(mask):
                scale = torch.where(mask, norm.squeeze(1), torch.ones_like(norm.squeeze(1)))
                state_u0 = state_u0 / scale.unsqueeze(1)
                log_acc_u0 += torch.log(scale)
            
            u0[:, i+1] = state_u0[:, 0]
            log_scale_u0[:, i+1] = log_acc_u0

        # 2. Backward integration
        uinf = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.complex128)
        log_scale_uinf = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.complex128)
        
        import mpmath
        rho_max = rho[-1].item()
        k2_ext = params['chi']*params['chi'] - params['m']*params['m']
        k2_ext = torch.where(torch.abs(k2_ext) < 1e-12, torch.tensor(1e-12 + 0j, dtype=torch.complex128), k2_ext)
        n_order = params['ml'].to(torch.float64) - (params['e'] * (F if F is not None else 0.0) / (2.0 * np.pi))

        u_inf_init = torch.zeros(n_batch, device=self.device, dtype=torch.complex128)
        du_inf_init = torch.zeros(n_batch, device=self.device, dtype=torch.complex128)
        log_acc_uinf_init = torch.zeros(n_batch, device=self.device, dtype=torch.float64)
        
        for b in range(n_batch):
            k2, ord_val = k2_ext[b].item(), n_order[b].item()
            if k2.real > 0:
                k_val = np.sqrt(k2)
                z = k_val * rho_max
                u_mp = mpmath.bessely(ord_val, z)
                du_mp = mpmath.diff(lambda x: mpmath.bessely(ord_val, x), z) * k_val
            else:
                kappa = np.sqrt(-k2)
                z = kappa * rho_max
                u_mp = mpmath.besselk(ord_val, z)
                du_mp = mpmath.diff(lambda x: mpmath.besselk(ord_val, x), z) * kappa
            mag_mp = mpmath.sqrt(abs(u_mp)**2 + abs(du_mp)**2)
            u_inf_init[b] = complex(u_mp / mag_mp)
            du_inf_init[b] = complex(du_mp / mag_mp)
            log_acc_uinf_init[b] = float(mpmath.log(mag_mp))

        state_uinf = torch.stack([u_inf_init, du_inf_init], dim=1)
        log_acc_uinf = log_acc_uinf_init.clone()
        uinf[:, -1] = state_uinf[:, 0]
        log_scale_uinf[:, -1] = log_acc_uinf

        for i in range(n_points - 1, 0, -1):
            h = rho[i-1] - rho[i]

            # Apply jump conditions in backward integration
            for disc in discontinuities:
                if rho[i] > disc.location >= rho[i-1]:
                    mag = disc.magnitude
                    if disc.is_sigma3_dependent:
                        mag = mag * params['sigma3'].to(torch.float64)
                    state_uinf[:, 1] += (-mag) * state_uinf[:, 0]

            state_uinf = self.rk4_step(rho[i], h, state_uinf, params, 
                                       a_phi[i], da_phi[i],
                                       0.5*(a_phi[i]+a_phi[i-1]), 0.5*(da_phi[i]+da_phi[i-1]),
                                       a_phi[i-1], da_phi[i-1])
            
            norm = torch.norm(state_uinf, dim=1, keepdim=True)
            mask = norm.squeeze(1) > 1e10
            if torch.any(mask):
                scale = torch.where(mask, norm.squeeze(1), torch.ones_like(norm.squeeze(1)))
                state_uinf = state_uinf / scale.unsqueeze(1)
                log_acc_uinf += torch.log(scale)
                
            uinf[:, i-1] = state_uinf[:, 0]
            log_scale_uinf[:, i-1] = log_acc_uinf

        W0 = rho[-1] * (state_u0[:, 1] * u_inf_init - state_u0[:, 0] * du_inf_init)
        
        # Ensure W0 is stable, avoiding division by zero or near-zero
        W0_stable = torch.where(torch.abs(W0) < 1e-12, torch.sgn(W0) * 1e-12, W0)
        log_diff = (log_scale_u0 + log_scale_uinf) - (log_acc_u0 + log_acc_uinf_init).unsqueeze(1)
        res = (rho.unsqueeze(0) * u0 * uinf) / (W0_stable.unsqueeze(1)) * torch.exp(log_diff)
        return res, W0
