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
        F = getattr(field_profile, 'F', None)

        params = {
            'chi': torch.tensor([p['chi'] for p in params_list], device=self.device, dtype=torch.complex128),
            'ml': torch.tensor([p['ml'] for p in params_list], device=self.device, dtype=torch.int32),
            'sigma3': torch.tensor([p['sigma3'] for p in params_list], device=self.device, dtype=torch.int32),
            'm': torch.tensor([p['m'] for p in params_list], device=self.device, dtype=torch.float64),
            'e': torch.tensor([p['e'] for p in params_list], device=self.device, dtype=torch.float64),
        }

        # Ensure vacuum integration BCs match interacting case
        # Field profile might be FieldProfile(vacuum), ensure F=0, lambd=0 passed or default
        F = getattr(field_profile, 'F', 0.0)
        
        # 1. Forward integration
        u0 = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.complex128)
        log_scale_u0 = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.complex128)
        
        abs_ml = torch.abs(params['ml']).to(torch.float64)
        v_eff_start = self.get_v_eff(rho[0], params, a_phi[0], da_phi[0])
        k_eff2 = -(v_eff_start - (abs_ml*abs_ml) / (rho[0]*rho[0]))
        
        u0_start_norm = 1.0 - k_eff2 * rho[0]*rho[0] / (4.0 * (abs_ml + 1.0))
        # Initialize derivative, ensuring stability for small rho
        rho_val = torch.max(rho[0], torch.tensor(1e-10, device=self.device))
        du0_start_norm = (abs_ml / rho_val) * u0_start_norm - (k_eff2 * rho_val / (2.0 * (abs_ml + 1.0)))
        
        # Ensure finite state
        state_u0 = torch.stack([u0_start_norm + 0j, du0_start_norm + 0j], dim=1)
        state_u0 = torch.where(torch.isfinite(state_u0), state_u0, torch.tensor([1.0+0j, 0.0+0j], device=self.device))
        
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
        # k2_ext = chi^2 - m^2. For Euclidean chi = iQ, k2_ext = -Q^2 - m^2 = -kappa^2.
        k2_ext = params['chi']*params['chi'] - params['m']*params['m']
        
        # Backward boundary condition: 
        F_val = F if F is not None else 0.0
        n_order = params['ml'].to(torch.float64) - (params['e'] * (F_val) / (constants.TWO_PI))

        u_inf_init = torch.zeros(n_batch, device=self.device, dtype=torch.complex128)
        du_inf_init = torch.zeros(n_batch, device=self.device, dtype=torch.complex128)
        log_acc_uinf_init = torch.zeros(n_batch, device=self.device, dtype=torch.float64)
        
        for b in range(n_batch):
            k2, ord_val = k2_ext[b].item(), n_order[b].item()
            # Handle both Minkowski (k2 > 0) and Euclidean (k2 < 0)
            if k2.real > 0:
                # Oscillatory (Minkowski)
                k_val = np.sqrt(k2)
                z = k_val * rho_max
                u_mp = mpmath.bessely(ord_val, z)
                du_mp = mpmath.diff(lambda x: mpmath.bessely(ord_val, x), z) * k_val
            else:
                # Decaying (Euclidean) - This is the preferred stable mode
                kappa = np.sqrt(-k2 + 1e-15 + 0j)
                z = kappa * rho_max
                u_mp = mpmath.besselk(ord_val, z)
                # Use a more stable derivative for K_n
                # K_n' = -1/2 (K_{n-1} + K_{n+1})
                du_mp = -0.5 * (mpmath.besselk(ord_val - 1, z) + mpmath.besselk(ord_val + 1, z)) * kappa
                
            # log(sqrt(|u|^2 + |du|^2)) = 0.5 * log(|u|^2 + |du|^2)
            # Use mpmath.log to avoid underflow
            mag_sq = mpmath.mpf(abs(u_mp))**2 + mpmath.mpf(abs(du_mp))**2
            if mag_sq > 0:
                log_mag = 0.5 * mpmath.log(mag_sq)
                # Normalize u_mp and du_mp to unit magnitude in mpmath before converting
                u_norm = u_mp / mpmath.sqrt(mag_sq)
                du_norm = du_mp / mpmath.sqrt(mag_sq)
                u_inf_init[b] = complex(u_norm)
                du_inf_init[b] = complex(du_norm)
                log_acc_uinf_init[b] = float(log_mag)
            else:
                # Pure zero fallback (should not happen with besselk unless z is huge)
                u_inf_init[b] = 1.0 + 0j
                du_inf_init[b] = 0.0 + 0j
                log_acc_uinf_init[b] = -1000.0 # Represent very small

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
        # Ensure W0 is stable, avoiding division by zero
        W0_stable = torch.where(torch.abs(W0) < 1e-12, torch.tensor(1e-12, device=self.device, dtype=W0.dtype), W0)
        log_diff = (log_scale_u0 + log_scale_uinf) - (log_acc_u0 + log_acc_uinf_init).unsqueeze(1)

        # Consistent with PyTorchSolver: G = - r * u0 * uinf / W0 satisfies the correct physical convention.
        res = - (rho.unsqueeze(0) * u0 * uinf) / (W0_stable.unsqueeze(1)) * torch.exp(log_diff)
        
        # Explicitly zero out the origin to avoid 0 * inf = NaN
        res[:, 0] = torch.where(rho[0] == 0, torch.tensor(0.0, device=self.device, dtype=res.dtype), res[:, 0])
        return res, W0
