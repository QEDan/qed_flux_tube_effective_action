import torch
import numpy as np

class PyTorchSolver:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
    def get_v_eff(self, r, params, a_phi, da_phi):
        """
        Computes the effective potential for a batch of parameters.
        params: dict of tensors (chi, ml, sigma3, m, e) each of shape (batch_size,)
        a_phi, da_phi: scalars or tensors of shape (batch_size,)
        """
        e = params['e']
        ml = params['ml'].to(torch.float64)
        s3 = params['sigma3'].to(torch.float64)
        chi = params['chi']
        m = params['m']
        
        # V_ml(rho) = e*sigma3 * (Aphi/rho + dAphi/drho) + (ml^2-1)/rho^2 + e^2*Aphi^2 - 2*e*ml*Aphi/rho
        v_ml = e * s3 * (a_phi / r + da_phi) + (ml**2 - 1.0) / (r**2) + (e * a_phi)**2 - 2.0 * e * ml * a_phi / r
        
        # ODE term: v_ml + 1/r^2 + chi^2 + m^2
        return v_ml + 1.0 / (r**2) + chi**2 + m**2

    def solve_batch(self, params_list, field_profile):
        """
        params_list: list of dicts (converted to batch tensors)
        field_profile: FieldProfile object
        """
        # Get tensors without detaching to preserve gradients
        rho, a_phi, da_phi = field_profile.get_arrays(as_numpy=False)
        rho = rho.to(self.device)
        a_phi = a_phi.to(self.device)
        da_phi = da_phi.to(self.device)
        
        n_batch = len(params_list)
        n_points = len(rho)
        
        # Batch parameters
        params = {
            'chi': torch.tensor([p['chi'] for p in params_list], device=self.device, dtype=torch.complex128),
            'ml': torch.tensor([p['ml'] for p in params_list], device=self.device, dtype=torch.int32),
            'sigma3': torch.tensor([p['sigma3'] for p in params_list], device=self.device, dtype=torch.int32),
            'm': torch.tensor([p['m'] for p in params_list], device=self.device, dtype=torch.float64),
            'e': torch.tensor([p['e'] for p in params_list], device=self.device, dtype=torch.float64),
        }
        
        # u0 integration (forward)
        u0 = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.complex128)
        
        # Initial conditions for u0
        rho_min = rho[0]
        abs_ml = torch.abs(params['ml']).to(torch.float64)
        
        u_init = torch.where(abs_ml == 0, 
                             torch.ones_like(abs_ml, dtype=torch.complex128), 
                             rho_min**abs_ml + 0j)
        du_init = torch.where(abs_ml == 0, 
                              torch.zeros_like(abs_ml, dtype=torch.complex128), 
                              abs_ml * rho_min**(abs_ml - 1) + 0j)
        
        curr_state = torch.stack([u_init, du_init], dim=1) # (batch, 2)
        u0[:, 0] = curr_state[:, 0]
        
        for i in range(n_points - 1):
            h = rho[i+1] - rho[i]
            curr_state = self.rk4_step(rho[i], h, curr_state, params, a_phi[i], da_phi[i])
            u0[:, i+1] = curr_state[:, 0]
            
        du0_last = curr_state[:, 1]
        
        # uinf integration (backward)
        uinf = torch.zeros((n_batch, n_points), device=self.device, dtype=torch.complex128)
        
        rho_max = rho[-1]
        k = torch.sqrt(params['chi']**2 + params['m']**2)
        # Ensure Re(k) > 0
        k = torch.where(k.real < 0, -k, k)
        
        u_inf_init = torch.exp(-k * rho_max) / torch.sqrt(rho_max)
        du_inf_init = (-k - 0.5/rho_max) * u_inf_init
        
        curr_state = torch.stack([u_inf_init, du_inf_init], dim=1)
        uinf[:, -1] = curr_state[:, 0]
        
        for i in range(n_points - 1, 0, -1):
            h = rho[i-1] - rho[i]
            # Use i-1 to be consistent with the forward pass for the same interval
            curr_state = self.rk4_step(rho[i], h, curr_state, params, a_phi[i-1], da_phi[i-1])
            uinf[:, i-1] = curr_state[:, 0]
            
        # Wronskian W0 = rho * (u0' * uinf - u0 * uinf') at rho_max
        W0 = rho_max * (du0_last * uinf[:, -1] - u0[:, -1] * du_inf_init)
        
        results = (u0 * uinf) / W0.unsqueeze(1)
        return results

    def f(self, r, state, params, a_phi, da_phi):
        u = state[:, 0]
        du = state[:, 1]
        v_eff = self.get_v_eff(r, params, a_phi, da_phi)
        
        d_u = du
        d_du = -1.0/r * du + v_eff * u
        return torch.stack([d_u, d_du], dim=1)

    def rk4_step(self, r, h, state, params, a_phi, da_phi):
        k1 = self.f(r, state, params, a_phi, da_phi)
        k2 = self.f(r + 0.5*h, state + 0.5*h*k1, params, a_phi, da_phi)
        k3 = self.f(r + 0.5*h, state + 0.5*h*k2, params, a_phi, da_phi)
        k4 = self.f(r + h, state + h*k3, params, a_phi, da_phi)
        return state + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
