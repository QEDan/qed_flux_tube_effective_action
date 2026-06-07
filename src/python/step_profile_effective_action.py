import torch
import numpy as np
from typing import Optional, Tuple
from src.python import constants
from src.python import torch_special

def _step_profile_dtypes(F: torch.Tensor) -> Tuple[torch.dtype, torch.dtype]:
    dtype = F.dtype
    if dtype == torch.float32:
        cdtype = torch.complex64
    else:
        cdtype = torch.complex128
    return dtype, cdtype

def step_profile_integration_grids(
        lambd: torch.Tensor,
        F_cal: torch.Tensor,
        *,
        n_Q: int = 50,
        n_rho: int = 200,
        Q_max: float = 10.0,
        Q_min: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (Q_vals, rho_vals, d_Q, d_rho) for the step-function flux tube.
    Q is the Euclidean spectral coordinate (χ = i Q).
    """
    dtype, _ = _step_profile_dtypes(F_cal)
    device = F_cal.device
    lam = lambd.to(dtype=dtype, device=device)

    Q_vals = torch.linspace(Q_min, Q_max, n_Q, dtype=dtype, device=device)
    rho_unit = torch.linspace(0.001, 2.0, n_rho, dtype=dtype, device=device)
    rho_vals = rho_unit * (lam / 1.0) # Scale such that rho=1 is at lam
    d_Q = Q_vals[1] - Q_vals[0]
    d_rho = rho_vals[1] - rho_vals[0]
    return Q_vals, rho_vals, d_Q, d_rho


def step_profile_mode_integrand(
        Q: torch.Tensor,
        ml: int,
        sigma3: float,
        F: torch.Tensor,
        lambd: torch.Tensor,
        rho_vals: torch.Tensor,
        *,
        m: float = constants.ELECTRON_MASS,
        global_mode: bool = False,
) -> torch.Tensor:
    """
    Vacuum-subtracted Euclidean radial spectral integrand for one or more (Q, m_l, σ₃),
    returned at every ρ ∈ ``rho_vals``.
    """
    dtype, cdtype = _step_profile_dtypes(F)
    F_cal = F
    abs_ml = abs(int(ml))
    mu = abs_ml / 2.0
    ml_t = torch.tensor(float(ml), dtype=dtype, device=F.device)
    sigma3_t = torch.tensor(sigma3, dtype=dtype, device=F.device)
    lam = lambd.to(dtype=dtype, device=F.device)

    # Euclidean rotation: chi^2 -> -Q^2
    k2 = -Q ** 2 - m ** 2 - (2.0 * F / lam ** 2) * (sigma3_t - ml_t)
    kappa = (lam ** 2 * k2) / (4.0 * F)
    
    k2_bg = -Q ** 2 - m ** 2 - (2.0 * F / lam ** 2) * (0.0 - ml_t)
    kappa_bg = (lam ** 2 * k2_bg) / (4.0 * F)

    gamma_arg = 0.5 + mu - kappa
    gamma_arg_bg = 0.5 + mu - kappa_bg
    
    log_factorial_ml = torch.lgamma(torch.tensor(float(abs_ml + 1), dtype=dtype, device=F.device))

    z = (F / lam ** 2) * rho_vals ** 2
    interior_mask = rho_vals <= lam
    rho_int = rho_vals[interior_mask]
    z_int = z[interior_mask]
    
    if Q.ndim == 0:
        log_abs_g_full = torch.zeros(rho_vals.shape, dtype=dtype, device=F.device)
        sign_g_full = torch.zeros(rho_vals.shape, dtype=cdtype, device=F.device)
        log_abs_g_base = torch.zeros(rho_vals.shape, dtype=dtype, device=F.device)
        sign_g_base = torch.zeros(rho_vals.shape, dtype=cdtype, device=F.device)
    else:
        log_abs_g_full = torch.zeros((Q.shape[0], rho_vals.shape[0]), dtype=dtype, device=F.device)
        sign_g_full = torch.zeros((Q.shape[0], rho_vals.shape[0]), dtype=cdtype, device=F.device)
        log_abs_g_base = torch.zeros((Q.shape[0], rho_vals.shape[0]), dtype=dtype, device=F.device)
        sign_g_base = torch.zeros((Q.shape[0], rho_vals.shape[0]), dtype=cdtype, device=F.device)
    
    if len(rho_int) > 0:
        if Q.ndim > 0:
            kp, kp_bg, zi = kappa.unsqueeze(-1), kappa_bg.unsqueeze(-1), z_int.unsqueeze(0)
        else:
            kp, kp_bg, zi = kappa, kappa_bg, z_int

        log_abs_M, sign_M = torch_special.whittaker_m_log(kp, mu, zi)
        log_abs_W, sign_W = torch_special.whittaker_w_log(kp, mu, zi)
        log_abs_M_bg, sign_M_bg = torch_special.whittaker_m_log(kp_bg, mu, zi)
        log_abs_W_bg, sign_W_bg = torch_special.whittaker_w_log(kp_bg, mu, zi)

        def torch_gamma_sign(x):
            return torch.where(x > 0, torch.ones_like(x), 
                              torch.where(torch.remainder(torch.floor(x), 2) == 0, 
                                          torch.ones_like(x), -torch.ones_like(x)))
        
        sg, sg_bg = torch_gamma_sign(gamma_arg), torch_gamma_sign(gamma_arg_bg)
        
        if Q.ndim > 0:
            lg, lg_bg = torch.lgamma(gamma_arg).unsqueeze(-1), torch.lgamma(gamma_arg_bg).unsqueeze(-1)
            ri, sg, sg_bg = rho_int.unsqueeze(0), sg.unsqueeze(-1), sg_bg.unsqueeze(-1)
        else:
            lg, lg_bg = torch.lgamma(gamma_arg), torch.lgamma(gamma_arg_bg)
            ri = rho_int

        # G = - (Gamma/ml!) * (lam^2 / 2F rho^2) * M * W
        log_pre = torch.log(lam ** 2 / (2.0 * F)) - 2.0 * torch.log(ri)
        
        log_abs_g_full_int = (lg - log_factorial_ml) + log_abs_M + log_abs_W + log_pre
        sign_g_full_int = -1.0 * sg * sign_M * sign_W

        log_abs_g_base_int = (lg_bg - log_factorial_ml) + log_abs_M_bg + log_abs_W_bg + log_pre
        sign_g_base_int = -1.0 * sg_bg * sign_M_bg * sign_W_bg
        
        if Q.ndim > 0:
            log_abs_g_full[:, interior_mask] = log_abs_g_full_int
            sign_g_full[:, interior_mask] = sign_g_full_int.to(cdtype)
            log_abs_g_base[:, interior_mask] = log_abs_g_base_int
            sign_g_base[:, interior_mask] = sign_g_base_int.to(cdtype)
        else:
            log_abs_g_full[interior_mask] = log_abs_g_full_int
            sign_g_full[interior_mask] = sign_g_full_int.to(cdtype)
            log_abs_g_base[interior_mask] = log_abs_g_base_int
            sign_g_base[interior_mask] = sign_g_base_int.to(cdtype)
    
    g_full = (sign_g_full * torch.exp(log_abs_g_full)).to(cdtype)
    g_base = (sign_g_base * torch.exp(log_abs_g_base)).to(cdtype)
    
    # Exterior
    kappa_E = torch.sqrt(Q ** 2 + m ** 2)
    nu_ext = torch.abs(ml_t - F_cal)
    if Q.ndim > 0:
        z_E, oa = kappa_E.unsqueeze(-1) * rho_vals.unsqueeze(0), nu_ext.unsqueeze(0)
    else:
        z_E, oa = kappa_E * rho_vals, nu_ext
        
    ik_prod_asym = torch_special.bessel_i_k_product(oa, z_E)
    g_free_asym = (-ik_prod_asym).to(cdtype)

    if global_mode:
        g_ref = g_free_asym
    else:
        g_ref = torch.where(interior_mask, g_base, g_free_asym)

    if Q.ndim > 0:
        return rho_vals.unsqueeze(0).to(cdtype) * (g_full - g_ref)
    else:
        return rho_vals.to(cdtype) * (g_full - g_ref)


def step_profile_Q_spectral_sum(
        F: torch.Tensor,
        lambd: torch.Tensor,
        rho_vals: torch.Tensor,
        Q_vals: torch.Tensor,
        d_Q: torch.Tensor,
        *,
        m: float = constants.ELECTRON_MASS,
        ml_max: int = 10,
        global_mode: bool = False,
) -> torch.Tensor:
    """
    h(ρ) = Σ_{σ₃, m_l} ∫ dQ  Q³  integrand(Q, ρ)
    """
    _, cdtype = _step_profile_dtypes(F)
    h = torch.zeros(rho_vals.shape, dtype=cdtype, device=F.device)

    for sigma3 in (1.0, -1.0):
        print(f"  Summing modes for sigma3={sigma3}...")
        for ml in range(-ml_max, ml_max + 1):
            if ml % 20 == 0:
                print(f"    ml = {ml}/{ml_max}")
            integrands = step_profile_mode_integrand(
                Q_vals, ml, sigma3, F, lambd, rho_vals, m=m, global_mode=global_mode,
            )
            mode_sum = torch.sum(integrands * (Q_vals.unsqueeze(-1) ** 3) * d_Q, dim=0)
            h = h + mode_sum

    # UV subtraction
    eB = (2.0 * F / lambd**2)
    b2 = (eB**2 / 6.0)
    uv_sum = b2 * torch.sum(d_Q * (Q_vals ** 3) / (Q_vals ** 2 + m ** 2) ** 2)
    
    interior_mask = rho_vals <= lambd
    h[interior_mask] = h[interior_mask] - rho_vals[interior_mask] * uv_sum

    return h


step_profile_chi_spectral_sum = step_profile_Q_spectral_sum


def step_profile_effective_action_density(
        F_cal: torch.Tensor,
        lambd: torch.Tensor,
        rho_cm: Optional[torch.Tensor] = None,
        *,
        m: float = constants.ELECTRON_MASS,
        n_chi: int = 20,
        n_rho: int = 20,
        n_ml: int = 100,
        Q_max: float = 10.0,
        global_mode: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _require_integer_flux(F_cal)
    F = F_cal
    dtype, _ = _step_profile_dtypes(F_cal)

    if rho_cm is None:
        _, rho_cm, _, _ = step_profile_integration_grids(
            lambd, F_cal, n_Q=n_chi, n_rho=n_rho, Q_max=Q_max,
        )
    else:
        rho_cm = rho_cm.to(dtype=dtype, device=F_cal.device)

    Q_vals, _, d_Q, _ = step_profile_integration_grids(
        lambd, F_cal, n_Q=n_chi, n_rho=n_rho, Q_max=Q_max,
    )
    h = step_profile_Q_spectral_sum(
        F, lambd, rho_cm, Q_vals, d_Q, m=m, ml_max=n_ml, global_mode=global_mode,
    )

    rho_density = (-h.real * constants.HE_NORMALIZATION_FACTOR * 2.0).to(dtype)
    mask = rho_cm > lambd
    rho_density[mask] = 0.0

    return rho_cm, rho_density

def _require_integer_flux(F_cal):
    if not torch.isclose(F_cal, torch.round(F_cal), atol=1e-5):
        raise NotImplementedError("Analytic effective action for non-integer flux is not supported.")

def step_profile_analytic_ea(
        F_cal: torch.Tensor,
        lambd: torch.Tensor,
        m: float = constants.ELECTRON_MASS,
        e: float = constants.ELECTRON_CHARGE,
        *,
        n_chi: int = 50,
        n_rho: int = 20,
        n_ml: int = 10,
        Q_max: float = 10.0,
) -> torch.Tensor:
    del e
    rho_cm, rho_density = step_profile_effective_action_density(
        F_cal, lambd, m=m, n_chi=n_chi, n_rho=n_rho, n_ml=n_ml, Q_max=Q_max,
    )
    return _integrate_density(rho_cm, rho_density)


def _integrate_density(rho_cm: torch.Tensor, rho_density: torch.Tensor) -> torch.Tensor:
    dr = rho_cm[1:] - rho_cm[:-1]
    avg = 0.5 * (rho_density[1:] + rho_density[:-1])
    return (-2.0 * constants.PI * torch.sum(avg * dr)).squeeze()
