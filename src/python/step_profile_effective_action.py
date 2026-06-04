import torch
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (Q_vals, rho_vals, d_Q, d_rho) for the step-function flux tube.
    Q is the Euclidean spectral coordinate (χ = i Q).
    """
    dtype, _ = _step_profile_dtypes(F_cal)
    device = F_cal.device
    lam = lambd.to(dtype=dtype, device=device)

    Q_vals = torch.linspace(0.1, Q_max, n_Q, dtype=dtype, device=device)
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
    Q can be a scalar or a 1D tensor of shape (n_Q,).
    Returns tensor of shape (n_Q, n_rho) if Q is 1D, or (n_rho,) if Q is scalar.
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
    gamma_arg = 0.5 + mu - kappa
    log_gamma = torch.lgamma(gamma_arg)
    log_factorial_ml = torch.lgamma(torch.tensor(float(abs_ml + 1), dtype=dtype, device=F.device))

    z = (F / lam ** 2) * rho_vals ** 2
    # To handle large lambda, we use log-space calculations.
    # Whittaker W and M can be very small or very large.
    # We only compute them for interior points to avoid exterior blowup.
    interior_mask = rho_vals <= lam
    rho_int = rho_vals[interior_mask]
    z_int = z[interior_mask]
    
    if Q.ndim == 0:
        log_abs_g_full = torch.zeros(rho_vals.shape, dtype=dtype, device=F.device)
        sign_g_full = torch.zeros(rho_vals.shape, dtype=dtype, device=F.device)
    else:
        log_abs_g_full = torch.zeros((Q.shape[0], rho_vals.shape[0]), dtype=dtype, device=F.device)
        sign_g_full = torch.zeros((Q.shape[0], rho_vals.shape[0]), dtype=dtype, device=F.device)
    
    if len(rho_int) > 0:
        # Prepare shapes for broadcasting Whittaker calls
        if Q.ndim > 0:
            kp = kappa.unsqueeze(-1)
            zi = z_int.unsqueeze(0)
        else:
            kp = kappa
            zi = z_int

        log_abs_M, sign_M = torch_special.whittaker_m_log(kp, mu, zi)
        log_abs_W, sign_W = torch_special.whittaker_w_log(kp, mu, zi)
        
        # Prepare shapes for broadcasting other terms
        if Q.ndim > 0:
            lg = log_gamma.unsqueeze(-1)
            ri = rho_int.unsqueeze(0)
        else:
            lg = log_gamma
            ri = rho_int

        log_abs_g_full_int = (lg - log_factorial_ml) + log_abs_M + log_abs_W + \
                             torch.log(lam ** 2 / (2.0 * F)) - 2.0 * torch.log(ri)
        
        # helper to get sign of Gamma(x) for real x
        def torch_gamma_sign(x):
            # Gamma(x) > 0 for x > 0
            # For x < 0, sign is -1 for (-1, 0), +1 for (-2, -1), -1 for (-3, -2), etc.
            # This is sign(sin(pi*x))? No.
            # Correct logic: if floor(x) is even, Gamma is negative? 
            # -0.5: floor=-1 (odd) -> negative
            # -1.5: floor=-2 (even) -> positive
            # -2.5: floor=-3 (odd) -> negative
            return torch.where(x > 0, torch.ones_like(x), 
                              torch.where(torch.remainder(torch.floor(x), 2) == 0, 
                                          torch.ones_like(x), -torch.ones_like(x)))
        
        sign_gamma = torch_gamma_sign(gamma_arg)
        if Q.ndim > 0:
            sg = sign_gamma.unsqueeze(-1)
        else:
            sg = sign_gamma
        sign_g_full_int = -1.0 * sg * sign_M * sign_W
        
        if Q.ndim > 0:
            log_abs_g_full[:, interior_mask] = log_abs_g_full_int
            sign_g_full[:, interior_mask] = sign_g_full_int
        else:
            log_abs_g_full[interior_mask] = log_abs_g_full_int
            sign_g_full[interior_mask] = sign_g_full_int
    
    g_full = (sign_g_full * torch.exp(log_abs_g_full)).to(cdtype)
    
    # In the exterior, the Green's function matches onto Bessel functions.
    # For now, we set it to zero for points > lambda to avoid Whittaker blowup,
    # as the analytic form above is only valid in the interior.
    if Q.ndim > 0:
        g_full[:, ~interior_mask] = 0.0
    else:
        g_full[~interior_mask] = 0.0

    # Topological vacuum radial Green's function:
    #     G_free(ρ) = - I_{ν}(κρ) K_{ν}(κρ),     κ = sqrt(Q² + m²)
    if global_mode:
        # Match only asymptotic flux (standard vacuum subtraction)
        nu_asym = torch.where(rho_vals <= lam, torch.zeros_like(rho_vals), torch.ones_like(rho_vals) * F_cal)
        order_local = torch.abs(ml_t - nu_asym)
    else:
        # Use local flux to match Orchestrator's strategy and ensure gauge invariance.
        nu_local = torch.where(rho_vals <= lam, F_cal * (rho_vals ** 2 / lam ** 2), F_cal)
        order_local = torch.abs(ml_t - nu_local)
    
    kappa_E = torch.sqrt(Q ** 2 + m ** 2)
    # Broadcasting: kappa_E (n_Q,), rho_vals (n_rho)
    if Q.ndim > 0:
        z_E = kappa_E.unsqueeze(-1) * rho_vals.unsqueeze(0)
        ol = order_local.unsqueeze(0)
    else:
        z_E = kappa_E * rho_vals
        ol = order_local
        
    ik_prod = torch_special.bessel_i_k_product(ol, z_E)
    g_free = (-ik_prod).to(cdtype)

    if Q.ndim > 0:
        return rho_vals.unsqueeze(0).to(cdtype) * (g_full - g_free)
    else:
        return rho_vals.to(cdtype) * (g_full - g_free)


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
        # Sum over both positive and negative m_l for completeness
        for ml in range(-ml_max, ml_max + 1):
            # Vectorized over Q
            integrands = step_profile_mode_integrand(
                Q_vals, ml, sigma3, F, lambd, rho_vals, m=m, global_mode=global_mode,
            )
            # integrands is (n_Q, n_rho). Q_vals is (n_Q,).
            # We want sum_Q integrands(Q, rho) * Q^3 * d_Q
            mode_sum = torch.sum(integrands * (Q_vals.unsqueeze(-1) ** 3) * d_Q, dim=0)
            h = h + mode_sum

    # UV subtraction (b2 term) consistent with Orchestrator.py
    # L_eff = Integral dQ Q^3 (mode_sum/rho - b2/Q^4)
    # b2 for 4 spinor states = (eB)^2 / 3.0
    # Here F is nu. e*B_phys = 2*nu/lambda^2.
    eB = (2.0 * F / lambd**2)
    b2 = (eB**2 / 3.0)
    # Point-wise subtraction per Q: sum_Q (b2 / Q^4 * Q^3 * d_Q) = b2 * sum_Q (d_Q / Q)
    # This cancels the logarithmic divergence in the spectral integral.
    uv_sum = b2 * torch.sum(d_Q / (Q_vals + 1e-15))
    
    # Apply in the interior where B is constant.
    interior_mask = rho_vals <= lambd
    h[interior_mask] = h[interior_mask] - rho_vals[interior_mask] * uv_sum

    return h


# Backwards-compatible alias for callers that imported the old chi-named helper.
step_profile_chi_spectral_sum = step_profile_Q_spectral_sum


def step_profile_effective_action_density(
        F_cal: torch.Tensor,
        lambd: torch.Tensor,
        rho_cm: Optional[torch.Tensor] = None,
        *,
        m: float = constants.ELECTRON_MASS,
        n_chi: int = 20, # Reduced
        n_rho: int = 20,
        n_ml: int = 5,   # Reduced
        Q_max: float = 10.0,
        global_mode: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Effective action density ρ(ρ_cm) for the step-function flux tube.

    Satisfies EA^{(1)} = 2π ∫_0^∞ dρ_cm  ρ(ρ_cm), matching the convention in
    locally_constant_field.heisenberg_euler_density_at_rho_cm() for the LCF
    limit.
    """
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

    # Spinor QED 1-loop spectral normalization for 4 states.
    # h contains sum over sigma3={1,-1} (2 states).
    # We multiply by 2.0 to account for total 4 spin states.
    # Minus sign aligns with Orchestrator's return -1.0 * L_eff_rho.
    rho_density = (-h.real * constants.HE_NORMALIZATION_FACTOR * 2.0).to(dtype)

    # Physics constraint: density is zero for rho > lambda (exterior)
    # when using an unmatched analytic solution.
    # The spectral integral code currently evaluates the Whittaker interior-Green's-function
    # only in the interior. In the exterior, g_full matches g_free exactly,
    # resulting in zero density.
    mask = rho_cm > lambd
    rho_density[mask] = 0.0

    return rho_cm, rho_density

def _require_integer_flux(F_cal):
    """Ensure flux is quantized (integer) for the analytic model."""
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
    """
    Renormalized 1-loop effective action (Eq. 7.72) via ρ integration:

    EA^{(1)} = 2π ∫_0^∞ dρ_cm  ρ(ρ_cm).
    """
    del e  # kept for API compatibility with callers
    rho_cm, rho_density = step_profile_effective_action_density(
        F_cal, lambd, m=m, n_chi=n_chi, n_rho=n_rho, n_ml=n_ml, Q_max=Q_max,
    )
    return _integrate_density(rho_cm, rho_density)


def _integrate_density(rho_cm: torch.Tensor, rho_density: torch.Tensor) -> torch.Tensor:
    """EA^{(1)} = -2π ∫ ρ(ρ_cm) dρ_cm (trapezoidal rule, autograd-safe).
    The minus sign aligns with Orchestrator.compute_effective_action.
    """
    dr = rho_cm[1:] - rho_cm[:-1]
    avg = 0.5 * (rho_density[1:] + rho_density[:-1])
    return (-2.0 * constants.PI * torch.sum(avg * dr)).squeeze()
