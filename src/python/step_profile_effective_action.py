"""
Analytic 1-loop effective action and density for a step-function flux tube
(dissertation Eq. 7.72).

EA^{(1)} = 2π ∫_0^∞ dρ_cm  ρ(ρ_cm)

The spectral integral is taken on the Euclidean axis (χ = i Q). For the radial
Green's function this gives k² = −Q² − m² − (2F/λ²)(σ₃ − m_l), real and negative
for moderate Q so Whittaker M/W are evaluated far from the integer-b
logarithmic-case shell that breaks scipy.special.hyperu. The integrand is then
vacuum-subtracted mode-by-mode against the free massive Euclidean radial
Green's function (−I_{|m_l|} · K_{|m_l|}), matching the convention used by the
Orchestrator (src.python.orchestrator.Orchestrator.compute_effective_action).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from src.python import constants
from src.python import torch_special


def _require_integer_flux(F_cal: torch.Tensor) -> None:
    if not torch.allclose(F_cal.detach(), torch.round(F_cal.detach()), atol=1e-5):
        raise NotImplementedError("F_cal must be an integer.")


def _step_profile_dtypes(F_cal: torch.Tensor) -> Tuple[torch.dtype, torch.dtype]:
    dtype = torch.float64 if F_cal.dtype == torch.float64 else torch.float32
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    return dtype, cdtype


def step_profile_integration_grids(
        lambd: torch.Tensor,
        F_cal: torch.Tensor,
        *,
        n_Q: int = 50,
        n_rho: int = 20,
        Q_max: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (Q_vals, rho_vals, d_Q, d_rho) on the F_cal device.

    Q is the Euclidean spectral coordinate (χ = i Q).
    """
    dtype, _ = _step_profile_dtypes(F_cal)
    device = F_cal.device
    lam = lambd.to(dtype=dtype, device=device)

    Q_vals = torch.linspace(0.1, Q_max, n_Q, dtype=dtype, device=device)
    rho_unit = torch.linspace(0.001, 1.0, n_rho, dtype=dtype, device=device)
    rho_vals = rho_unit * (lam / rho_unit[-1])
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
) -> torch.Tensor:
    """
    Vacuum-subtracted Euclidean radial spectral integrand for one (Q, m_l, σ₃),
    returned at every ρ ∈ ``rho_vals``.

    Returns ρ × [G_full(ρ; iQ, m_l, σ₃) − G_free(ρ; iQ, |m_l|)] where G_full
    is the constant-B interior coincident-point radial Green's function
    (Whittaker M·W with the dissertation Eq. 2.76 normalization) and G_free is
    the massive Euclidean free radial Green's function (−I·K).
    """
    dtype, cdtype = _step_profile_dtypes(F)
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
    w0_inv = (lam ** 2 / (2.0 * F)) * torch.exp(log_gamma - log_factorial_ml)

    z = (F / lam ** 2) * rho_vals ** 2
    M_val = torch_special.whittaker_m(kappa, mu, z)
    W_val = torch_special.whittaker_w(kappa, mu, z)
    # Interior coincident-point radial Green's function (Eq. 2.76):
    #     G_full(ρ) = - w0_inv * M(z) * W(z)
    g_full = -(w0_inv * M_val * W_val).to(cdtype)

    # Free massive Euclidean radial Green's function:
    #     G_free(ρ) = - I_{|m_l|}(κρ) K_{|m_l|}(κρ),     κ = sqrt(Q² + m²)
    kappa_E = torch.sqrt(Q ** 2 + m ** 2)
    z_E = kappa_E * rho_vals
    i_val = torch_special.bessel_iv(abs_ml, z_E)
    k_val = torch_special.bessel_kv(abs_ml, z_E)
    g_free = (-i_val * k_val).to(cdtype)

    # Return ρ × ΔG so the spectral measure Q³ dQ aligns with the orchestrator
    # (which returns rho × radial Green's function from the solver and from
    # AnalyticBackgroundStrategy.compute_g0).
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
) -> torch.Tensor:
    """
    h(ρ) = Σ_{σ₃, m_l} ∫ dQ  Q³  integrand(Q, ρ)

    The m_l sum runs over m_l ∈ {0, 1, …, ml_max−1}, matching the Orchestrator's
    default `ml_values = range(10)`.
    """
    _, cdtype = _step_profile_dtypes(F)
    h = torch.zeros(rho_vals.shape, dtype=cdtype, device=F.device)

    for sigma3 in (1.0, -1.0):
        for ml in range(ml_max):
            for Q in Q_vals:
                integrand = step_profile_mode_integrand(
                    Q, ml, sigma3, F, lambd, rho_vals, m=m,
                )
                h = h + integrand * (Q ** 3) * d_Q

    return h


# Backwards-compatible alias for callers that imported the old chi-named helper.
step_profile_chi_spectral_sum = step_profile_Q_spectral_sum


def step_profile_effective_action_density(
        F_cal: torch.Tensor,
        lambd: torch.Tensor,
        rho_cm: Optional[torch.Tensor] = None,
        *,
        m: float = constants.ELECTRON_MASS,
        n_chi: int = 50,
        n_rho: int = 20,
        n_ml: int = 10,
        Q_max: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Effective action density ρ(ρ_cm) for the step-function flux tube.

    Satisfies EA^{(1)} = 2π ∫_0^∞ dρ_cm  ρ(ρ_cm), matching the convention in
    locally_constant_field.heisenberg_euler_density_at_rho_cm() for the LCF
    limit.

    Parameters
    ----------
    F_cal, lambd
        Flux-tube parameters (tensors support autograd). F_cal = e F / (2π).
    rho_cm
        Radial grid; default spans (0, λ] with ``n_rho`` points.
    n_chi
        Number of Euclidean-Q quadrature points (legacy name kept for backwards
        compatibility with callers passing ``n_chi``).
    n_ml
        Number of m_l values to sum (m_l ∈ {0, …, n_ml−1}).
    Q_max
        Upper limit of the Euclidean spectral integration.
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
        F, lambd, rho_cm, Q_vals, d_Q, m=m, ml_max=n_ml,
    )
    # Spinor QED 1-loop spectral normalization. Factor of 2 multiplies the
    # mode_sum to recover all 4 spin states from the σ₃ = ±1 partial sums, as
    # in Orchestrator.compute_effective_action (mode_sums * 2).
    norm = 2.0 / (32.0 * constants.PI ** 2)

    # ρ_density(ρ_cm) = ρ_cm × L_eff(ρ_cm). The integrand already carries one
    # factor of ρ inside h (see step_profile_mode_integrand), so we do NOT
    # multiply by ρ again here.
    rho_density = (h.real * norm).to(dtype)
    return rho_cm, rho_density


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
    """EA^{(1)} = 2π ∫ ρ(ρ_cm) dρ_cm (trapezoidal rule, autograd-safe)."""
    dr = rho_cm[1:] - rho_cm[:-1]
    avg = 0.5 * (rho_density[1:] + rho_density[:-1])
    return (2.0 * constants.PI * torch.sum(avg * dr)).squeeze()
