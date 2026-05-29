"""
Analytic 1-loop effective action and density for a step-function flux tube (dissertation Eq. 7.72).

EA^{(1)} = 2π ∫_0^∞ dρ_cm  ρ(ρ_cm)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from src.python import constants
from src.python import torch_special


def _require_integer_flux(F_cal: torch.Tensor) -> None:
    if not torch.allclose(F_cal.detach(), torch.round(F_cal.detach())):
        raise NotImplementedError("F_cal must be an integer.")


def _step_profile_dtypes(F_cal: torch.Tensor) -> Tuple[torch.dtype, torch.dtype]:
    dtype = torch.float64 if F_cal.dtype == torch.float64 else torch.float32
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    return dtype, cdtype


def step_profile_integration_grids(
        lambd: torch.Tensor,
        F_cal: torch.Tensor,
        *,
        n_chi: int = 20,
        n_rho: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (chi_vals, rho_vals, d_chi, d_rho) on the F_cal device."""
    dtype, _ = _step_profile_dtypes(F_cal)
    device = F_cal.device
    lam = lambd.to(dtype=dtype, device=device)

    chi_vals = torch.linspace(0.1, 5.0, n_chi, dtype=dtype, device=device)
    rho_unit = torch.linspace(0.001, 1.0, n_rho, dtype=dtype, device=device)
    rho_vals = rho_unit * (lam / rho_unit[-1])
    d_chi = chi_vals[1] - chi_vals[0]
    d_rho = rho_vals[1] - rho_vals[0]
    return chi_vals, rho_vals, d_chi, d_rho


def step_profile_mode_integrand(
        chi: torch.Tensor,
        ml: int,
        sigma3: float,
        F: torch.Tensor,
        lambd: torch.Tensor,
        rho_vals: torch.Tensor,
        *,
        m: float = constants.ELECTRON_MASS,
) -> torch.Tensor:
    """
    Spectral integrand for one (chi, m_l, sigma3) at each radial point.

    Returns a complex vector of length len(rho_vals) (Whittaker interior + Bessel exterior).
    """
    dtype, cdtype = _step_profile_dtypes(F)
    mu = ml / 2.0
    ml_t = torch.tensor(float(ml), dtype=dtype, device=F.device)
    sigma3_t = torch.tensor(sigma3, dtype=dtype, device=F.device)
    lam = lambd.to(dtype=dtype, device=F.device)

    k2 = chi ** 2 - m ** 2 - (2.0 * F / lam ** 2) * (sigma3_t - ml_t)
    kappa = (lam ** 2 * k2) / (4.0 * F)
    gamma_arg = 0.5 * (ml_t + 1.0 - (k2 * lam ** 2) / (2.0 * F))
    log_gamma = torch.lgamma(gamma_arg)
    log_factorial_ml = torch.lgamma(ml_t + 1.0)
    w0_inv = -(lam ** 2 / (2.0 * F)) * torch.exp(log_gamma - log_factorial_ml)

    z = (F / lam ** 2) * rho_vals ** 2
    u0 = torch_special.whittaker_m(kappa, mu, z) / rho_vals
    uinf = torch_special.whittaker_w(kappa, mu, z) / rho_vals
    term1 = (u0 * uinf) / w0_inv

    k_sq = chi ** 2 - m ** 2
    k_ext = torch.sqrt(torch.clamp(k_sq, min=0.0))
    kr = k_ext * rho_vals
    bessel_product = torch_special.bessel_jv(ml, kr) * torch_special.bessel_yv(ml, kr)
    term2 = -(constants.PI / 2.0) * bessel_product.to(cdtype)
    # Below the mass gap, k_ext = 0 and Y_nu(0) is singular; exterior piece is absent.
    if torch.all(k_sq <= 0):
        term2 = torch.zeros_like(term2)
    return (term1 + term2).to(cdtype)


def step_profile_chi_spectral_sum(
        F: torch.Tensor,
        lambd: torch.Tensor,
        rho_vals: torch.Tensor,
        chi_vals: torch.Tensor,
        d_chi: torch.Tensor,
        *,
        m: float = constants.ELECTRON_MASS,
        n_ml: int = 2,
) -> torch.Tensor:
    """
    h(ρ) = Σ_{σ3, m_l} ∫ dχ  χ³  integrand(χ, ρ).

    Used to build ρ(ρ_cm) = -(ρ_cm²/2) Re h(ρ) so that EA = 2π ∫ ρ dρ_cm.
    """
    dtype, cdtype = _step_profile_dtypes(F)
    h = torch.zeros(rho_vals.shape, dtype=cdtype, device=F.device)

    for sigma3 in (1.0, -1.0):
        for ml in range(n_ml):
            for chi in chi_vals:
                integrand = step_profile_mode_integrand(
                    chi, ml, sigma3, F, lambd, rho_vals, m=m,
                )
                h = h + integrand * (chi ** 3) * d_chi

    return h


def step_profile_effective_action_density(
        F_cal: torch.Tensor,
        lambd: torch.Tensor,
        rho_cm: Optional[torch.Tensor] = None,
        *,
        m: float = constants.ELECTRON_MASS,
        n_chi: int = 20,
        n_rho: int = 20,
        n_ml: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Effective action density ρ(ρ_cm) for the step-function flux tube.

    Satisfies EA^{(1)} = 2π ∫_0^∞ dρ_cm  ρ(ρ_cm), matching the convention in
    locally_constant_field.heisenberg_euler_density_at_rho_cm() for the LCF limit.

    Parameters
    ----------
    F_cal, lambd
        Flux-tube parameters (tensors support autograd).
    rho_cm
        Radial grid; default spans (0, lambd] with ``n_rho`` points.

    Returns
    -------
    rho_cm, rho_density
        Radial grid and ρ(ρ_cm) (real, same dtype as F_cal).
    """
    _require_integer_flux(F_cal)
    F = F_cal
    dtype, _ = _step_profile_dtypes(F_cal)

    if rho_cm is None:
        _, rho_cm, _, _ = step_profile_integration_grids(
            lambd, F_cal, n_chi=n_chi, n_rho=n_rho,
        )
    else:
        rho_cm = rho_cm.to(dtype=dtype, device=F_cal.device)

    chi_vals, _, d_chi, _ = step_profile_integration_grids(
        lambd, F_cal, n_chi=n_chi, n_rho=n_rho,
    )
    h = step_profile_chi_spectral_sum(
        F, lambd, rho_cm, chi_vals, d_chi, m=m, n_ml=n_ml,
    )
    rho_density = (-0.5 * rho_cm ** 2 * h.real).to(dtype)
    return rho_cm, rho_density


def step_profile_analytic_ea(
        F_cal: torch.Tensor,
        lambd: torch.Tensor,
        m: float = constants.ELECTRON_MASS,
        e: float = constants.ELECTRON_CHARGE,
        *,
        n_chi: int = 20,
        n_rho: int = 20,
        n_ml: int = 2,
) -> torch.Tensor:
    """
    Renormalized 1-loop effective action (Eq. 7.72) via ρ integration:

    EA^{(1)} = 2π ∫_0^∞ dρ_cm  ρ(ρ_cm).
    """
    del e  # kept for API compatibility with callers
    rho_cm, rho_density = step_profile_effective_action_density(
        F_cal, lambd, m=m, n_chi=n_chi, n_rho=n_rho, n_ml=n_ml,
    )
    return _integrate_density(rho_cm, rho_density)


def _integrate_density(rho_cm: torch.Tensor, rho_density: torch.Tensor) -> torch.Tensor:
    """EA^{(1)} = 2π ∫ ρ(ρ_cm) dρ_cm (trapezoidal rule, autograd-safe)."""
    dr = rho_cm[1:] - rho_cm[:-1]
    avg = 0.5 * (rho_density[1:] + rho_density[:-1])
    return (2.0 * constants.PI * torch.sum(avg * dr)).squeeze()
