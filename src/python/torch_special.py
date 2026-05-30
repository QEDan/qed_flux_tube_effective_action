"""
PyTorch special functions for differentiable step-profile analytics.

Whittaker M/W follow DLMF 13.14.13–14 via confluent hypergeometric functions
(same route as SciPy's hyp1f1 / hyperu for real arguments). Bessel J/Y use
torch.special for orders 0 and 1 and standard recurrence for higher integer orders.
"""

from __future__ import annotations

import math
from typing import Union

import mpmath
import numpy as np
import torch
from scipy.special import hyp1f1, hyperu, jv, jvp, yv, yvp

Number = Union[float, complex, torch.Tensor]


def _lgamma_torch(z: torch.Tensor) -> torch.Tensor:
    """log(Gamma(z)); real via torch.lgamma, complex via mpmath."""
    if z.is_complex():
        flat = z.reshape(-1)
        vals = [complex(mpmath.loggamma(complex(v))) for v in flat]
        out = torch.tensor(vals, dtype=torch.complex128, device=z.device)
        return out.reshape(z.shape)
    return torch.lgamma(z)


def _promote_complex(
    z: torch.Tensor,
    *others: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Promote real tensors to complex when any operand is complex."""
    if z.is_complex() or any(o.is_complex() for o in others):
        z = z.to(torch.complex128)
        others = tuple(
            o.to(torch.complex128) if not o.is_complex() else o for o in others
        )
    return (z, *others)


def _hyp1f1_series(
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    max_terms: int = 256,
    tol: float = 1e-13,
) -> torch.Tensor:
    """
    Confluent hypergeometric 1F1(a; b; z) by Kummer series (DLMF 13.2.2).
    """
    z, a, b = _promote_complex(z, a, b)
    term = torch.ones_like(z)
    total = term.clone()
    for k in range(1, max_terms + 1):
        term = term * (a + (k - 1)) / (b + (k - 1)) * z / k
        total = total + term
        if k > 8:
            scale = torch.max(torch.abs(total)).clamp_min(1e-30)
            if torch.max(torch.abs(term) / scale) < tol:
                break
    return total


def _is_real_tensor(x: torch.Tensor) -> bool:
    return not x.is_complex()


class _HyperUScipy(torch.autograd.Function):
    """SciPy hyperu forward; finite-difference backward (matches scipy.special)."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b, z)
        ar, br = float(a.reshape(()).detach().cpu()), float(b.reshape(()).detach().cpu())
        z_np = z.detach().cpu().numpy()
        out = hyperu(ar, br, z_np)
        return torch.as_tensor(out, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b, z = ctx.saved_tensors
        h = 1e-8
        ar, br = float(a.reshape(()).cpu()), float(b.reshape(()).cpu())
        z_np = z.detach().cpu().numpy()
        g_np = grad_output.detach().cpu().numpy()
        f0 = hyperu(ar, br, z_np)
        da = db = 0.0
        for _ in range(z_np.size):
            da += ((hyperu(ar + h, br, z_np) - f0) / h * g_np).sum()
            db += ((hyperu(ar, br + h, z_np) - f0) / h * g_np).sum()
        dz_np = np.zeros_like(z_np, dtype=np.float64)
        it = np.nditer(z_np, flags=["multi_index"])
        for zi in it:
            idx = it.multi_index
            zi_f = float(zi)
            gi = float(g_np[idx])
            fzi = hyperu(ar, br, zi_f)
            dz_np[idx] = gi * (hyperu(ar, br, zi_f + h) - fzi) / h
        return (
            grad_output.new_tensor(da),
            grad_output.new_tensor(db),
            torch.as_tensor(dz_np, dtype=z.dtype, device=z.device),
        )


def _hyperu_kummer(
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    max_terms: int = 256,
    tol: float = 1e-13,
) -> torch.Tensor:
    """
    Tricomi confluent hypergeometric U(a, b, z) via Kummer connection (DLMF 13.2.43).
    """
    z, a, b = _promote_complex(z, a, b)
    pi = torch.tensor(math.pi, dtype=z.real.dtype, device=z.device)
    sin_pi_b = torch.sin(pi * b)

    coeff1 = torch.exp(_lgamma_torch(1 - b) - _lgamma_torch(a - b + 1))
    coeff2 = torch.exp(_lgamma_torch(b - 1) - _lgamma_torch(a))

    f1 = _hyp1f1_series(a, b, z, max_terms=max_terms, tol=tol)
    f2 = _hyp1f1_series(a - b + 1, 2 - b, z, max_terms=max_terms, tol=tol)
    z_pow = torch.pow(z, 1 - b)
    return (sin_pi_b / pi) * (coeff1 * f1 - coeff2 * z_pow * f2)


def whittaker_m(
    kappa: Number,
    mu: Number,
    z: Number,
    *,
    max_terms: int = 256,
    tol: float = 1e-13,
) -> torch.Tensor:
    """
    Whittaker M_{kappa, mu}(z) (DLMF 13.14.13).
    """
    if isinstance(z, complex):
        z_t = torch.tensor(z, dtype=torch.complex128)
    else:
        z_t = torch.as_tensor(z, dtype=torch.float64)
    kappa_t = torch.as_tensor(kappa, dtype=z_t.dtype, device=z_t.device)
    mu_t = torch.as_tensor(mu, dtype=z_t.dtype, device=z_t.device)
    if z_t.is_complex() or kappa_t.is_complex() or mu_t.is_complex():
        z_t = z_t.to(torch.complex128)
        kappa_t = kappa_t.to(torch.complex128)
        mu_t = mu_t.to(torch.complex128)

    a = 0.5 + mu_t - kappa_t
    b = 1.0 + 2.0 * mu_t
    pref = torch.pow(z_t, 0.5 + mu_t) * torch.exp(-0.5 * z_t)
    return pref * _hyp1f1_series(a, b, z_t, max_terms=max_terms, tol=tol)


def _whittaker_w_mpmath(kappa_t: torch.Tensor, mu_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """Element-wise mpmath fallback for Whittaker W (handles real or complex args)."""
    out_dtype = torch.complex128 if z_t.is_complex() else z_t.dtype
    kappa_v = complex(kappa_t.detach().cpu()) if kappa_t.is_complex() else float(kappa_t.detach().cpu())
    mu_v = complex(mu_t.detach().cpu()) if mu_t.is_complex() else float(mu_t.detach().cpu())
    if z_t.dim() == 0:
        val = complex(mpmath.whitw(kappa_v, mu_v, complex(z_t.detach().cpu())))
        if out_dtype != torch.complex128:
            val = val.real
        return torch.tensor(val, dtype=out_dtype, device=z_t.device)
    flat = z_t.reshape(-1)
    if z_t.is_complex():
        vals = [complex(mpmath.whitw(kappa_v, mu_v, complex(v))) for v in flat]
    else:
        vals = [float(mpmath.whitw(kappa_v, mu_v, float(v)).real) for v in flat]
    return torch.tensor(vals, dtype=out_dtype, device=z_t.device).reshape(z_t.shape)


def whittaker_w(
    kappa: Number,
    mu: Number,
    z: Number,
    *,
    max_terms: int = 256,
    tol: float = 1e-13,
) -> torch.Tensor:
    """
    Whittaker W_{kappa, mu}(z) (DLMF 13.14.14).

    For integer b = 1 + 2μ on real arguments, scipy.special.hyperu enters its
    logarithmic-case branch (DLMF 13.2.18) and suffers catastrophic cancellation
    in a shell around a = 1/2 + μ − κ near a non-negative integer. We route those
    cases through mpmath.whitw, which evaluates the limit correctly.
    """
    if isinstance(z, complex):
        z_t = torch.tensor(z, dtype=torch.complex128)
    else:
        z_t = torch.as_tensor(z, dtype=torch.float64)
    kappa_t = torch.as_tensor(kappa, dtype=z_t.dtype, device=z_t.device)
    mu_t = torch.as_tensor(mu, dtype=z_t.dtype, device=z_t.device)
    if z_t.is_complex() or kappa_t.is_complex() or mu_t.is_complex():
        z_t = z_t.to(torch.complex128)
        kappa_t = kappa_t.to(torch.complex128)
        mu_t = mu_t.to(torch.complex128)

    a = 0.5 + mu_t - kappa_t
    b = 1.0 + 2.0 * mu_t
    pref = torch.pow(z_t, 0.5 + mu_t) * torch.exp(-0.5 * z_t)
    if _is_real_tensor(z_t) and _is_real_tensor(a) and _is_real_tensor(b):
        b_val = float(b.detach().cpu())
        b_int = round(b_val)
        # scipy.special.hyperu is unreliable for integer b ≥ 2 (narrow log-case
        # cancellation shell for b=2, broad NaN regions for b ≥ 3). For integer
        # b we always fall back to mpmath.whitw.
        if abs(b_val - b_int) < 1e-9 and b_int >= 1:
            return _whittaker_w_mpmath(kappa_t, mu_t, z_t)
        return pref * _HyperUScipy.apply(a, b, z_t)
    if isinstance(z, complex) or isinstance(kappa, complex):
        ref = complex(mpmath.whitw(complex(kappa), float(mu), complex(z)))
        return torch.tensor(ref, dtype=torch.complex128, device=z_t.device)
    return pref * _hyperu_kummer(a, b, z_t, max_terms=max_terms, tol=tol)


class _BesselJScipy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu: torch.Tensor, z: torch.Tensor):
        nu_f = float(nu.detach().cpu())
        z_np = z.detach().cpu().numpy()
        ctx.nu = nu_f
        ctx.save_for_backward(z)
        out = jv(nu_f, z_np)
        return torch.as_tensor(out, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (z,) = ctx.saved_tensors
        z_np = z.detach().cpu().numpy()
        dz_np = jvp(ctx.nu, z_np, 1)
        dz = grad_output * torch.as_tensor(dz_np, dtype=z.dtype, device=z.device)
        return None, dz


class _BesselYScipy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu: torch.Tensor, z: torch.Tensor):
        nu_f = float(nu.detach().cpu())
        z_np = z.detach().cpu().numpy()
        ctx.nu = nu_f
        ctx.save_for_backward(z)
        out = yv(nu_f, z_np)
        return torch.as_tensor(out, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (z,) = ctx.saved_tensors
        z_np = z.detach().cpu().numpy()
        dz_np = yvp(ctx.nu, z_np, 1)
        dz = grad_output * torch.as_tensor(dz_np, dtype=z.dtype, device=z.device)
        return None, dz


def _bessel_jv_int(n: int, z: torch.Tensor) -> torch.Tensor:
    n = int(n)
    z = torch.as_tensor(z)
    if n < 0:
        n = -n
        return (-1.0) ** n * _bessel_jv_int(n, z)
    if n == 0:
        return torch.special.bessel_j0(z)
    if n == 1:
        return torch.special.bessel_j1(z)
    j_nm2 = torch.special.bessel_j0(z)
    j_nm1 = torch.special.bessel_j1(z)
    for order in range(2, n + 1):
        j_n = (2.0 * (order - 1) / z) * j_nm1 - j_nm2
        j_nm2, j_nm1 = j_nm1, j_n
    return j_nm1


def _bessel_yv_int(n: int, z: torch.Tensor) -> torch.Tensor:
    n = int(n)
    z = torch.as_tensor(z)
    if n < 0:
        n = -n
        return (-1.0) ** n * _bessel_yv_int(n, z)
    if n == 0:
        return torch.special.bessel_y0(z)
    if n == 1:
        return torch.special.bessel_y1(z)
    y_nm2 = torch.special.bessel_y0(z)
    y_nm1 = torch.special.bessel_y1(z)
    for order in range(2, n + 1):
        y_n = (2.0 * (order - 1) / z) * y_nm1 - y_nm2
        y_nm2, y_nm1 = y_nm1, y_n
    return y_nm1


def bessel_jv(nu: Number, z: Number) -> torch.Tensor:
    """
    Bessel J_nu(z). Integer orders use torch.special recurrence when z does not
    need gradients; otherwise SciPy forward with jvp backward.
    """
    z_t = torch.as_tensor(z, dtype=torch.float64)
    nu_f = float(nu) if not torch.is_tensor(nu) else float(nu.item())
    nu_t = torch.as_tensor(nu_f, dtype=z_t.dtype, device=z_t.device)
    if abs(nu_f - round(nu_f)) < 1e-12 and not z_t.requires_grad:
        return _bessel_jv_int(int(round(nu_f)), z_t)
    return _BesselJScipy.apply(nu_t, z_t)


def bessel_yv(nu: Number, z: Number) -> torch.Tensor:
    """
    Bessel Y_nu(z). Integer orders use torch.special recurrence when z does not
    need gradients; otherwise SciPy forward with yvp backward.
    """
    z_t = torch.as_tensor(z, dtype=torch.float64)
    nu_f = float(nu) if not torch.is_tensor(nu) else float(nu.item())
    nu_t = torch.as_tensor(nu_f, dtype=z_t.dtype, device=z_t.device)
    if abs(nu_f - round(nu_f)) < 1e-12 and not z_t.requires_grad:
        return _bessel_yv_int(int(round(nu_f)), z_t)
    return _BesselYScipy.apply(nu_t, z_t)


class _BesselIScipy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu: torch.Tensor, z: torch.Tensor):
        from scipy.special import iv
        nu_f = float(nu.detach().cpu())
        z_np = z.detach().cpu().numpy()
        ctx.nu = nu_f
        ctx.save_for_backward(z)
        out = iv(nu_f, z_np)
        return torch.as_tensor(out, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from scipy.special import ivp
        (z,) = ctx.saved_tensors
        z_np = z.detach().cpu().numpy()
        dz_np = ivp(ctx.nu, z_np, 1)
        dz = grad_output * torch.as_tensor(dz_np, dtype=z.dtype, device=z.device)
        return None, dz


class _BesselKScipy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu: torch.Tensor, z: torch.Tensor):
        from scipy.special import kv
        nu_f = float(nu.detach().cpu())
        z_np = z.detach().cpu().numpy()
        ctx.nu = nu_f
        ctx.save_for_backward(z)
        out = kv(nu_f, z_np)
        return torch.as_tensor(out, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from scipy.special import kvp
        (z,) = ctx.saved_tensors
        z_np = z.detach().cpu().numpy()
        dz_np = kvp(ctx.nu, z_np, 1)
        dz = grad_output * torch.as_tensor(dz_np, dtype=z.dtype, device=z.device)
        return None, dz


def bessel_iv(nu: Number, z: Number) -> torch.Tensor:
    z_t = torch.as_tensor(z, dtype=torch.float64)
    nu_f = float(nu) if not torch.is_tensor(nu) else float(nu.item())
    nu_t = torch.as_tensor(nu_f, dtype=z_t.dtype, device=z_t.device)
    return _BesselIScipy.apply(nu_t, z_t)


def bessel_kv(nu: Number, z: Number) -> torch.Tensor:
    z_t = torch.as_tensor(z, dtype=torch.float64)
    nu_f = float(nu) if not torch.is_tensor(nu) else float(nu.item())
    nu_t = torch.as_tensor(nu_f, dtype=z_t.dtype, device=z_t.device)
    return _BesselKScipy.apply(nu_t, z_t)


def whittaker_m_scipy_reference(kappa: complex, mu: float, z: complex) -> complex:
    """SciPy/hyp1f1 reference (real z only)."""
    a = 0.5 + mu - kappa
    b = 1.0 + 2.0 * mu
    zc = complex(z)
    if abs(zc.imag) > 0:
        raise ValueError("SciPy hyp1f1 reference requires real z")
    return zc ** (0.5 + mu) * np.exp(-zc / 2) * complex(hyp1f1(a, b, zc))


def whittaker_w_scipy_reference(kappa: complex, mu: float, z: complex) -> complex:
    """SciPy/hyperu reference (real a, b, z)."""
    a = 0.5 + mu - kappa
    b = 1.0 + 2.0 * mu
    zc = complex(z)
    if any(abs(v.imag) > 0 for v in (a, b, zc)):
        raise ValueError("SciPy hyperu reference requires real a, b, z")
    return zc ** (0.5 + mu) * np.exp(-zc / 2) * complex(hyperu(float(a.real), float(b.real), float(zc.real)))


def bessel_jv_scipy_reference(nu: float, z: float) -> complex:
    return complex(jv(nu, z))


def bessel_yv_scipy_reference(nu: float, z: float) -> complex:
    return complex(yv(nu, z))
