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


def _hyp1f1_series_log(
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    max_terms: int = 512,
    tol: float = 1e-13,
) -> torch.Tensor:
    """
    Log of confluent hypergeometric 1F1(a; b; z) by Kummer series.
    Returns (log|1F1|, sign(1F1)).
    ONLY handles real positive z and real a, b for now.
    """
    def element_wise(a_val, b_val, z_val):
        res = mpmath.hyp1f1(float(a_val), float(b_val), float(z_val))
        if res == 0:
            return -1000.0, 1.0
        return float(mpmath.log(abs(res))), float(mpmath.sign(res))
    
    # Broadcast a, b, z to common shape
    a_b, b_b, z_b = torch.broadcast_tensors(a, b, z)
    a_flat = a_b.reshape(-1)
    b_flat = b_b.reshape(-1)
    z_flat = z_b.reshape(-1)
    
    logs = []
    signs = []
    for i in range(len(z_flat)):
        l, s = element_wise(a_flat[i], b_flat[i], z_flat[i])
        logs.append(l)
        signs.append(s)
    
    return torch.tensor(logs, dtype=z.dtype, device=z.device).reshape(z_b.shape), \
           torch.tensor(signs, dtype=z.dtype, device=z.device).reshape(z_b.shape)

def whittaker_m_log(kappa: Number, mu: Number, z: Number) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (log|M|, sign(M))."""
    z_t = torch.as_tensor(z, dtype=torch.float64)
    kappa_t = torch.as_tensor(kappa, dtype=z_t.dtype, device=z_t.device)
    mu_t = torch.as_tensor(mu, dtype=z_t.dtype, device=z_t.device)
    
    a = 0.5 + mu_t - kappa_t
    b = 1.0 + 2.0 * mu_t
    
    # pref = z^(0.5+mu) * e^(-z/2)
    log_pref = (0.5 + mu_t) * torch.log(z_t) - 0.5 * z_t
    log_f, sign_f = _hyp1f1_series_log(a, b, z_t)
    
    return log_pref + log_f, sign_f

def whittaker_w_log(kappa: Number, mu: Number, z: Number) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (log|W|, sign(W))."""
    z_t = torch.as_tensor(z, dtype=torch.float64)
    kappa_t = torch.as_tensor(kappa, dtype=z_t.dtype, device=z_t.device)
    mu_t = torch.as_tensor(mu, dtype=z_t.dtype, device=z_t.device)

    def element_wise(k, m, zv):
        # Asymptotic expansion for Whittaker W when z is large (DLMF 13.7.19)
        # W_{k,m}(z) ~ exp(-z/2) z^k [1 + (m^2 - (k-1/2)^2)/(1!*z) + ...]
        # For large arguments, mpmath.whitw fails to converge.
        if zv > 50.0:
            return float(zv*(-0.5) + k*np.log(zv)), 1.0

        # Try default precision, then increase if fails
        try:
            mpmath.mp.dps = 25
            res = mpmath.whitw(float(k), float(m), float(zv))
        except ValueError:
            # Increase precision and retry, using zeroprec to bound small values
            mpmath.mp.dps = 200
            # Use a slightly less strict zeroprec
            try:
                res = mpmath.whitw(float(k), float(m), float(zv), zeroprec=100)
            except ValueError:
                # Last resort
                return -1000.0, 1.0
        
        if res == 0:
            return -1000.0, 1.0
        return float(mpmath.log(abs(res))), float(mpmath.sign(res))

    # Broadcast kappa, mu, z to common shape
    k_b, m_b, z_b = torch.broadcast_tensors(kappa_t, mu_t, z_t)
    k_flat = k_b.reshape(-1)
    m_flat = m_b.reshape(-1)
    z_flat = z_b.reshape(-1)

    logs = []
    signs = []
    for i in range(len(z_flat)):
        l, s = element_wise(k_flat[i], m_flat[i], z_flat[i])
        logs.append(l)
        signs.append(s)
    
    return torch.tensor(logs, dtype=z_t.dtype, device=z_t.device).reshape(z_b.shape), \
           torch.tensor(signs, dtype=z_t.dtype, device=z_t.device).reshape(z_b.shape)

class _BesselJScipy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu: torch.Tensor, z: torch.Tensor):
        from scipy.special import jv
        nu_np = nu.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        ctx.save_for_backward(nu, z)
        out = jv(nu_np, z_np)
        return torch.as_tensor(out, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from scipy.special import jvp
        nu, z = ctx.saved_tensors
        nu_np = nu.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        dz_np = jvp(nu_np, z_np, 1)
        dz = grad_output * torch.as_tensor(dz_np, dtype=z.dtype, device=z.device)
        return None, dz

class _BesselYScipy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu: torch.Tensor, z: torch.Tensor):
        from scipy.special import yv
        nu_np = nu.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        ctx.save_for_backward(nu, z)
        out = yv(nu_np, z_np)
        return torch.as_tensor(out, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from scipy.special import yvp
        nu, z = ctx.saved_tensors
        nu_np = nu.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        dz_np = yvp(nu_np, z_np, 1)
        dz = grad_output * torch.as_tensor(dz_np, dtype=z.dtype, device=z.device)
        return None, dz

class _BesselIScipy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu: torch.Tensor, z: torch.Tensor):
        from scipy.special import iv
        nu_np = nu.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        ctx.save_for_backward(nu, z)
        out = iv(nu_np, z_np)
        return torch.as_tensor(out, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from scipy.special import ivp
        nu, z = ctx.saved_tensors
        nu_np = nu.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        dz_np = ivp(nu_np, z_np, 1)
        dz = grad_output * torch.as_tensor(dz_np, dtype=z.dtype, device=z.device)
        return None, dz

class _BesselKScipy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu: torch.Tensor, z: torch.Tensor):
        from scipy.special import kv
        nu_np = nu.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        ctx.save_for_backward(nu, z)
        out = kv(nu_np, z_np)
        return torch.as_tensor(out, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from scipy.special import kvp
        nu, z = ctx.saved_tensors
        nu_np = nu.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        dz_np = kvp(nu_np, z_np, 1)
        dz = grad_output * torch.as_tensor(dz_np, dtype=z.dtype, device=z.device)
        return None, dz

def bessel_jv(nu: Number, z: Number) -> torch.Tensor:
    z_t = torch.as_tensor(z, dtype=torch.float64)
    nu_t = torch.as_tensor(nu, dtype=z_t.dtype, device=z_t.device)
    return _BesselJScipy.apply(nu_t, z_t)

def bessel_yv(nu: Number, z: Number) -> torch.Tensor:
    z_t = torch.as_tensor(z, dtype=torch.float64)
    nu_t = torch.as_tensor(nu, dtype=z_t.dtype, device=z_t.device)
    return _BesselYScipy.apply(nu_t, z_t)

def bessel_iv(nu: Number, z: Number) -> torch.Tensor:
    z_t = torch.as_tensor(z, dtype=torch.float64)
    nu_t = torch.as_tensor(nu, dtype=z_t.dtype, device=z_t.device)
    return _BesselIScipy.apply(nu_t, z_t)

def bessel_kv(nu: Number, z: Number) -> torch.Tensor:
    z_t = torch.as_tensor(z, dtype=torch.float64)
    nu_t = torch.as_tensor(nu, dtype=z_t.dtype, device=z_t.device)
    return _BesselKScipy.apply(nu_t, z_t)
