"""
PyTorch special functions for differentiable step-profile analytics.

Whittaker M/W follow DLMF 13.14.13–14 via confluent hypergeometric functions.
This implementation provides vectorized, differentiable versions in log-space
to support massive resolution increases and auto-diffable effective actions.
"""

from __future__ import annotations

import math
from typing import Union

import mpmath
import numpy as np
import torch
from scipy.special import jv, jvp, yv, yvp, iv, ivp, kv, kvp, ive, kve

Number = Union[float, complex, torch.Tensor]


def _lgamma_torch(z: torch.Tensor) -> torch.Tensor:
    """log(Gamma(z)); real via torch.lgamma, complex via mpmath with autograd."""
    if z.is_complex() or z.requires_grad:
        return _LGammaAutograd.apply(z)
    return torch.lgamma(z)


class _LGammaAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor):
        ctx.save_for_backward(z)
        flat = z.detach().reshape(-1)
        # Always use complex for internal mpmath to avoid TypeError and handle negative reals
        vals = [complex(mpmath.loggamma(complex(v))) for v in flat]
        
        if z.is_complex():
            dtype = z.dtype
            res = torch.tensor(vals, dtype=dtype, device=z.device)
        else:
            # If input was real, the result might be complex (for negative z)
            # or it might be mostly real. Let's keep it complex if any imag part is significant.
            res_c = torch.tensor(vals, dtype=torch.complex128, device=z.device)
            if torch.all(torch.abs(res_c.imag) < 1e-15):
                res = res_c.real
            else:
                res = res_c
        return res.reshape(z.shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z, = ctx.saved_tensors
        flat = z.detach().reshape(-1)
        if z.is_complex() or grad_output.is_complex():
            vals = [complex(mpmath.digamma(complex(v))) for v in flat]
            dtype = torch.complex128
        else:
            vals = [float(mpmath.digamma(float(v))) for v in flat]
            dtype = torch.float64
        psi = torch.tensor(vals, dtype=dtype, device=z.device).reshape(z.shape)
        grad = grad_output * psi
        if not z.is_complex():
            grad = grad.real
        return grad


def _logsumexp_signed(logs: torch.Tensor, signs: torch.Tensor, dim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes log|sum(signs * exp(logs))| and sign(sum) in a numerically stable way.
    Supports complex logs.
    """
    mask_inf = (logs == -float('inf'))
    if torch.all(mask_inf):
        return torch.tensor(-float('inf'), dtype=logs.dtype, device=logs.device), \
               torch.tensor(0.0, dtype=logs.dtype, device=logs.device)
    
    temp_logs = torch.where(mask_inf, torch.tensor(-1e30, dtype=logs.dtype, device=logs.device), logs)
    # Stabilize using the real part of the logs
    max_log_real = torch.max(temp_logs.real, dim=dim, keepdim=True)[0]
    
    scaled_vals = signs * torch.exp(logs - max_log_real.to(logs.dtype))
    sum_val = torch.sum(scaled_vals, dim=dim)
    
    # We want log|sum| and sign(sum)
    # For complex sum, we return log|sum| (real) and sign(sum) = sum/|sum| (complex)
    abs_sum = torch.abs(sum_val)
    abs_sum_clamped = torch.clamp(abs_sum, min=1e-300)
    
    out_log = torch.log(abs_sum_clamped) + max_log_real.squeeze(dim)
    out_sign = sum_val / abs_sum_clamped
    
    mask_zero = (abs_sum < 1e-300)
    out_log = torch.where(mask_zero, torch.tensor(-float('inf'), dtype=out_log.dtype, device=out_log.device), out_log)
    out_sign = torch.where(mask_zero, torch.tensor(0.0, dtype=out_sign.dtype, device=out_sign.device), out_sign)
    
    return out_log, out_sign


def _hyp1f1_series_log(
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    max_terms: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast vectorized log of 1F1(a; b; z) by Kummer series."""
    a, b, z = torch.broadcast_tensors(a, b, z)
    shape = z.shape
    device = z.device
    
    ks = torch.arange(max_terms, device=device, dtype=torch.float64)
    ks_v = ks.view(-1, *(1 for _ in shape))
    a_v, b_v, z_v = a.unsqueeze(0), b.unsqueeze(0), z.unsqueeze(0)
    
    factors_a = a_v + ks_v
    factors_b = b_v + ks_v
    
    log_diff = torch.log(torch.abs(factors_a) + 1e-300) - torch.log(torch.abs(factors_b) + 1e-300)
    log_prefix = torch.cat([torch.zeros_like(a_v), torch.cumsum(log_diff, dim=0)[:-1]], dim=0)
    
    sign_factors = torch.sign(factors_a) * torch.sign(factors_b)
    sign_prefix = torch.cat([torch.ones_like(a_v), torch.cumprod(sign_factors, dim=0)[:-1]], dim=0)
    
    log_terms = log_prefix + ks_v * torch.log(torch.abs(z_v) + 1e-300) - _lgamma_torch(ks_v + 1.0)
    sign_terms = sign_prefix * (torch.sign(z_v)**ks_v)
    
    if torch.any(z == 0):
        log_terms[0] = torch.where(z_v[0] == 0, torch.zeros_like(log_terms[0]), log_terms[0])
        log_terms[1:] = torch.where(z_v[1:] == 0, torch.tensor(-float('inf'), dtype=z.dtype, device=device), log_terms[1:])

    return _logsumexp_signed(log_terms, sign_terms, dim=0)


def _hyperu_series_log(
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    max_terms: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Log of Tricomi's U(a; b; z).
    Uses epsilon-shift trick for differentiability while maintaining stability.
    """
    eps = 1e-9
    sin_pi_b = torch.sin(torch.pi * b)
    is_near_int = torch.abs(sin_pi_b) < 1e-8
    b_eff = torch.where(is_near_int, b + eps, b)
    
    def log_gamma_signed(x):
        l = _lgamma_torch(x)
        s = torch.where(x > 0, torch.ones_like(x), torch.sign(torch.sin(torch.pi * x)))
        mask_pole = torch.abs(torch.sin(torch.pi * x)) < 1e-15
        s = torch.where(mask_pole, torch.zeros_like(s), s)
        return l, s

    lg_b, sg_b = log_gamma_signed(b_eff)
    lg_1ab, sg_1ab = log_gamma_signed(1.0 + a - b_eff)
    log_m1, sign_m1 = _hyp1f1_series_log(a, b_eff, z, max_terms=max_terms)
    
    lg_a, sg_a = log_gamma_signed(a)
    lg_2b, sg_2b = log_gamma_signed(2.0 - b_eff)
    log_m2, sign_m2 = _hyp1f1_series_log(1.0 + a - b_eff, 2.0 - b_eff, z, max_terms=max_terms)
    
    log_t1 = log_m1 - lg_1ab - lg_b
    sign_t1 = sign_m1 * sg_1ab * sg_b
    
    log_t2 = (1.0 - b_eff) * torch.log(z) + log_m2 - lg_a - lg_2b
    sign_t2 = sign_m2 * sg_a * sg_2b
    
    log_diff, sign_diff = _logsumexp_signed(torch.stack([log_t1, log_t2]), 
                                            torch.stack([sign_t1, -sign_t2]), dim=0)
    
    log_sin = torch.log(torch.abs(torch.sin(torch.pi * b_eff)) + 1e-300)
    sign_sin = torch.sign(torch.sin(torch.pi * b_eff))
    
    log_u = torch.log(torch.tensor(torch.pi, dtype=z.dtype, device=z.device)) + log_diff - log_sin
    sign_u = sign_diff * sign_sin
    
    return log_u, sign_u


def whittaker_m_log(kappa: Number, mu: Number, z: Number) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (log|M|, sign(M)) using mpmath for accuracy, series for grad."""
    z_t = torch.as_tensor(z, dtype=torch.float64)
    kappa_t = torch.as_tensor(kappa, dtype=z_t.dtype, device=z_t.device)
    mu_t = torch.as_tensor(mu, dtype=z_t.dtype, device=z_t.device)
    
    if z_t.requires_grad or kappa_t.requires_grad or mu_t.requires_grad:
        a = 0.5 + mu_t - kappa_t
        b = 1.0 + 2.0 * mu_t
        log_f, sign_f = _hyp1f1_series_log(a, b, z_t)
        log_m = -0.5 * z_t + (mu_t + 0.5) * torch.log(z_t) + log_f
        return log_m, sign_f

    # High precision fallback for validation
    def element_wise(k, m, zv):
        mpmath.mp.dps = 25
        res = mpmath.whitm(float(k), float(m), float(zv))
        return float(mpmath.log(abs(res))), float(mpmath.sign(res))

    k_b, m_b, z_b = torch.broadcast_tensors(kappa_t, mu_t, z_t)
    logs = [element_wise(k, m, z)[0] for k, m, z in zip(k_b.reshape(-1), m_b.reshape(-1), z_b.reshape(-1))]
    signs = [element_wise(k, m, z)[1] for k, m, z in zip(k_b.reshape(-1), m_b.reshape(-1), z_b.reshape(-1))]
    return torch.tensor(logs, dtype=z_t.dtype, device=z_t.device).reshape(z_b.shape), \
           torch.tensor(signs, dtype=z_t.dtype, device=z_t.device).reshape(z_b.shape)


def whittaker_w_log(kappa: Number, mu: Number, z: Number) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (log|W|, sign(W)) using mpmath for accuracy, series for grad."""
    z_t = torch.as_tensor(z, dtype=torch.float64)
    kappa_t = torch.as_tensor(kappa, dtype=z_t.dtype, device=z_t.device)
    mu_t = torch.as_tensor(mu, dtype=z_t.dtype, device=z_t.device)

    if z_t.requires_grad or kappa_t.requires_grad or mu_t.requires_grad:
        a = 0.5 + mu_t - kappa_t
        b = 1.0 + 2.0 * mu_t
        log_u, sign_u = _hyperu_series_log(a, b, z_t)
        log_w = -0.5 * z_t + (mu_t + 0.5) * torch.log(z_t) + log_u
        return log_w, sign_u

    def element_wise(k, m, zv):
        mpmath.mp.dps = 25
        res = mpmath.whitw(float(k), float(m), float(zv))
        return float(mpmath.log(abs(res))), float(mpmath.sign(res))

    k_b, m_b, z_b = torch.broadcast_tensors(kappa_t, mu_t, z_t)
    logs = [element_wise(k, m, z)[0] for k, m, z in zip(k_b.reshape(-1), m_b.reshape(-1), z_b.reshape(-1))]
    signs = [element_wise(k, m, z)[1] for k, m, z in zip(k_b.reshape(-1), m_b.reshape(-1), z_b.reshape(-1))]
    return torch.tensor(logs, dtype=z_t.dtype, device=z_t.device).reshape(z_b.shape), \
           torch.tensor(signs, dtype=z_t.dtype, device=z_t.device).reshape(z_b.shape)


# --- Bessel Functions with Autograd ---

def _bessel_nu_grad(grad_output: torch.Tensor, nu_np: np.ndarray, z_np: np.ndarray, bessel_func_name: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Computes d/dnu of Bessel function via mpmath with zero-masking optimization."""
    flat_nu = nu_np.reshape(-1)
    flat_z = z_np.reshape(-1)
    flat_grad = grad_output.detach().reshape(-1)
    
    # Filter out points where grad_output is zero to save mpmath calls
    # Use a slightly larger epsilon to be safe with float precision
    mask = (torch.abs(flat_grad) > 1e-15).cpu().numpy()
    vals = np.zeros(flat_nu.shape, dtype=complex)
    
    # Select mpmath function
    func_map = {
        'jv': mpmath.besselj,
        'yv': mpmath.bessely,
        'iv': mpmath.besseli,
        'kv': mpmath.besselk
    }
    f = func_map[bessel_func_name]
    
    for idx in np.where(mask)[0]:
        n_val, z_val = flat_nu[idx], flat_z[idx]
        d_nu = complex(mpmath.diff(lambda n: f(n, complex(z_val)), n_val))
        vals[idx] = d_nu
    
    return grad_output * torch.tensor(vals, dtype=dtype, device=device).reshape(grad_output.shape)


class _BesselJScipy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu: torch.Tensor, z: torch.Tensor):
        ctx.save_for_backward(nu, z)
        nu_np, z_np = nu.detach().cpu().numpy(), z.detach().cpu().numpy()
        if np.all(np.isreal(nu_np)):
            nu_np = nu_np.real
        return torch.as_tensor(jv(nu_np, z_np), dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        nu, z = ctx.saved_tensors
        nu_np, z_np = nu.detach().cpu().numpy(), z.detach().cpu().numpy()
        n_eff = nu_np.real if np.all(np.isreal(nu_np)) else nu_np
        dz = grad_output * torch.as_tensor(jvp(n_eff, z_np, 1), dtype=z.dtype, device=z.device)
        dnu = _bessel_nu_grad(grad_output, nu_np, z_np, 'jv', z.dtype, z.device) if nu.requires_grad else None
        
        # Project to real if inputs were real to satisfy PyTorch scalar type checks
        if dnu is not None and not nu.is_complex():
            dnu = dnu.real
        if dz is not None and not z.is_complex():
            dz = dz.real
        return dnu, dz

class _BesselYScipy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu: torch.Tensor, z: torch.Tensor):
        ctx.save_for_backward(nu, z)
        nu_np, z_np = nu.detach().cpu().numpy(), z.detach().cpu().numpy()
        if np.all(np.isreal(nu_np)):
            nu_np = nu_np.real
        return torch.as_tensor(yv(nu_np, z_np), dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        nu, z = ctx.saved_tensors
        nu_np, z_np = nu.detach().cpu().numpy(), z.detach().cpu().numpy()
        n_eff = nu_np.real if np.all(np.isreal(nu_np)) else nu_np
        dz = grad_output * torch.as_tensor(yvp(n_eff, z_np, 1), dtype=z.dtype, device=z.device)
        dnu = _bessel_nu_grad(grad_output, nu_np, z_np, 'yv', z.dtype, z.device) if nu.requires_grad else None
        
        if dnu is not None and not nu.is_complex():
            dnu = dnu.real
        if dz is not None and not z.is_complex():
            dz = dz.real
        return dnu, dz

class _BesselIScipy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu: torch.Tensor, z: torch.Tensor):
        ctx.save_for_backward(nu, z)
        nu_np, z_np = nu.detach().cpu().numpy(), z.detach().cpu().numpy()
        if np.all(np.isreal(nu_np)):
            nu_np = nu_np.real
        return torch.as_tensor(iv(nu_np, z_np), dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        nu, z = ctx.saved_tensors
        nu_np, z_np = nu.detach().cpu().numpy(), z.detach().cpu().numpy()
        n_eff = nu_np.real if np.all(np.isreal(nu_np)) else nu_np
        dz = grad_output * torch.as_tensor(ivp(n_eff, z_np, 1), dtype=z.dtype, device=z.device)
        dnu = _bessel_nu_grad(grad_output, nu_np, z_np, 'iv', z.dtype, z.device) if nu.requires_grad else None
        
        if dnu is not None and not nu.is_complex():
            dnu = dnu.real
        if dz is not None and not z.is_complex():
            dz = dz.real
        return dnu, dz

class _BesselKScipy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu: torch.Tensor, z: torch.Tensor):
        ctx.save_for_backward(nu, z)
        nu_np, z_np = nu.detach().cpu().numpy(), z.detach().cpu().numpy()
        if np.all(np.isreal(nu_np)):
            nu_np = nu_np.real
        return torch.as_tensor(kv(nu_np, z_np), dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        nu, z = ctx.saved_tensors
        nu_np, z_np = nu.detach().cpu().numpy(), z.detach().cpu().numpy()
        n_eff = nu_np.real if np.all(np.isreal(nu_np)) else nu_np
        dz = grad_output * torch.as_tensor(kvp(n_eff, z_np, 1), dtype=z.dtype, device=z.device)
        dnu = _bessel_nu_grad(grad_output, nu_np, z_np, 'kv', z.dtype, z.device) if nu.requires_grad else None
        
        if dnu is not None and not nu.is_complex():
            dnu = dnu.real
        if dz is not None and not z.is_complex():
            dz = dz.real
        return dnu, dz


def bessel_jv(nu: Number, z: Number) -> torch.Tensor:
    nu_t, z_t = torch.broadcast_tensors(torch.as_tensor(nu, dtype=torch.complex128), torch.as_tensor(z, dtype=torch.complex128))
    return _BesselJScipy.apply(nu_t, z_t)

def bessel_yv(nu: Number, z: Number) -> torch.Tensor:
    nu_t, z_t = torch.broadcast_tensors(torch.as_tensor(nu, dtype=torch.complex128), torch.as_tensor(z, dtype=torch.complex128))
    return _BesselYScipy.apply(nu_t, z_t)

def bessel_iv(nu: Number, z: Number) -> torch.Tensor:
    nu_t, z_t = torch.broadcast_tensors(torch.as_tensor(nu, dtype=torch.complex128), torch.as_tensor(z, dtype=torch.complex128))
    return _BesselIScipy.apply(nu_t, z_t)

def bessel_kv(nu: Number, z: Number) -> torch.Tensor:
    nu_t, z_t = torch.broadcast_tensors(torch.as_tensor(nu, dtype=torch.complex128), torch.as_tensor(z, dtype=torch.complex128))
    return _BesselKScipy.apply(nu_t, z_t)

def bessel_i_k_product(nu: Number, z: Number) -> torch.Tensor:
    """Safe product I_nu(z) * K_nu(z) using scaled Bessel functions to avoid overflow."""
    # I_nu(z) * K_nu(z) = I_nu(z) * exp(z) * K_nu(z) * exp(-z) = Ive_nu(z) * Kve_nu(z)
    nu_t = torch.as_tensor(nu, dtype=torch.complex128)
    z_t = torch.as_tensor(z, dtype=torch.complex128)
    
    # ive and kve are scipy functions, need to wrap them for autograd if necessary
    # For now, use the scipy functions directly.
    return torch.as_tensor(ive(nu_t.detach().cpu().numpy().real, z_t.detach().cpu().numpy().real) * 
                           kve(nu_t.detach().cpu().numpy().real, z_t.detach().cpu().numpy().real), 
                           dtype=z_t.dtype, device=z_t.device)
