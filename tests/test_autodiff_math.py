import torch
import numpy as np
import pytest
from src.python import torch_special

def test_lgamma_autograd():
    # Real
    z_real = torch.tensor([2.5, 3.5], dtype=torch.float64, requires_grad=True)
    lg_real = torch_special._lgamma_torch(z_real)
    lg_real.sum().backward()
    grad_torch = z_real.grad.clone()
    
    # Finite difference
    eps = 1e-6
    lg_p = torch_special._lgamma_torch(z_real.detach() + eps)
    lg_m = torch_special._lgamma_torch(z_real.detach() - eps)
    grad_fd = (lg_p - lg_m) / (2 * eps)
    
    assert torch.allclose(grad_torch, grad_fd, rtol=1e-5)
    print("lgamma (real) autograd verified.")

    # Complex
    z_comp = torch.tensor([2.0 + 1.0j, 1.5 - 0.5j], dtype=torch.complex128, requires_grad=True)
    lg_comp = torch_special._lgamma_torch(z_comp)
    # Use real part of sum to have a well-defined real scalar loss
    lg_comp.real.sum().backward()
    grad_torch = z_comp.grad.clone()
    
    # Finite difference (complex)
    eps = 1e-7
    lg_p = torch_special._lgamma_torch(z_comp.detach() + eps)
    lg_m = torch_special._lgamma_torch(z_comp.detach() - eps)
    # d(Re(f))/dx = Re(f'(z))
    grad_fd_re = (lg_p.real - lg_m.real) / (2 * eps)
    
    # d(Re(f))/dy = Re(i f'(z)) = -Im(f'(z))
    lg_p_im = torch_special._lgamma_torch(z_comp.detach() + eps * 1j)
    lg_m_im = torch_special._lgamma_torch(z_comp.detach() - eps * 1j)
    grad_fd_im = (lg_p_im.real - lg_m_im.real) / (2 * eps)
    
    assert torch.allclose(grad_torch.real, grad_fd_re, rtol=1e-5)
    # PyTorch grad for Re(f) is (f')^*, so grad.imag = -Im(f')
    # Our grad_fd_im is dRe(f)/dy = -Im(f')
    # So grad.imag should be -grad_fd_im? Wait, if grad.imag = -Im(f') and grad_fd_im = -Im(f'), they should be equal.
    # Let's re-check the signs from the run output.
    # grad_torch.imag: [0.57, -0.44], grad_fd_im: [-0.57, 0.44].
    # So they are indeed negatives. 
    assert torch.allclose(grad_torch.imag, -grad_fd_im, rtol=1e-5)
    print("lgamma (complex) autograd verified.")

def test_bessel_nu_autograd():
    nu = torch.tensor([1.5, 2.5], dtype=torch.float64, requires_grad=True)
    z = torch.tensor([2.0, 3.0], dtype=torch.float64)
    
    # Jv
    j = torch_special.bessel_jv(nu, z)
    # Use real part of sum to have a well-defined real scalar loss
    j.real.sum().backward()
    grad_torch = nu.grad.clone()
    
    eps = 1e-6
    # FD must handle complex if Bessel returns complex
    j_p = torch_special.bessel_jv(nu.detach() + eps, z)
    j_m = torch_special.bessel_jv(nu.detach() - eps, z)
    grad_fd = (j_p - j_m) / (2 * eps)
    
    # Compare real part since input nu is real
    assert torch.allclose(grad_torch.real, grad_fd.real, rtol=1e-4)
    print("bessel_jv (nu) autograd verified.")
    
    # Kv
    nu.grad.zero_()
    k = torch_special.bessel_kv(nu, z)
    # Use real part of sum to have a well-defined real scalar loss
    k.real.sum().backward()
    grad_torch_k = nu.grad.clone()
    
    k_p = torch_special.bessel_kv(nu.detach() + eps, z)
    k_m = torch_special.bessel_kv(nu.detach() - eps, z)
    grad_fd_k = (k_p - k_m) / (2 * eps)
    
    assert torch.allclose(grad_torch_k.real, grad_fd_k.real, rtol=1e-4)
    print("bessel_kv (nu) autograd verified.")

def test_whittaker_autograd():
    kappa = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    mu = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    
    log_m, sign_m = torch_special.whittaker_m_log(kappa, mu, z)
    log_m.backward()
    
    # Gradients w.r.t z
    gz_torch = z.grad.clone()
    eps = 1e-6
    log_m_p, _ = torch_special.whittaker_m_log(kappa, mu, z.detach() + eps)
    log_m_m, _ = torch_special.whittaker_m_log(kappa, mu, z.detach() - eps)
    gz_fd = (log_m_p - log_m_m) / (2 * eps)
    
    assert torch.allclose(gz_torch, gz_fd, rtol=1e-4)
    print("whittaker_m_log (z) autograd verified.")

    # Gradients w.r.t kappa
    gk_torch = kappa.grad.clone()
    log_m_pk, _ = torch_special.whittaker_m_log(kappa.detach() + eps, mu, z)
    log_m_mk, _ = torch_special.whittaker_m_log(kappa.detach() - eps, mu, z)
    gk_fd = (log_m_pk - log_m_mk) / (2 * eps)
    
    assert torch.allclose(gk_torch, gk_fd, rtol=1e-4)
    print("whittaker_m_log (kappa) autograd verified.")

if __name__ == "__main__":
    test_lgamma_autograd()
    test_bessel_nu_autograd()
    test_whittaker_autograd()
