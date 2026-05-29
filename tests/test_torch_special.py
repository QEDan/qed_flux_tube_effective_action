import mpmath
import numpy as np
import pytest
import torch
from scipy.special import jv, yv

from src.python import torch_special

mpmath.mp.dps = 25


@pytest.mark.parametrize(
    "kappa,mu,z",
    [
        (0.5, 0.5, 1.2),
        (0.3, 0.25, 0.8),
        (-0.2, 1.0, 2.5),
        (0.5 + 0.3j, 0.5, 1.2 + 0.1j),
        (-0.1 + 0.2j, 0.75, 0.9 + 0.05j),
    ],
)
def test_whittaker_m_matches_references(kappa, mu, z):
    ref_mp = complex(mpmath.whitm(kappa, mu, z))
    got = torch_special.whittaker_m(kappa, mu, z)
    if got.is_complex():
        assert torch.isfinite(got.real) and torch.isfinite(got.imag)
    else:
        assert torch.isfinite(got)
    rel = abs(got - ref_mp) / max(abs(ref_mp), 1e-30)
    assert rel < 1e-9

    if abs(complex(z).imag) < 1e-15:
        ref_sp = torch_special.whittaker_m_scipy_reference(kappa, mu, z)
        assert abs(got - ref_sp) / max(abs(ref_sp), 1e-30) < 1e-10


@pytest.mark.parametrize(
    "kappa,mu,z",
    [
        (0.5, 0.5, 1.2),
        (0.3, 0.25, 0.8),
        (-0.2, 1.0, 2.5),
        (0.5 + 0.3j, 0.5, 1.2 + 0.1j),
    ],
)
def test_whittaker_w_matches_references(kappa, mu, z):
    ref_mp = complex(mpmath.whitw(kappa, mu, z))
    got = torch_special.whittaker_w(kappa, mu, z)
    rel = abs(got - ref_mp) / max(abs(ref_mp), 1e-30)
    assert rel < 1e-8

    if abs(complex(z).imag) < 1e-15:
        ref_sp = torch_special.whittaker_w_scipy_reference(kappa, mu, z)
        assert abs(got - ref_sp) / max(abs(ref_sp), 1e-30) < 1e-9


@pytest.mark.parametrize(
    "nu,z",
    [
        (0, 1.5),
        (1, 1.5),
        (2, 2.0),
        (3, 0.8),
    ],
)
def test_bessel_matches_scipy(nu, z):
    ref_j = jv(nu, z)
    ref_y = yv(nu, z)
    got_j = torch_special.bessel_jv(nu, z)
    got_y = torch_special.bessel_yv(nu, z)
    assert np.isclose(got_j.item(), ref_j, rtol=1e-12, atol=1e-12)
    assert np.isclose(got_y.item(), ref_y, rtol=1e-12, atol=1e-12)


def test_whittaker_autograd():
    kappa = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    mu = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    z = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)
    m_val = torch_special.whittaker_m(kappa, mu, z).real
    m_val.backward()
    assert kappa.grad is not None
    assert mu.grad is not None
    assert z.grad is not None
    assert not torch.allclose(kappa.grad, torch.zeros_like(kappa.grad))


def test_bessel_autograd():
    z = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    j_val = torch_special.bessel_jv(1, z)
    j_val.backward()
    assert z.grad is not None
    assert z.grad.item() != 0.0
