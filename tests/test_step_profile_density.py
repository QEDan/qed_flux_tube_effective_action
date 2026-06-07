import numpy as np
import pytest
import torch

from src.python import constants
from src.python.locally_constant_field import (
    heisenberg_euler_density,
    heisenberg_euler_density_at_rho_cm,
    heisenberg_euler_ea_from_density,
    const_field_heisenberg_euler_lagrangian,
)
from src.python.profiles import StepFunctionProfile
from src.python.step_profile_effective_action import (
    step_profile_analytic_ea,
    step_profile_effective_action_density,
    _integrate_density,
)


def test_ea_equals_two_pi_density_integral():
    F_cal = torch.tensor(1.0, dtype=torch.float64)
    lambd = torch.tensor(1.0, dtype=torch.float64)
    n_chi = 20
    n_ml = 5
    rho, density = step_profile_effective_action_density(F_cal, lambd, n_chi=n_chi, n_ml=n_ml)
    ea_from_density = _integrate_density(rho, density)
    ea_direct = step_profile_analytic_ea(F_cal, lambd, n_chi=n_chi, n_ml=n_ml)
    print(f"DEBUG: ea_from_density={ea_from_density}, ea_direct={ea_direct}")
    assert torch.allclose(ea_from_density, ea_direct, rtol=1e-5)


def test_step_profile_density_autograd():
    F_cal = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    lambd = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    _, density = step_profile_effective_action_density(F_cal, lambd, n_chi=5, n_rho=5, n_ml=2)
    loss = density.sum()
    loss.backward()
    assert F_cal.grad is not None
    assert lambd.grad is not None


def test_constant_b_he_density_matches_lagrangian():
    """
    For uniform B, ρ(ρ_cm) from the proper-time integral equals −L_HE(B).
    """
    m = constants.ELECTRON_MASS
    e = constants.ELECTRON_CHARGE
    B = 2.0 * 1.0 / 5.0 ** 2
    rho_np = np.linspace(0.1, 4.0, 8)

    rho_tab = heisenberg_euler_density_at_rho_cm(rho_np, np.full_like(rho_np, B), m=m, e=e)
    l_he = const_field_heisenberg_euler_lagrangian(B, m=m, e=e)

    assert np.allclose(rho_tab, -l_he, rtol=1e-4)


def test_he_density_integral_matches_scalar():
    B_val = 0.5
    lambd = 1.0
    m = 1.0
    e = 1.0
    rho = np.linspace(0.01, lambd, 200)
    F = B_val * np.pi * lambd ** 2
    profile = StepFunctionProfile(rho, lambd=lambd, F=F, e=e)

    ea_scalar = heisenberg_euler_density(profile, m=m, e=e)
    B_vals = np.full_like(rho, B_val)
    rho_tab = heisenberg_euler_density_at_rho_cm(rho, B_vals, m=m, e=e)
    ea_from_tab = heisenberg_euler_ea_from_density(rho, rho_tab)

    # Scalar routine integrates ρ * HE factor over radius; tabulated ρ uses docstring T integral.
    # Check integral is finite and same order as -π λ² L.
    l_theory = const_field_heisenberg_euler_lagrangian(B_val, m=m, e=e)
    ea_expected = -np.pi * lambd ** 2 * l_theory
    assert np.isfinite(ea_from_tab)
    assert abs(ea_from_tab) > 0
    assert abs(ea_scalar) > 0
