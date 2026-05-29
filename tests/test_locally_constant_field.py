import numpy as np
import pytest
import torch
from src.python import constants
from src.python.locally_constant_field import (
    heisenberg_euler_integrand,
    heisenberg_euler_density,
    const_field_heisenberg_euler_lagrangian
)
from src.python.profiles import StepFunctionProfile, SuperGaussianProfile

def test_heisenberg_euler_integrand_scalar():
    """Test the integrand for scalar inputs and small/large eBT limits."""
    m = 1.0
    e = 1.0
    T = 1.0
    
    # Small eBT limit: expansion is -1/45 * (eBT)**4
    eBT_small = 1e-4
    B_small = eBT_small / (e * T)
    val_small = heisenberg_euler_integrand(T, np.array([B_small]), m, e)[0]
    expected_small = np.exp(-m**2 * T) * (-1.0/45.0) * (eBT_small**4)
    assert np.allclose(val_small, expected_small, rtol=1e-10)
    
    # Standard value
    B = 0.5
    eBT = e * B * T
    val = heisenberg_euler_integrand(T, np.array([B]), m, e)[0]
    expected = np.exp(-m**2 * T) * ((eBT / np.tanh(eBT)) - 1.0 - (1.0/3.0)*(eBT**2))
    assert np.allclose(val, expected)

def test_heisenberg_euler_integrand_vectorization():
    """Verify that the integrand handles numpy arrays correctly."""
    B = np.array([0.0, 0.1, 0.5, 1.0])
    T = 0.5
    vals = heisenberg_euler_integrand(T, B)
    assert len(vals) == len(B)
    assert isinstance(vals, np.ndarray)
    
    for i in range(len(B)):
        scalar_val = heisenberg_euler_integrand(T, np.array([B[i]]))[0]
        assert np.isclose(vals[i], scalar_val)

def test_heisenberg_euler_lagrangian_basic():
    """Check the Lagrangian density for known limits."""
    m = constants.ELECTRON_MASS
    e = constants.ELECTRON_CHARGE
    
    # Zero field should yield zero
    assert const_field_heisenberg_euler_lagrangian(0.0) == 0.0
    
    # Small field expansion: L_HE = (eB)**4 / (360 * pi**2 * m**4)
    B_small = 1e-6
    l_small = const_field_heisenberg_euler_lagrangian(B_small, m=m, e=e)
    expected_small = (e * B_small)**4 / (360.0 * constants.PI**2 * m**4)
    assert np.allclose(l_small, expected_small, rtol=1e-5)

def test_heisenberg_euler_density_step_function():
    """
    Compare total action EA from heisenberg_euler_density against
    the analytic result for a StepFunction (constant B interior).
    """
    B_val = 0.5
    lambd = 1.0
    m = 1.0
    e = 1.0
    rho = np.linspace(0.01, 10.0, 1000)
    
    # Action EA = Integral 2*pi*rho * L(rho) d_rho
    # For StepFunction: EA = pi * lambda^2 * L_theory
    # Note: heisenberg_euler_density returns the NEGATIVE action per docstring convention.
    F = B_val * np.pi * lambd**2
    profile = StepFunctionProfile(rho, lambd=lambd, F=F, e=e)
    
    ea_numerical = heisenberg_euler_density(profile, m=m, e=e)
    l_theory = const_field_heisenberg_euler_lagrangian(B_val, m=m, e=e)
    ea_expected = - np.pi * lambd**2 * l_theory
    
    # Use 1% tolerance for numerical integration over proper time and space
    assert np.allclose(ea_numerical, ea_expected, rtol=1e-2)

def test_heisenberg_euler_density_super_gaussian():
    """
    Verify heisenberg_euler_density for a non-constant profile
    by comparing it against a manual radial integration of the Lagrangian.
    """
    B0 = 1.0
    lambd = 2.0
    m = 0.5
    e = 1.0
    rho = np.linspace(0.0, 10.0, 1000)
    
    profile = SuperGaussianProfile(rho, B0=B0, lambd=lambd, e=e)
    ea_numerical = heisenberg_euler_density(profile, m=m, e=e)
    
    # Manual radial integration
    B_vals = B0 * np.exp(-(rho / lambd)**4)
    l_vals = np.array([const_field_heisenberg_euler_lagrangian(B, m=m, e=e) for B in B_vals])
    # Expected Action (with the docstring sign convention)
    ea_expected = - np.trapz(2 * np.pi * rho * l_vals, rho)
    
    assert np.allclose(ea_numerical, ea_expected, rtol=1e-3)

def test_heisenberg_euler_density_scaling():
    """Check that the density scales correctly with mass m."""
    B0 = 0.5
    lambd = 1.0
    rho = np.linspace(0.01, 5.0, 500)
    profile = StepFunctionProfile(rho, lambd=lambd, F=B0 * np.pi * lambd**2, e=1.0)
    
    ea_m1 = heisenberg_euler_density(profile, m=1.0, e=1.0)
    ea_m2 = heisenberg_euler_density(profile, m=2.0, e=1.0)
    
    # HE Lagrangian scales as 1/m**4 for small B. 
    # For finite B it's more complex, but we expect m=2 to be much smaller.
    assert abs(ea_m2) < abs(ea_m1)
