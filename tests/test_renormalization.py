import torch
import numpy as np
import pytest
from src.python.renormalization import Renormalizer
from src.python.profiles import StepFunctionProfile

from src.python.pytorch_solver import PyTorchSolver

@pytest.fixture
def solver():
    return PyTorchSolver(device="cpu")

def test_g0_computation(solver):
    renorm = Renormalizer(device="cpu", strategy="numerical", solver=solver)
    chi = torch.tensor([1.1, 2.1], dtype=torch.complex128)
    ml = torch.tensor([0, 1], dtype=torch.int32)
    m = 1.0
    rho = torch.linspace(0.1, 1.0, 10, dtype=torch.float64)
    
    profile = StepFunctionProfile(rho, lambd=0.5, F=0.0)
    e = 1.0 # Arbitrary e for test
    g0 = renorm.compute_g0(chi, ml, m, e, rho, profile)
    assert g0.shape == (2, 10)
    assert torch.all(torch.isfinite(g0))

def test_uv_subtraction_shape(solver):
    renorm = Renormalizer(device="cpu", strategy="numerical", solver=solver)
    chi = torch.tensor([1.1, 2.1], dtype=torch.complex128)
    ml = torch.tensor([0, 1], dtype=torch.int32)
    m = 1.0
    rho = torch.linspace(0.1, 1.0, 10, dtype=torch.float64)
    profile = StepFunctionProfile(rho, lambd=0.5, F=1.0)
    
    uv_sub = renorm.compute_uv_subtraction(chi, ml, m, rho, profile)
    assert uv_sub.shape == (2, 10)
    assert torch.all(torch.isfinite(uv_sub))

def test_uv_subtraction_massive_form(solver):
    renorm = Renormalizer(device="cpu", strategy="numerical", solver=solver)
    m = 2.0
    Q = 1.5
    chi = torch.tensor([Q], dtype=torch.complex128)
    ml = torch.tensor([0], dtype=torch.int32)
    rho = torch.tensor([1.0], dtype=torch.float64)
    profile = StepFunctionProfile(rho, lambd=0.5, F=1.0)
    
    uv_sub = renorm.compute_uv_subtraction(chi, ml, m, rho, profile)
    b2 = renorm.get_b2_term(profile, rho)
    
    # Expected: - b2 / (Q^2 + m^2)^2
    expected = - b2 / (Q**2 + m**2)**2
    assert torch.allclose(uv_sub.real, expected.real, rtol=1e-6)
