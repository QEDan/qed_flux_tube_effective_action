import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src/python'))

import torch
import numpy as np
import pytest
from renormalization import Renormalizer
from profiles import StepFunctionProfile

def test_g0_computation():
    renorm = Renormalizer(device="cpu")
    chi = torch.tensor([1.0, 2.0], dtype=torch.complex128)
    ml = torch.tensor([0, 1], dtype=torch.int32)
    m = 1.0
    rho = torch.linspace(0.1, 1.0, 10, dtype=torch.float64)
    
    g0 = renorm.compute_g0(chi, ml, m, rho)
    assert g0.shape == (2, 10)
    assert torch.all(torch.isfinite(g0))

def test_uv_subtraction_shape():
    renorm = Renormalizer(device="cpu")
    chi = torch.tensor([1.0, 2.0], dtype=torch.complex128)
    ml = torch.tensor([0, 1], dtype=torch.int32)
    m = 1.0
    rho = torch.linspace(0.1, 1.0, 10, dtype=torch.float64)
    profile = StepFunctionProfile(rho, lambd=0.5, F=1.0)
    
    uv_sub = renorm.compute_uv_subtraction(chi, ml, m, rho, profile)
    assert uv_sub.shape == (2, 10)
    assert torch.all(torch.isfinite(uv_sub))

def test_uv_subtraction_zero_field():
    renorm = Renormalizer(device="cpu")
    chi = torch.tensor([1.0], dtype=torch.complex128)
    ml = torch.tensor([0], dtype=torch.int32)
    m = 1.0
    rho = torch.linspace(0.1, 1.0, 10, dtype=torch.float64)
    # Zero flux F=0
    profile = StepFunctionProfile(rho, lambd=0.5, F=0.0)
    
    uv_sub = renorm.compute_uv_subtraction(chi, ml, m, rho, profile)
    # UV sub should be zero if B=0
    assert torch.allclose(uv_sub, torch.zeros_like(uv_sub))
