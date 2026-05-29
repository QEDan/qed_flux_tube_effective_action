import torch
import pytest
from src.python.step_profile_effective_action import step_profile_analytic_ea

def test_step_profile_analytic_ea_runs():
    F_cal = torch.tensor(1.0, dtype=torch.float64)
    lambd = torch.tensor(1.0, dtype=torch.float64)
    ea = step_profile_analytic_ea(F_cal, lambd)
    assert isinstance(ea, torch.Tensor)
    assert torch.isfinite(ea).item()
    print(f"Effective action computed: {ea.item()}")

def test_step_profile_analytic_ea_integer_check():
    F_cal = torch.tensor(1.5, dtype=torch.float64)
    lambd = torch.tensor(1.0, dtype=torch.float64)
    with pytest.raises(NotImplementedError):
        step_profile_analytic_ea(F_cal, lambd)
