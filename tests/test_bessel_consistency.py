import numpy as np
from scipy.special import jv, yv, jvp, yvp
import pytest

def test_bessel_consistency():
    rho = 1.0
    k = 2.0
    ml = 1
    
    u0 = jv(ml, k * rho)
    uinf = yv(ml, k * rho)
    
    # W = u0 * uinf' - uinf * u0'
    # W(J, Y) = 2 / (pi * k * rho) * k = 2 / (pi * rho)
    du0 = k * jvp(ml, k * rho)
    duinf = k * yvp(ml, k * rho)
    
    W = u0 * duinf - du0 * uinf
    expected_W = 2.0 / (np.pi * rho)
    
    assert np.isclose(W, expected_W)
    
    # G = (rho * u0 * uinf) / (rho * W)
    # rho * W = 2/pi
    g_val = (rho * u0 * uinf) / (rho * W)
    expected_g = 0.5 * np.pi * rho * u0 * uinf
    
    assert np.isclose(g_val, expected_g)
