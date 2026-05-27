import numpy as np
import mpmath
from typing import Union, Tuple, Any
from src.python import constants

# Set precision for mpmath
mpmath.mp.dps = 25

def M_whittaker(z: Union[float, np.ndarray], kappa: complex, mu: float) -> Union[complex, np.ndarray]:
    """
    Whittaker M function: M_{kappa, mu}(z)
    """
    def single_val(zv: complex) -> complex:
        res = mpmath.whitm(kappa, mu, zv)
        return complex(res)
        
    if isinstance(z, np.ndarray):
        return np.array([single_val(zv) for zv in z])
    return single_val(z)

def W_whittaker(z: Union[float, np.ndarray], kappa: complex, mu: float) -> Union[complex, np.ndarray]:
    """
    Whittaker W function: W_{kappa, mu}(z)
    """
    def single_val(zv: complex) -> complex:
        res = mpmath.whitw(kappa, mu, zv)
        return complex(res)
        
    if isinstance(z, np.ndarray):
        return np.array([single_val(zv) for zv in z])
    return single_val(z)

def get_step_function_params(chi: complex, ml: int, sigma3: int, m: float, lambd: float, F: float, e: float = constants.ELECTRON_CHARGE) -> Tuple[float, complex, complex, float]:
    """
    Calculate intermediate parameters for step function analytic solution.
    """
    # F is the total flux, F_cal is e*F / (2*pi)
    F_cal = e * F / (constants.TWO_PI)
    
    # Whittaker parameters: kappa = (chi**2 - m**2) / (4 * e * F_dim / lambda**2) + (ml - sigma3)/2
    # Equation derived from matching ODE to Whittaker form z'' + (-1/4 + kappa/z + (1/4-mu**2)/z**2) z = 0
    # Our ODE: u'' + 1/r u' - ((ml-eA)**2/r**2 + e*s3*B + m**2 - chi**2) u = 0
    # For interior A = (F_cal/l**2) * r, B = 2*F_cal/l**2
    # kappa = (chi**2 - m**2 - 2*e*s3*F_cal/l**2) / (4*F_cal/l**2) + ml/2 ?? 
    # Let's use the dissertation's kappa if possible.
    # Eq 2.76: k**2 = chi**2 - m**2 - 2*F_cal/l**2 * (sigma3 - ml)
    # Eq 2.73/2.74: kappa = l**2 * k**2 / (4 * F_cal)
    
    k2 = chi**2 - m**2 - (2.0 * F_cal / lambd**2) * (sigma3 - ml)
    kappa = (lambd**2 * k2) / (4.0 * F_cal)
    mu = float(ml) / 2.0
    
    return F_cal, k2, kappa, mu

def get_interior_solutions(rho: np.ndarray, chi: complex, ml: int, sigma3: int, m: float, lambd: float, F: float, e: float = constants.ELECTRON_CHARGE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute analytic solutions u0 (regular at 0) and uinf (regular at infinity)
    for the interior region (rho < lambd).
    The radial wavefunction u(rho) satisfies the radial ODE.
    Whittaker M/W functions for the potential V ~ r**2 are related to u via:
    u(rho) = M_{kappa, mu}(z) / rho, where z ~ rho**2 and mu = ml/2.
    """
    F_cal, _, kappa, mu = get_step_function_params(chi, ml, sigma3, m, lambd, F, e)

    z = (F_cal / lambd**2) * rho**2

    u0 = M_whittaker(z, kappa, mu) / rho
    uinf = W_whittaker(z, kappa, mu) / rho

    return u0, uinf
def get_full_analytic_solution(rho_grid: np.ndarray, chi: complex, ml: int, sigma3: int, m: float, lambd: float, F: float, e: float = constants.ELECTRON_CHARGE) -> np.ndarray:
    """
    Computes the full analytic Green's function G(rho, rho) for the step function profile
    by matching interior (Whittaker) and exterior (Bessel) solutions at rho = lambd.
    """
    from scipy.special import jv, yv
    
    F_cal, k2_int, kappa, mu = get_step_function_params(chi, ml, sigma3, m, lambd, F, e)
    k_ext = np.sqrt(chi**2 - m**2 + 0j)
    n = ml - F_cal
    
    # 1. Interior Solutions (at lambda)
    # Use mpmath for precision at the matching point
    z_l = mpmath.mpf(F_cal) / (mpmath.mpf(lambd)**2) * (mpmath.mpf(lambd)**2) # = F_cal
    z_l = complex(z_l)
    
    u0_int_l = complex(mpmath.whitm(kappa, mu, z_l)) / lambd
    uinf_int_l = complex(mpmath.whitw(kappa, mu, z_l)) / lambd
    
    # Derivatives at lambda
    def get_u_and_du(r, is_u0=True):
        z = (F_cal / lambd**2) * r**2
        if is_u0:
            val = complex(mpmath.whitm(kappa, mu, z)) / r
            dval = complex(mpmath.diff(lambda x: mpmath.whitm(kappa, mu, (F_cal/lambd**2)*x**2)/x, r))
        else:
            val = complex(mpmath.whitw(kappa, mu, z)) / r
            dval = complex(mpmath.diff(lambda x: mpmath.whitw(kappa, mu, (F_cal/lambd**2)*x**2)/x, r))
        return val, dval

    u0_l, du0_l = get_u_and_du(lambd, True)
    
    # 2. Exterior Solutions at lambda
    j_l = jv(n, k_ext * lambd)
    y_l = yv(n, k_ext * lambd)
    dj_l = k_ext * 0.5 * (jv(n-1, k_ext * lambd) - jv(n+1, k_ext * lambd))
    dy_l = k_ext * 0.5 * (yv(n-1, k_ext * lambd) - yv(n+1, k_ext * lambd))
    
    # 3. Matching Conditions
    # Physical jump condition: du_ext - du_int = - (2*F_cal / lambd**2) * u(l)
    jump_val = (2.0 * F_cal / lambd**2)
    
    # For u0 (regular at origin):
    # du_ext = du_int - jump
    mat = np.array([[j_l, y_l], [dj_l, dy_l]])
    vec = np.array([u0_l, du0_l - jump_val * u0_l])
    coeffs_u0 = np.linalg.solve(mat, vec)

    # For uinf (regular at infinity/boundary):
    # du_int = du_ext + jump
    uinf_l = y_l
    duinf_l_ext = dy_l
    duinf_l_int = duinf_l_ext + jump_val * uinf_l

    u_m_l, du_m_l = get_u_and_du(lambd, True)
    u_w_l, du_w_l = get_u_and_du(lambd, False)
    mat_inf = np.array([[u_m_l, u_w_l], [du_m_l, du_w_l]])
    vec_inf = np.array([uinf_l, duinf_l_int])
    coeffs_uinf = np.linalg.solve(mat_inf, vec_inf)

    # 4. Construct Full Solution
    g_full = np.zeros(len(rho_grid), dtype=complex)
    
    # W0 = rho * (u0' * uinf - u0 * uinf')
    # Using rho * W(A*J+B*Y, Y) = A * rho * W(J, Y) = A * (-2/pi)
    W0 = coeffs_u0[0] * (-2.0 / constants.PI)
    
    for i, r in enumerate(rho_grid):
        if r <= lambd:
            u0_val, _ = get_u_and_du(r, True)
            uinf_val = coeffs_uinf[0] * get_u_and_du(r, True)[0] + coeffs_uinf[1] * get_u_and_du(r, False)[0]
        else:
            u0_val = coeffs_u0[0] * jv(n, k_ext * r) + coeffs_u0[1] * yv(n, k_ext * r)
            uinf_val = yv(n, k_ext * r)
        
        # Consistent with PyTorchSolver: G = - r * u0 * uinf / W0
        g_full[i] = - (r * u0_val * uinf_val) / W0
        
    return g_full

def get_analytic_wronskian(chi: complex, ml: int, sigma3: int, m: float, lambd: float, F: float, e: float = constants.ELECTRON_CHARGE) -> complex:
    """
    Compute analytic Wronskian W0 = rho * (u0' * uinf - u0 * uinf')
    for the interior solutions.
    """
    F_cal, _, kappa, mu = get_step_function_params(chi, ml, sigma3, m, lambd, F, e)
    
    # M' W - M W' identity
    # DLMF 13.14.30: W{M, W} = M W' - M' W = -gamma(1+2mu)/gamma(1/2+mu-kappa)
    # Our W0 = rho * (u0' uinf - u0 uinf')
    # The solver yields a positive Wronskian for the standard case.
    wronskian_mw = -mpmath.gamma(1 + 2 * mu) / mpmath.gamma(0.5 + mu - kappa)
    
    W0 = (2.0 * F_cal / lambd**2) * complex(wronskian_mw)
    return W0

def get_exterior_solutions(rho: np.ndarray, chi: complex, ml: int, sigma3: int, m: float, lambd: float, F: float) -> Any:
    """
    Bessel function solutions for the exterior region (rho > lambd).
    """
    raise NotImplementedError("Exterior analytic solutions not yet fully implemented.")
