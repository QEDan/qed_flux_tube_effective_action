import numpy as np
import mpmath
from typing import Union, Tuple, Any

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

def get_step_function_params(chi: complex, ml: int, sigma3: int, m: float, lambd: float, F: float, e: float = 1.0) -> Tuple[float, complex, complex, float]:
    """
    Calculate intermediate parameters for step function analytic solution.
    """
    # F is the total flux, F_cal is e*F / (2*pi)
    F_cal = e * F / (2.0 * np.pi)
    
    # k^2 in the interior
    k2 = chi**2 - m**2 - (2.0 * F_cal / lambd**2) * (sigma3 - ml)
    
    # Whittaker parameters: kappa = (lambda^2 * k^2) / (4 * F_cal)
    kappa = (lambd**2 * k2) / (4.0 * F_cal)
    mu = ml / 2.0
    
    return F_cal, k2, kappa, mu

def get_interior_solutions(rho: np.ndarray, chi: complex, ml: int, sigma3: int, m: float, lambd: float, F: float, e: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute analytic solutions u0 (regular at 0) and uinf (regular at infinity)
    for the interior region (rho < lambd).
    """
    F_cal, _, kappa, mu = get_step_function_params(chi, ml, sigma3, m, lambd, F, e)
    
    z = (F_cal / lambd**2) * rho**2
    
    u0 = M_whittaker(z, kappa, mu) / rho
    uinf = W_whittaker(z, kappa, mu) / rho
    
    return u0, uinf

def get_full_analytic_solution(rho_grid: np.ndarray, chi: complex, ml: int, sigma3: int, m: float, lambd: float, F: float, e: float = 1.0) -> np.ndarray:
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
    
    # Derivatives at lambda (using Whittaker recurrence or finite difference for simplicity here)
    def d_whitm(k, m_val, z):
        return (0.5 - k/z) * mpmath.whitm(k, m_val, z) + (m_val + 0.5 + k) * mpmath.whitm(k-1, m_val, z) / z # Verification needed
    
    # Simple numerical derivative for robustness
    dz = 1e-7
    du0_int_l = (complex(mpmath.whitm(kappa, mu, z_l + dz)) / (lambd * (1 + dz/z_l)**0.5) - u0_int_l) / (dz * (2*F_cal/lambd))
    # Correct derivative of u(rho) = M(z(rho))/rho:
    # du/drho = (dM/dz * dz/drho) / rho - M/rho^2
    # dz/drho = 2 * F_cal * rho / lambd^2
    dz_drho = 2.0 * F_cal / lambd
    
    def get_u_and_du(r, is_u0=True):
        z = (F_cal / lambd**2) * r**2
        if is_u0:
            val = complex(mpmath.whitm(kappa, mu, z)) / r
            val_p = complex(mpmath.whitm(kappa, mu, z * (1+1e-8))) / (r * (1+1e-8)**0.5)
            # Actually just use mpmath diff
            dval = complex(mpmath.diff(lambda x: mpmath.whitm(kappa, mu, (F_cal/lambd**2)*x**2)/x, r))
        else:
            val = complex(mpmath.whitw(kappa, mu, z)) / r
            dval = complex(mpmath.diff(lambda x: mpmath.whitw(kappa, mu, (F_cal/lambd**2)*x**2)/x, r))
        return val, dval

    u0_l, du0_l = get_u_and_du(lambd, True)
    
    # 2. Exterior Solutions at lambda
    # Regular at infinity: we use H1 or Y depending on convention, but solver uses Y/K.
    # Dissertation Eq 2.56 uses Y_n(k Lrho) for u_inf.
    # Let's use J_n and Y_n as the basis.
    j_l = jv(n, k_ext * lambd)
    y_l = yv(n, k_ext * lambd)
    dj_l = k_ext * 0.5 * (jv(n-1, k_ext * lambd) - jv(n+1, k_ext * lambd))
    dy_l = k_ext * 0.5 * (yv(n-1, k_ext * lambd) - yv(n+1, k_ext * lambd))
    
    # 3. Matching Conditions
    # u_ext(l) = u_int(l)
    # u'_ext(l) - u'_int(l) = - (2*F_cal / lambd^2) * u(l)
    
    # For u0 (regular at origin):
    # u0_full = u0_int (rho < l)
    # u0_full = A*J_n + B*Y_n (rho > l)
    # A*J + B*Y = u0_l
    # A*DJ + B*DY = du0_l - (2*F_cal/lambd^2)*u0_l
    mat = np.array([[j_l, y_l], [dj_l, dy_l]])
    vec = np.array([u0_l, du0_l - (2.0 * F_cal / lambd**2) * u0_l])
    coeffs_u0 = np.linalg.solve(mat, vec)
    
    # For uinf (regular at infinity/boundary):
    # We define uinf_ext = Y_n (matching student's BC)
    # uinf_full = Y_n (rho > l)
    # uinf_full = C*M + D*W (rho < l)
    uinf_l = y_l
    duinf_l_ext = dy_l
    duinf_l_int = duinf_l_ext + (2.0 * F_cal / lambd**2) * uinf_l
    
    # C*M/l + D*W/l = uinf_l
    # C*(M/l)' + D*(W/l)' = duinf_l_int
    u_m_l, du_m_l = get_u_and_du(lambd, True)
    u_w_l, du_w_l = get_u_and_du(lambd, False)
    mat_inf = np.array([[u_m_l, u_w_l], [du_m_l, du_w_l]])
    vec_inf = np.array([uinf_l, duinf_l_int])
    coeffs_uinf = np.linalg.solve(mat_inf, vec_inf)
    
    # 4. Construct Full Solution
    g_full = np.zeros(len(rho_grid), dtype=complex)
    
    # Wronskian W0 = rho * (u0' * uinf - u0 * uinf')
    # Can compute in exterior: u0 = A*J + B*Y, uinf = Y
    # W = (A*J' + B*Y')*Y - (A*J + B*Y)*Y' = A(J'Y - JY') = A * (-2 / (pi * k * rho))
    # W0 = rho * W = -2*A / (pi * k)
    W0 = -2.0 * coeffs_u0[0] / (np.pi * k_ext)
    
    for i, r in enumerate(rho_grid):
        if r <= lambd:
            u0_val = get_u_and_du(r, True)[0]
            uinf_val = coeffs_uinf[0] * get_u_and_du(r, True)[0] + coeffs_uinf[1] * get_u_and_du(r, False)[0]
        else:
            u0_val = coeffs_u0[0] * jv(n, k_ext * r) + coeffs_u0[1] * yv(n, k_ext * r)
            uinf_val = yv(n, k_ext * r)
        
        g_full[i] = (u0_val * uinf_val) / W0
        
    return g_full

def get_analytic_wronskian(chi: complex, ml: int, sigma3: int, m: float, lambd: float, F: float, e: float = 1.0) -> complex:
    """
    Compute analytic Wronskian W0 = rho * (u0' * uinf - u0 * uinf')
    for the interior solutions.
    """
    F_cal, _, kappa, mu = get_step_function_params(chi, ml, sigma3, m, lambd, F, e)
    
    # M W' - M' W = -gamma(1+2*mu) / gamma(1/2+mu-kappa)
    # Use mpmath for complex gamma
    wronskian_mw = -mpmath.gamma(1 + 2 * mu) / mpmath.gamma(0.5 + mu - kappa)
    
    W0 = (2.0 * F_cal / lambd**2) * complex(wronskian_mw)
    return W0

def get_exterior_solutions(rho: np.ndarray, chi: complex, ml: int, sigma3: int, m: float, lambd: float, F: float) -> Any:
    """
    Bessel function solutions for the exterior region (rho > lambd).
    """
    raise NotImplementedError("Exterior analytic solutions not yet fully implemented.")
