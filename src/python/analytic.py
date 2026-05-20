import numpy as np
import mpmath
from typing import Union, Tuple, Any

# Set precision for mpmath
mpmath.mp.dps = 25

def heisenberg_euler_lagrangian(B: float, m: float = 1.0, e: float = 1.0) -> float:
    """
    Computes the exact renormalized Heisenberg-Euler Lagrangian density for a constant B field.
    Includes all orders in B, but excludes derivative terms.
    Matches the finite part of Eq 11 in Dunne & Hall (1997).
    """
    from scipy.integrate import quad
    
    if abs(B) < 1e-12:
        return 0.0
        
    # For small B, use the B^4 expansion to avoid numerical issues with quad
    # L_HE = (eB)^4 / (360 * pi^2 * m^4)
    if abs(e * B) < 0.05 * m**2:
        return (e * B)**4 / (360.0 * np.pi**2 * m**4)

    def integrand(s):
        # (s*coth(s) - 1 - s^2/3) / s^3
        # Expansion: 1 + s^2/3 - s^4/45 + 2s^6/945
        # f(s) = (1 + s^2/3 - s^4/45 - 1 - s^2/3) / s^3 = -s/45
        if s < 1e-4:
            return -s/45.0 + 2.0*s**3/945.0
        
        # Guard against large s in sinh/cosh
        if s > 100.0:
            # coth(s) -> 1
            return (s - 1.0 - s**2/3.0) / s**3 * np.exp(-m**2 * s / (e * abs(B)))
            
        return (1.0/s**3) * np.exp(-m**2 * s / (e * abs(B))) * (s/np.tanh(s) - 1.0 - s**2/3.0)
    
    # Use a finite upper bound to avoid issues with divergent integrands in quad
    # The exponential factor exp(-2s) (for m=1, B=0.5) suppresses the integrand.
    val, _ = quad(integrand, 0, 500.0)
    return - (e * B)**2 / (8.0 * np.pi**2) * val

def heisenberg_euler_integrand(Q: float, B: float, m: float = 1.0, e: float = 1.0) -> float:
    """
    Computes the renormalized Heisenberg-Euler spectral integrand:
    L_LCF = Integral Q dQ * (1/8pi^2) * [-(eBT/tanh(eBT) - 1 + 1/6(eBT)^2)]
    This integrand is finite and renormalized.
    """
    if abs(B) < 1e-12:
        return 0.0
    
    # T = 1/Q^2
    T = 1.0 / (Q**2 + 1e-15)
    eBT = e * B * T
    
    # Thesis DELO renormalized integrand: [eBT/tanh(eBT) - 1 + 1/6(eBT)^2]
    # Small eBT expansion: 0.5 * (eBT)^2 - (1/45) * (eBT)^4
    if abs(eBT) < 1e-3:
        f_val = 0.5 * (eBT**2) - (1.0/45.0) * (eBT**4)
    else:
        f_val = (eBT / np.tanh(eBT)) - 1.0 + (1.0/6.0)*(eBT**2)
        
    # Standard HE density is - (eB)^2 / 8pi^2 * Integral...
    # L_LCF = - (1/8pi^2) * Q^2 * exp(-m^2/Q^2) * f_val
    return - (1.0 / (8.0 * np.pi**2)) * Q**2 * np.exp(-m**2 / Q**2) * f_val

def derivative_correction_lagrangian(B: float, dB: float, m: float = 1.0, e: float = 1.0) -> float:
    """
    Computes the first-order derivative correction to the effective Lagrangian.
    Matches Eq 15 in Dunne & Hall (1997).
    """
    from scipy.integrate import quad
    
    if abs(B) < 1e-10:
        return 0.0
        
    def integrand(s):
        # (s*coth(s))'''
        # Small s: -8/15 * s
        if s < 1e-4:
            return -8.0/15.0 * s
        
        if s > 100.0:
            # (s*coth(s))' -> 1, (s*coth(s))'' -> 0, (s*coth(s))''' -> 0
            return 0.0
            
        # Stable form for f''' = (s*coth(s))'''
        # f = s/tanh(s)
        # f' = coth(s) - s*csch^2(s)
        # f'' = -2*csch^2(s) + 2*s*csch^2(s)*coth(s)
        # f''' = 4*csch^2(s)*coth(s) - 2*csch^2(s)*coth(s)*2*s*coth(s) - 4*s*csch^4(s)
        #      = 4*csch^2*coth - 4*s*csch^2*coth^2 - 4*s*csch^4
        cs = 1.0/np.sinh(s)
        ct = 1.0/np.tanh(s)
        f3 = 4*cs**2*ct - 4*s*cs**2*ct**2 - 4*s*cs**4
        
        return (1.0/s) * np.exp(-m**2 * s / (e * abs(B))) * f3
        
    val, _ = quad(integrand, 0, 500.0)
    return - e * (dB**2) / (64.0 * np.pi**2 * abs(B)) * val

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
    
    # Whittaker parameters: kappa = (chi^2 - m^2) / (4 * e * F_dim / lambda^2) + (ml - sigma3)/2
    # Equation derived from matching ODE to Whittaker form z'' + (-1/4 + kappa/z + (1/4-mu^2)/z^2) z = 0
    # Our ODE: u'' + 1/r u' - ((ml-eA)^2/r^2 + e*s3*B + m^2 - chi^2) u = 0
    # For interior A = (F_cal/l^2) * r, B = 2*F_cal/l^2
    # kappa = (chi^2 - m^2 - 2*e*s3*F_cal/l^2) / (4*F_cal/l^2) + ml/2 ?? 
    # Let's use the dissertation's kappa if possible.
    # Eq 2.76: k^2 = chi^2 - m^2 - 2*F_cal/l^2 * (sigma3 - ml)
    # Eq 2.73/2.74: kappa = l^2 * k^2 / (4 * F_cal)
    
    k2 = chi**2 - m**2 - (2.0 * F_cal / lambd**2) * (sigma3 - ml)
    kappa = (lambd**2 * k2) / (4.0 * F_cal)
    mu = float(ml) / 2.0
    
    return F_cal, k2, kappa, mu

def get_interior_solutions(rho: np.ndarray, chi: complex, ml: int, sigma3: int, m: float, lambd: float, F: float, e: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute analytic solutions u0 (regular at 0) and uinf (regular at infinity)
    for the interior region (rho < lambd).
    The radial wavefunction u(rho) satisfies the radial ODE.
    Whittaker M/W functions for the potential V ~ r^2 are related to u via:
    u(rho) = M_{kappa, mu}(z) / rho, where z ~ rho^2 and mu = ml/2.
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
    # Physical jump condition: du_ext - du_int = - (2*F_cal / lambd^2) * u(l)
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
    W0 = coeffs_u0[0] * (-2.0 / np.pi)
    
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

def get_analytic_wronskian(chi: complex, ml: int, sigma3: int, m: float, lambd: float, F: float, e: float = 1.0) -> complex:
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
