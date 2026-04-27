import numpy as np
from scipy.special import hyp1f1, hyperu, gamma

def M_whittaker(z, kappa, mu):
    """
    Whittaker M function: M_{kappa, mu}(z)
    M_{k,m}(z) = z^{m+1/2} e^{-z/2} 1F1(1/2+m-k, 1+2m, z)
    """
    return z**(mu + 0.5) * np.exp(-0.5 * z) * hyp1f1(0.5 + mu - kappa, 1 + 2 * mu, z)

def W_whittaker(z, kappa, mu):
    """
    Whittaker W function: W_{kappa, mu}(z)
    W_{k,m}(z) = z^{m+1/2} e^{-z/2} U(1/2+m-k, 1+2m, z)
    """
    return z**(mu + 0.5) * np.exp(-0.5 * z) * hyperu(0.5 + mu - kappa, 1 + 2 * mu, z)

def get_step_function_params(chi, ml, sigma3, m, lambd, F):
    """
    Calculate intermediate parameters for step function analytic solution.
    """
    # F is the total flux, F_dim is e*F / (2*pi)
    F_dim = F / (2.0 * np.pi)
    
    # k^2 in the interior
    k2 = chi**2 - m**2 - (2.0 * F_dim / lambd**2) * (sigma3 - ml)
    
    # Whittaker parameters
    kappa = (lambd**2 * k2) / (4.0 * F_dim)
    mu = ml / 2.0
    
    return F_dim, k2, kappa, mu

def get_interior_solutions(rho, chi, ml, sigma3, m, lambd, F):
    """
    Compute analytic solutions u0 (regular at 0) and uinf (regular at infinity)
    for the interior region (rho < lambd) of a step function flux tube.
    
    u0(rho) = M_{kappa, mu}(z) / rho
    uinf(rho) = W_{kappa, mu}(z) / rho
    where z = (F_dim / lambd^2) * rho^2
    """
    F_dim, k2, kappa, mu = get_step_function_params(chi, ml, sigma3, m, lambd, F)
    
    z = (F_dim / lambd**2) * rho**2
    
    u0 = M_whittaker(z, kappa, mu) / rho
    uinf = W_whittaker(z, kappa, mu) / rho
    
    return u0, uinf

def get_analytic_wronskian(chi, ml, sigma3, m, lambd, F):
    """
    Compute analytic Wronskian W0 = rho * (u0' * uinf - u0 * uinf')
    for the interior solutions.
    
    Using M, W Wronskian properties:
    M' W - M W' = gamma(1+2*mu) / gamma(1/2+mu-kappa)
    W0 = (2 * F_dim / lambd^2) * (M' W - M W')
    """
    F_dim, k2, kappa, mu = get_step_function_params(chi, ml, sigma3, m, lambd, F)
    
    # M W' - M' W = -gamma(1+2*mu) / gamma(1/2+mu-kappa)
    # So M' W - M W' = gamma(1+2*mu) / gamma(1/2+mu-kappa)
    wronskian_mw = gamma(1 + 2 * mu) / gamma(0.5 + mu - kappa)
    
    W0 = (2.0 * F_dim / lambd**2) * wronskian_mw
    return W0

def get_exterior_solutions(rho, chi, ml, sigma3, m, lambd, F):
    """
    Bessel function solutions for the exterior region (rho > lambd).
    Placeholder for now as per step_plan.md Task 2 focus on Whittaker.
    """
    # n = ml - F / (2*pi)
    # k_ext = sqrt(chi^2 - m^2)
    # u = J_n(k_ext * rho) or Y_n(k_ext * rho)
    raise NotImplementedError("Exterior analytic solutions not yet fully implemented.")
