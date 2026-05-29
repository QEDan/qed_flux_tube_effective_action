import numpy as np
from scipy.integrate import quad

from src.python.profiles import FieldProfile
from src.python import constants


def heisenberg_euler_integrand(
        T: float,
        B: np.ndarray,
        m: float = constants.ELECTRON_MASS,
        e: float = constants.ELECTRON_CHARGE) -> np.ndarray:
    """
    Computes the renormalized Heisenberg-Euler integrand:
    L_LCF = (1/4pi) * Integral dT Integral rho drho * \
    [exp(-m**2 T)/T^3 * (eBT*coth(eBT) - 1 - 1/3(eBT)**2)]

    This function returns only the portion of the integrand
    in square brackets above, multiplied by exp(-m^2 T).
    """
    eBT = e * B * T

    # HE renormalized integrand part: [eBT/tanh(eBT) - 1 - 1/3(eBT)**2]
    # Small eBT expansion: -1/45 * (eBT)**4
    # Use safe division to avoid RuntimeWarning when eBT is 0
    tanh_eBT = np.tanh(eBT)
    safe_eBT_over_tanh = np.divide(eBT, tanh_eBT, out=np.ones_like(eBT), where=np.abs(eBT) > 1e-5)
    
    f_val = np.where(np.abs(eBT) < 1e-3,
                     -(1.0/45.0) * (eBT**4),
                     safe_eBT_over_tanh - 1.0 - (1.0/3.0)*(eBT**2))

    return np.exp(-m**2 * T) * f_val

def heisenberg_euler_density(
        B_profile: FieldProfile,
        m: float = constants.ELECTRON_MASS,
        e: float = constants.ELECTRON_CHARGE) -> float:
    """
    Computes the renormalized Heisenberg-Euler density:
    ρ(ρ_cm) = 1/(8π^2) ∫_0^∞ dT  e^{-m^2 T} T^{-3}
    [ eB(ρ_cm) T coth(eB(ρ_cm) T) − 1 − (1/3)(eB(ρ_cm) T)^2 ]

    so that: EA^{(1)}_{ferm} = 2π ∫_0^∞ dρ_cm  ρ(ρ_cm),
    """
    # Extract rho and B field from the profile
    rho, a_phi, da_phi = B_profile.get_arrays(as_numpy=True)

    if hasattr(B_profile, 'B_vals'):
        B_vals = B_profile.B_vals
        if hasattr(B_vals, 'detach'):
            B_vals = B_vals.detach().cpu().numpy()
    else:
        # Compute B = A/rho + dA/dr
        r_safe = np.where(rho == 0, 1e-15, rho)
        B_vals = a_phi / r_safe + da_phi

    def action_integrand(T: float) -> float:
        # Avoid singularity at T=0
        if T < 1e-12:
            return 0.0

        # Proper time integrand: exp(-m^2 T) * f_HE(eBT) / T^3
        # heisenberg_euler_integrand is vectorized for B
        igrand_vals = heisenberg_euler_integrand(T, B_vals, m, e)

        # Integrate over radial coordinate: Integral rho * f_HE d_rho
        radial_int = np.trapz(rho * igrand_vals, rho)

        return radial_int / (T**3)

    # Use quad for the proper time integration, starting from a small T > 0
    # to avoid the T=0 singularity.
    limit = 500.0 / (m**2 + 1e-15)
    total_val, _ = quad(action_integrand, 1e-12, limit, points=[1e-4, 1e-2, 1.0])

    # Factor: 2*pi (from EA integral) * 1/(8*pi^2) (from density formula) = 1/(4*pi)
    return (1.0 / (4.0 * np.pi)) * total_val


def heisenberg_euler_density_at_rho_cm(
        rho_cm: np.ndarray,
        B: np.ndarray,
        m: float = constants.ELECTRON_MASS,
        e: float = constants.ELECTRON_CHARGE) -> np.ndarray:
    """
    Local effective action density ρ(ρ_cm) in the LCF / Heisenberg–Euler approximation:

    ρ(ρ_cm) = 1/(8π²) ∫_0^∞ dT  e^{-m^2 T} T^{-3}
    [ eB(ρ_cm) T coth(eB(ρ_cm) T) − 1 − (1/3)(eB(ρ_cm) T)^2 ]

    so that EA^{(1)}_{ferm} = 2π ∫_0^∞ dρ_cm  ρ(ρ_cm).

    Parameters
    ----------
    rho_cm, B
        Radial coordinate(s) and magnetic field B(ρ_cm) (same shape, broadcastable).
    """
    rho_cm = np.atleast_1d(np.asarray(rho_cm, dtype=float))
    B = np.atleast_1d(np.asarray(B, dtype=float))
    if rho_cm.shape != B.shape:
        B = np.broadcast_to(B, rho_cm.shape)

    limit = 500.0 / (m ** 2 + 1e-15)
    quad_pts = [1e-4, 1e-2, 1.0]
    rho_out = np.zeros_like(rho_cm, dtype=float)

    for i, b_val in enumerate(B):

        def proper_time_integrand(T: float) -> float:
            if T < 1e-12:
                return 0.0
            return float(heisenberg_euler_integrand(T, np.array([b_val]), m, e)[0] / (T ** 3))

        val, _ = quad(proper_time_integrand, 1e-12, limit, points=quad_pts)
        rho_out[i] = val / (8.0 * np.pi ** 2)

    return rho_out


def heisenberg_euler_ea_from_density(
        rho_cm: np.ndarray,
        rho_density: np.ndarray) -> float:
    """EA^{(1)} = 2π ∫ ρ(ρ_cm) dρ_cm from tabulated density."""
    return float(2.0 * np.pi * np.trapz(rho_density, rho_cm))


def const_field_heisenberg_euler_lagrangian(
        B: float,
        m: float = constants.ELECTRON_MASS,
        e: float = constants.ELECTRON_CHARGE) -> float:
    """
    Computes the exact renormalized Heisenberg-Euler Lagrangian density for a constant B field.
    Includes all orders in B, but excludes derivative terms.
    Matches the finite part of Eq 11 in Dunne & Hall (1997).
    """


    if abs(B) < 1e-12:
        return 0.0

    # For small B, use the B**4 expansion to avoid numerical issues with quad
    # L_HE = (eB)**4 / (360 * pi**2 * m**4)
    if abs(e * B) < 0.05 * m ** 2:
        return (e * B) ** 4 / (360.0 * constants.PI ** 2 * m ** 4)

    def integrand(s):
        # (s*coth(s) - 1 - s**2/3) / s**3
        # Expansion: 1 + s**2/3 - s**4/45 + 2s**6/945
        # f(s) = (1 + s**2/3 - s**4/45 - 1 - s**2/3) / s**3 = -s/45
        if s < 1e-4:
            return -s / 45.0 + 2.0 * s ** 3 / 945.0

        # Guard against large s in sinh/cosh
        if s > 100.0:
            # coth(s) -> 1
            return (s - 1.0 - s ** 2 / 3.0) / s ** 3 * np.exp(-m ** 2 * s / (e * abs(B)))

        return (1.0 / s ** 3) * np.exp(-m ** 2 * s / (e * abs(B))) * (s / np.tanh(s) - 1.0 - s ** 2 / 3.0)

    # Use a finite upper bound to avoid issues with divergent integrands in quad
    # The exponential factor exp(-2s) (for m=1, B=0.5) suppresses the integrand.
    val, _ = quad(integrand, 0, 500.0)
    return - (e * B) ** 2 * constants.HE_NORMALIZATION_FACTOR * val