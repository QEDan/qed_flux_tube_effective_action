import numpy as np
from src.python import constants

def step_profile_analytic_ea(
        F_cal: float,
        lambd: float,
        m: float = constants.ELECTRON_MASS,
        e: float = constants.ELECTRON_CHARGE) -> float:
    """
    Computes the renormalized 1-loop effective action for a step-function
    flux tube (LCF approximation) according to

    Γ^(1)_int = -ℏ·π · ∑_{σ³ = {±1}} ∑_{m_l=0}^{m_l ≫ ℱ} ∫₀^∞ χ³ dχ ∫₀^λ dρ ρ² (
                        [λ² / (2ℱ)] * [Γ(1/2 * (m_l + 1 - k²λ² / (2ℱ))) / m_l!]
                        * W_{λ²k²/(4ℱ), m_l/2}(ℱρ² / λ²) * M_{λ²k²/(4ℱ), m_l/2}(ℱρ² / λ²)
                        - [π / 2] * J_{m_l}(√(χ² - m²) * ρ) * Y_{m_l}(√(χ² - m²) * ρ)
                        + (ℱ / λ²)² * [ (ρ³ / (2k²)) * sin(Θ_{m_l,k}(ρ)) + (ρ² / (6k³)) * cos(Θ_{m_l,k}(ρ)) ]
                    )
    Where Θ_{m_l,k} = 2kρ - (1/4 - m_l^2)/kρ

    For a step-function flux tube:
    B(ρ) = (2 * F_cal / e * λ^2) * θ(λ - ρ)

    Returns Γ.
    """
    if not np.isclose(F_cal, round(F_cal)):
        raise NotImplementedError("F_cal must be an integer.")

    raise NotImplementedError()
