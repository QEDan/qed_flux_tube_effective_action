import numpy as np

"""
Physical Constants in Heaviside-Lorentz Natural Units (c=hbar=1).
Dimensions are specified in brackets: [M] (mass), [L] (length), [T] (time), [1] (dimensionless).
In Natural units, [M] = [L]^-1 = [T]^-1.
"""

# Mathematical Constants
PI = np.pi
TWO_PI = 2.0 * np.pi
FOUR_PI = 4.0 * np.pi
EIGHT_PI = 8.0 * np.pi
SIXTEEN_PI = 16.0 * np.pi

# Fundamental Physical Constants
# Fine Structure Constant
FINE_STRUCTURE_CONST = 0.0072973525643
# Electron Charge
ELECTRON_CHARGE = np.sqrt(4 * PI * FINE_STRUCTURE_CONST)
# Electron Mass [M]
ELECTRON_MASS = 0.51099895  # MeV
# Compton Wavelength [L] = [M]^-1
COMPTON_WAVELENGTH = 1.0 / ELECTRON_MASS
# Flux Quantum
FLUX_QUANTUM = 2.0 * PI / ELECTRON_CHARGE

# Normalization Constants
# Scalar QED/Spinor QED spectral normalization factor [1]
# Based on 1/(8*pi^2) for HE Lagrangian spectral integration
HE_NORMALIZATION_FACTOR = 1.0 / (8.0 * PI**2)
THIRTY_TWO_PI_SQUARED = 32.0 * PI**2
EIGHT_PI_SQUARED = 8.0 * PI**2

# Field Strength Renormalization Factor [1]
# Matches 1/3 (eB)^2 density for 4 spinor states (standard QED)
FIELD_STRENGTH_RENORM = 1.0 / 3.0
