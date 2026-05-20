# SageMath: Rederivation of Step-Function Green's Function
from sage.all import *

# Define Symbols
rho = var('rho')
lambd = var('lambd')
kappa_sol = var('kappa_sol') # Equivalent to sqrt(chi^2 - m^2)
ml = var('ml')
F = var('F')
# Potential V = (ml - F_cal)^2 / rho^2 + ... (for rho > lambda)
# Interior: u'' + 1/rho u' - (V_int) u = 0
# Exterior: u'' + 1/rho u' - (V_ext) u = 0

# The Green's function matching at rho=lambda:
# 1. Continuity: u_in(lambda) = u_out(lambda)
# 2. Jump condition: du_out(lambda) - du_in(lambda) = - (2 * F_cal / lambd^2) * u(lambda)
#    (Note: This jump condition sign is critical)

# Let's derive matching coefficients for u0 (regular at 0) and uinf (regular at inf)
# u0_in ~ WhittakerM(z) / rho
# u0_out ~ BesselJ(n, k*rho)

# Define the jump coefficient based on current code's F_cal/lambd convention
F_cal = F / (2.0 * pi)
jump_val = (2.0 * F_cal) # Note: Removed lambd^2 based on delta-function definition at lambda

print("Symbolic derivation of jump condition and Wronskian initialized.")
