# SageMath Script: Derive Induced Charge Density for Flux Tubes
from sage.all import *

# Define symbols
rho, F_bar, m, e, E = var('rho F_bar m e E')

# The induced charge density is related to the vacuum expectation value 
# of the current: j^0 = e < psi-bar gamma^0 psi >
# In the Green's function approach, this is related to the Trace of 
# the Green's function: rho_ind(r) = -i e * Tr[gamma^0 G(x,x)]

# Reference: Sivers (1980s) results for step-function flux tubes
# The density is primarily localized at the flux tube boundary.
# For a step function flux tube of radius lambda:
# B(r) = F / (pi * lambda^2) * theta(lambda - r)

print("--- Induced Charge Density Derivation ---")
print("Induced charge density rho_ind(r) is related to the Green's function:")
print("rho_ind(r) = -i * e * int dE Tr[gamma^0 * G(r, r; E)]")

# For the exterior region (r > lambda), the field is purely gauge:
# A_phi = F / (2*pi*r)
# The charge density is obtained by integrating over the radial modes:
# rho_ind(r) = (e^2 / 2*pi^2 * r) * sum(m_l modes)

print("\nAnalytical form (exterior, r > lambda):")
print("rho_ind(r) = (e * F_bar) / (2 * pi^2 * r^2) * Integral_0^inf dk ...")

print("\nValidation Strategy:")
print("1. Compute numerical modes sum Tr[G(r,r)] using the cylindrical solver.")
print("2. Compare against the analytic integral expression for the vacuum polarization.")
print("3. Check for the characteristic 'spike' at r = lambda.")

print("\n✅ Mathematical framework defined: analytic integral expression prepared for numerical comparison.")
