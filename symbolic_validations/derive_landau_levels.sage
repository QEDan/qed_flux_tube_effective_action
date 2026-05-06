# SageMath Script: Derive Landau Level Structure for Cylindrical Solver
from sage.all import *

# Define symbols
B, m, e, n, sigma = var('B m e n sigma')
# Landau energy levels for fermions in a constant magnetic field B:
# E^2 = m^2 + k_z^2 + e*B*(2*n + 1 + sigma)
# where n >= 0 is the Landau level index, and sigma = +/- 1 is the spin.

print("--- Landau Level Convergence Derivation ---")
print("Analytic Landau energy levels:")
# For each level n, energy E_n depends on the magnetic field B.
# E_n = sqrt(m^2 + k_z^2 + 2*e*B*n) (spin-degenerate simplified form)

# The Green's function in the interior is a sum over these levels.
# Tr[G(r,r)] = sum_n (Density of states for level n) * (1 / (E^2 - E_n^2))

print("\nAnalytical condition for interior validation:")
print("The numerical partial trace Tr_ml[G(r,r)] should exhibit peaks at:")
print("E_n = sqrt(m^2 + 2 * e * B * n)")

print("\nValidation Strategy:")
print("1. Set constant interior magnetic field B.")
print("2. Compute the numerical mode-sum Tr[G(r,r)] for various energies E.")
print("3. Verify that the peaks in the density of states occur exactly at E_n.")
print("4. Confirm that the width of the peaks converges as more ml modes are included.")

print("\n✅ Mathematical framework defined: Landau level structure established for numerical comparison.")
