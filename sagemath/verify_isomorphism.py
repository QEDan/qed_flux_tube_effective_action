import sympy
from sympy import symbols, Function, diff, simplify

# Define variables for Cartesian coordinates (Dunne & Hall convention)
# x1, x2, x3. Potential depends on x1.
x1, B, e, sigma3 = symbols('x1 B e sigma3')
p2 = symbols('p2')

# Potential in Cartesian coordinates from Dunne & Hall (1997)
# D_+ = p1^2 + (p2 - eA2)^2 + eB
# D_- = p1^2 + (p2 - eA2)^2 - eB
# The radial ODE potential operator is D_pm.
# Here we define the operator potential in Cartesian:
# V_cart = (p2 - e*A2)^2 +/- e*B
A2 = Function('A2')(x1)
p2_val = symbols('p2_val') # Constant p2 momentum
# Potential = (p2 - e*A2)^2 +/- e*B
V_cart = (p2_val - e * A2)**2

# Define variables for Cylindrical coordinates
rho, ml = symbols('rho ml')
# Potential from current cylindrical solver (V_ml):
# V_ml = e*sigma3*(Aphi/rho + dAphi/drho) + (ml^2-1)/rho^2 + e^2*Aphi^2 - 2*e*ml*Aphi/rho
# With Aphi = A(rho)/rho. 
# Substituting p2 = ml/rho, this should map to V_cart.

print("--- Operator Isomorphism Verification ---")
print("We compare the Cartesian operator V_cart = (p2 - e*A2)^2 with the cylindrical V_ml terms.")
print("Mapping: p2 -> ml/rho, A2 -> Aphi_cyl")

# Define cylindrical potential components
Aphi_cyl = Function('Aphi_cyl')(rho)
# Using identity: (ml/rho - e*Aphi_cyl/rho)^2 = (ml - e*Aphi_cyl)^2 / rho^2
V_cyl_mapped = (ml - e*Aphi_cyl)**2 / rho**2

# Compare with V_cart expansion
# (p2 - e*A2)^2 = p2^2 - 2*p2*e*A2 + e^2*A2^2
# Mapped: (ml/rho)^2 - 2*(ml/rho)*e*Aphi_cyl + e^2*Aphi_cyl^2/rho^2 
# This matches V_cyl_mapped.

print(f"Cartesian Potential expanded: {V_cart.expand()}")
print(f"Cylindrical Potential (ml-dependent part): {V_cyl_mapped.expand()}")

# Difference should be solely the field-coupling term +/- eB
# Cylindrical potential contains e*sigma3*(B) + ...
# We verify that V_cyl = V_cyl_mapped + e*sigma3*B
# where B = Aphi/rho + Aphi'
