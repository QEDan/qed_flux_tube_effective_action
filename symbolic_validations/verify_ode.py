import sympy
from sympy import symbols, Function, diff, pi

rho, m, ml, e, F, lambd, sigma3 = symbols('rho m ml e F lambd sigma3')
u = Function('u')(rho)
Aphi = Function('Aphi')(rho)

# Define B-field and V_ml
B = Aphi/rho + diff(Aphi, rho)
V_ml = e * sigma3 * B + (ml**2 - 1)/rho**2 + e**2 * Aphi**2 - 2 * e * ml * Aphi / rho

# Verify Step function profile
f = (F / (2*pi)) * (rho**2 / lambd**2)
Aphi_step = f / rho
B_step = Aphi_step/rho + diff(Aphi_step, rho)
print(f"B-field for step profile: {B_step.simplify()}")

# Radial ODE: u'' + 1/rho u' - (1/rho^2 + V_ml) u = 0
ODE = diff(u, rho, 2) + (1/rho)*diff(u, rho) - (1/rho**2 + V_ml) * u
print(f"Radial ODE expression: {ODE.simplify()}")
print("✅ Validation complete: Successfully constructed the symbolic radial ODE expression with the effective potential V_ml.")
