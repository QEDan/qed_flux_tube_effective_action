import sympy
from sympy import symbols, Function, diff, pi

rho, m, ml, e, F, lambd, sigma3 = symbols('rho m ml e F lambd sigma3')
u = Function('u')(rho)

# Step function profile
Aphi_step = (F / (2*pi)) * (rho / lambd**2)
# B = A/rho + dA/drho
B_step = Aphi_step/rho + diff(Aphi_step, rho)
print(f"B-field (interior): {B_step.simplify()}")

# V_ml = e*sigma3*B + (ml^2-1)/rho^2 + e^2*A^2 - 2*e*ml*A/rho
V_ml = e * sigma3 * B_step + (ml**2 - 1)/rho**2 + e**2 * Aphi_step**2 - 2 * e * ml * Aphi_step / rho
print(f"V_ml (interior): {V_ml.simplify()}")

# ODE: u'' + 1/rho * u' - (V_ml + 1/rho^2 - chi^2 + m^2) u = 0
# Potential for u'': V_ml + 1/rho^2 - chi^2 + m^2
chi, m = symbols('chi m')
pot = V_ml + 1/rho**2 - chi**2 + m**2
print(f"Potential for u'' (interior): {pot.simplify()}")
