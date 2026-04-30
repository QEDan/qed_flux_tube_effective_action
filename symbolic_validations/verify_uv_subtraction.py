import sympy
from sympy import symbols, pi, exp, sqrt, sin, cos

rho, k, F, lambd, ml = symbols('rho k F lambd ml')

# Define Theta for the UV subtraction term (Eq 2.58/2.59)
# Theta = 2k*rho - (1/4 - ml^2)/(k*rho)
Theta = 2 * k * rho - (0.25 - ml**2) / (k * rho)

# Define UV subtraction term components (Eq 2.59)
# term = rho^3 / (2*k^2) * sin(Theta) + rho^2 / (6*k^3) * cos(Theta)
# Note: The document actually has sin(Theta)cos(Theta) and cos(2*Theta)
# I need to verify the exact expression in greensfunc.tex.
# Reading Eq 2.59 again: ... + [ rho^3 / 2k^2 * sin(Theta) + rho^2 / 6k^3 * cos(Theta) ]

term1 = (rho**3 / (2 * k**2)) * sin(Theta)
term2 = (rho**2 / (6 * k**3)) * cos(Theta)

print(f"UV Subtraction Term: {(term1 + term2).simplify()}")
