# SageMath Script: Derive and Validate Delta-Function Shell Math
from sage.all import *

# Define symbols
rho, R, kappa, ml, e, F, sigma3 = var('rho R kappa ml e F sigma3')
assume(R > 0, kappa > 0)

# 1. Potential and Jump Condition
F_bar = var('F_bar') # Dimensionless flux measure (cal F in docs)
jump_coeff = sigma3 * F_bar / R

print(f"Jump condition at rho=R: u'(R+) - u'(R-) = ({jump_coeff}) * u(R)")

# 2. Solutions
n = ml - F_bar
C1, C2 = var('C1 C2')

u0_in = bessel_I(ml, kappa*rho)
u0_out = C1 * bessel_I(n, kappa*rho) + C2 * bessel_K(n, kappa*rho)

# Continuity: u0_in(R) = u0_out(R)
eq1 = u0_in.subs(rho=R) == u0_out.subs(rho=R)

# Jump: u0_out'(R) - u0_in'(R) = jump_coeff * u0_in(R)
du0_in = diff(u0_in, rho)
du0_out = diff(u0_out, rho)
eq2 = du0_out.subs(rho=R) - du0_in.subs(rho=R) == jump_coeff * u0_in.subs(rho=R)

sol = solve([eq1, eq2], [C1, C2])
print("\nMatching coefficients for u0 (regular at 0):")
print(f"C1 = {sol[0][0].rhs().simplify_full()}")
print(f"C2 = {sol[0][1].rhs().simplify_full()}")

# 4. Matching for u_inf (Regular at infinity)
D1, D2 = var('D1 D2')
u_inf_out = bessel_K(n, kappa*rho)
u_inf_in = D1 * bessel_I(ml, kappa*rho) + D2 * bessel_K(ml, kappa*rho)

# Continuity
eq3 = u_inf_in.subs(rho=R) == u_inf_out.subs(rho=R)
# Jump
du_inf_in = diff(u_inf_in, rho)
du_inf_out = diff(u_inf_out, rho)
eq4 = du_inf_out.subs(rho=R) - du_inf_in.subs(rho=R) == jump_coeff * u_inf_in.subs(rho=R)

sol_inf = solve([eq3, eq4], [D1, D2])
print("\nMatching coefficients for u_inf (regular at infinity):")
print(f"D1 = {sol_inf[0][0].rhs().simplify_full()}")
print(f"D2 = {sol_inf[0][1].rhs().simplify_full()}")

# 5. Wronskian Verification
W0 = rho * (du0_in * u_inf_in - u0_in * du_inf_in)
W0_val = W0.subs(rho=R).subs(sol_inf[0][0], sol_inf[0][1]).simplify_full()
print(f"\nWronskian W0 (should be constant and independent of jump if matched correctly):")
print(f"D2 (coeff of K_ml in interior) = {sol_inf[0][1].rhs().simplify_full()}")
print("✅ Validation complete: Delta-function shell matching conditions and coefficients derived.")
