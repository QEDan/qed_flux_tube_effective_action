import sympy as sp
R, F_cal, ml, chi, m, sigma3 = sp.symbols('R F_cal ml chi m sigma3')
M_R, dM_R, J_R, Y_R, dJ_R, dY_R = sp.symbols('M_R dM_R J_R Y_R dJ_R dY_R')
A = sp.symbols('A')
jump_coeff = sigma3 * F_cal / R
C, D = sp.symbols('C D')

eq1 = sp.Eq(C*J_R + D*Y_R, M_R * A / R)
eq2 = sp.Eq(C*dJ_R + D*dY_R, (dM_R - jump_coeff * M_R / R) * A)
sol = sp.solve([eq1, eq2], [C, D])
print("C:", sp.simplify(sol[C]))
print("D:", sp.simplify(sol[D]))
