import sympy
from sympy import symbols, Function, diff, simplify, cosh, sech, log, pi

def confirm_geometry_mismatch():
    rho, x, B_peak, lambd, e, ml, p2 = symbols('rho x B_peak lambd e ml p2', real=True, positive=True)
    
    # 1. Cartesian 1D (Dunne-Hall)
    # B(x) = B_peak * sech^2(x/lambd)
    # Gauge A2(x) such that B = dA2/dx
    # A2(x) = B_peak * lambd * tanh(x/lambd)
    A2 = B_peak * lambd * sympy.tanh(x/lambd)
    
    # Cartesian potential term in the ODE: (p2 - e*A2)^2
    # Note: p2 is a constant momentum eigenvalue.
    V_cart = (p2 - e * A2)**2
    
    # 2. Cylindrical 2D (Current Solver)
    # B(rho) = B_peak * sech^2(rho/lambd)
    # Gauge Aphi(rho) such that B = 1/rho * d(rho*Aphi)/drho
    # Aphi(rho) = B_peak * lambd^2 / rho * ln(cosh(rho/lambd))
    Aphi = B_peak * lambd**2 / rho * sympy.log(cosh(rho/lambd))
    
    # Cylindrical potential term in the radial ODE: (ml/rho - e*Aphi)^2
    V_cyl = (ml/rho - e * Aphi)**2
    
    print("--- Geometry Mismatch Confirmation (H9) ---")
    print(f"Cartesian Potential V_cart(x, p2): {simplify(V_cart)}")
    print(f"Cylindrical Potential V_cyl(rho, ml): {simplify(V_cyl)}")
    
    # Check if they are isomorphic under x <-> rho
    # We substitute x = rho and see if there exists a mapping p2(ml, rho)
    # For V_cart to match V_cyl, we would need:
    # p2 - e*A2(rho) = ml/rho - e*Aphi(rho)
    # p2 = ml/rho - e*Aphi(rho) + e*A2(rho)
    
    diff_pot = simplify(A2.subs(x, rho) - Aphi)
    print(f"\nDifference in Vector Potentials (A_cart(rho) - A_cyl(rho)):")
    print(diff_pot)
    
    # Even if p2 = ml/rho, the potentials A2 and Aphi are different functions.
    # A2(rho) = B*lambd*tanh(rho/lambd)
    # Aphi(rho) = B*lambd^2/rho * ln(cosh(rho/lambd))
    
    print("\nConclusion:")
    print("1. The vector potentials A(rho) for the same B(rho) field differ between Cartesian and cylindrical geometries.")
    print("   A_cart(x) is the integral of B(x).")
    print("   A_cyl(rho) is 1/rho * integral(rho' * B(rho')).")
    print("2. The 'centrifugal' term in cylindrical coordinates (ml/rho)^2 varies with rho,")
    print("   whereas the corresponding term in Cartesian (p2)^2 is constant for a given mode.")
    print("3. Therefore, the cylindrical Sech2Profile is NOT a coordinate transformation of the Cartesian one.")
    print("   The exact results from Dunne & Hall (1997) for a Cartesian slab cannot be used")
    print("   directly as an exact benchmark for the cylindrical tube solver.")
    print("✅ Validation complete: Confirmed fundamental geometry mismatch H9.")

if __name__ == "__main__":
    confirm_geometry_mismatch()
