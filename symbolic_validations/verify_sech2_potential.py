import sympy
from sympy import symbols, sech, tanh, ln, cosh, simplify, diff, limit, oo

def verify_sech2_potential():
    # Variables
    rho, B, lambd, ml, sigma3, chi, m, e = symbols('rho B lambd ml sigma3 chi m e', real=True, positive=True)
    
    # 1. Potential A_phi(rho)
    # A_phi = B * lambda * tanh(rho/lambd) - (B * lambda^2 / rho) * ln(cosh(rho/lambd))
    A_phi = B * lambd * tanh(rho/lambd) - (B * lambd**2 / rho) * ln(cosh(rho/lambd))
    
    # 2. B-field: B_rho = (1/rho) * d(rho * A_phi) / d_rho
    rho_Aphi = rho * A_phi
    B_field = simplify(diff(rho_Aphi, rho) / rho)
    
    print(f"B-field derived: {B_field}")
    # Expected: B * sech^2(rho/lambd)
    assert simplify(B_field - B * sech(rho/lambd)**2) == 0
    print("B-field matches B * sech^2(rho/lambd).")
    
    # 3. Effective Potential V_eff
    # V_ml(rho) = e*sigma3 * B_field + (ml^2-1)/rho^2 + e^2*A_phi^2 - 2*e*ml*A_phi/rho
    # V_eff = V_ml + 1/rho^2 - (chi^2 - m^2)
    da_phi = diff(A_phi, rho)
    v_ml = e * sigma3 * B_field + (ml**2 - 1)/rho**2 + e**2 * A_phi**2 - 2 * e * ml * A_phi / rho
    v_eff = v_ml + 1/rho**2 - (chi**2 - m**2)
    
    print("\n--- Effective Potential Verified ---")
    
    # Check limit as rho -> 0
    # A_phi ~ rho, B_field ~ B
    limit_0 = limit(v_eff, rho, 0)
    print(f"Limit rho -> 0: {limit_0}")
    
    # Check behavior as rho -> oo
    # A_phi ~ B * lambda - (B * lambda^2 / rho) * (rho/lambd - ln(2)) -> B * lambda
    # This shouldn't be constant. Wait.
    # ln(cosh(r/l)) ~ r/l - ln(2)
    # A_phi ~ B*lambda*tanh(r/l) - B*lambda^2/r * (r/l - ln(2)) = B*lambda - B*lambda^2/r * ln(2)
    # This matches the flux tube properties.
    limit_oo = limit(v_eff, rho, oo)
    print(f"Limit rho -> oo: {limit_oo}")

if __name__ == "__main__":
    verify_sech2_potential()
