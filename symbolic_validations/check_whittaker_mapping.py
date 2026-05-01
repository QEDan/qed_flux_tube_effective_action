import sympy
from sympy import symbols, Function, diff, simplify, collect, Wild

def verify_mapping():
    rho, F_cal, lambd, ml, sigma3, chi, m = symbols('rho F_cal lambd ml sigma3 chi m', real=True)
    beta = F_cal / lambd**2
    k2 = chi**2 - m**2
    
    # 1. Define our Radial ODE: u'' + 1/rho u' - v_eff u = 0
    # Centrifugal term is ml**2 / rho**2
    v_eff = beta**2 * rho**2 + 2*beta*(sigma3 - ml) + ml**2/rho**2 - k2
    
    W = Function('W')
    z = symbols('z')
    
    def test_sub(a_val):
        print(f"\n--- Testing substitution u = W(z) / rho^{a_val} ---")
        # u(rho) = W(beta*rho^2) / rho^a
        # Use r as the radial variable to avoid confusion with rho symbol
        r = symbols('r')
        u_sub = W(beta * r**2) / r**a_val
        
        # ODE operator
        L_u = diff(u_sub, r, 2) + (1/r)*diff(u_sub, r) - v_eff.subs(rho, r) * u_sub
        
        # Substitute z = beta*r^2 back
        # dW/dr = W'(z) * 2*beta*r
        # d^2W/dr^2 = W''(z) * 4*beta^2*r^2 + W'(z) * 2*beta
        
        Wp = Function('Wp')(z)
        Wpp = Function('Wpp')(z)
        
        # Perform derivatives manually to match z
        # u' = [ W'(z)*2*beta*r * r^a - W(z)*a*r^(a-1) ] / r^(2a)
        # ... or just use sympy and then substitute
        
        expr = simplify(L_u)
        
        # Replace W derivatives with symbolic ones
        W_r = W(beta*r**2)
        W_r_p = diff(W_r, r)
        W_r_pp = diff(W_r, r, 2)
        
        # We know:
        # W_r_p = W'(z) * 2*beta*r
        # W_r_pp = W''(z) * 4*beta^2*r^2 + W'(z) * 2*beta
        
        # Let's solve for W'' and W'
        L_final = expr.subs(diff(W(beta*r**2), r, 2), Wpp * 4*beta**2*r**2 + Wp * 2*beta)
        L_final = L_final.subs(diff(W(beta*r**2), r), Wp * 2*beta*r)
        L_final = L_final.subs(W(beta*r**2), W(z))
        
        # Substitute r^2 = z/beta
        L_final = simplify(L_final.subs(r**2, z/beta))
        
        print("Substituted ODE in terms of W(z):")
        print(L_final)
        
        # Standard Whittaker: W'' + (-1/4 + kappa/z + (1/4 - mu^2)/z^2) W = 0
        # Coefficient of W'' should be 1
        coeff_Wpp = L_final.coeff(Wpp)
        ode_norm = simplify(L_final / coeff_Wpp)
        
        print("\nNormalized ODE (W'' + ... = 0):")
        print(ode_norm)
        
        # Now extract kappa and mu
        # Term with 1/z is kappa
        # Term with 1/z^2 is (1/4 - mu^2)
        
        kappa_found = ode_norm.coeff(W(z)).coeff(z, -1)
        mu2_found = 0.25 - ode_norm.coeff(W(z)).coeff(z, -2)
        
        print(f"\nExtracted parameters for a={a_val}:")
        print(f"kappa = {simplify(kappa_found)}")
        print(f"mu^2  = {simplify(mu2_found)}")
        print(f"mu     = {simplify(sqrt(mu2_found))}")

    test_sub(1)
    test_sub(0)

if __name__ == "__main__":
    verify_mapping()
