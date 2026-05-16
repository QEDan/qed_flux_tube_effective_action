from sage.all import *

def check_rho_zero_limit():
    print("--- Checking rho -> 0 Limit of Green's Function ---")
    k, rho, l = var('k rho l')
    # Vacuum Green's function of order l
    # G_l = -pi/2 * rho * J_l(k*rho) * Y_l(k*rho)
    G = -pi/2 * rho * bessel_J(l, k*rho) * bessel_Y(l, k*rho)
    
    # Limit of G/rho as rho -> 0
    # J_l ~ (k*rho/2)^l / l!
    # Y_l ~ - (l-1)! / pi (k*rho/2)^-l for l > 0
    # Y_0 ~ 2/pi ln(k*rho/2)
    
    print("\nLimit for l=0:")
    lim0 = (G.subs(l=0) / rho).series(rho, 2)
    print(lim0)
    
    print("\nLimit for l=1:")
    lim1 = (G.subs(l=1) / rho).series(rho, 2)
    print(lim1)

if __name__ == "__main__":
    check_rho_zero_limit()
