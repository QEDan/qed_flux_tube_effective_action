from sage.all import *

def verify_small_t_expansion():
    print("--- Verifying Small T Expansions ---")
    T, m, eB = var('T m eB')
    
    # Fermionic Wilson Loop (Constant Field)
    # W_ferm = eBT * coth(eBT)
    W_ferm = (eB*T) * coth(eB*T)
    
    # Integrand: (e^-m^2T / T^3) * [W_ferm - 1 - 1/3 (eBT)^2]
    I_ferm = exp(-m**2 * T) / T**3 * (W_ferm - 1 - (eB*T)**2 / 3)
    
    # Series expansion around T=0
    exp_ferm = I_ferm.series(T, 6)
    print("\nFermionic Integrand Expansion (T=0):")
    print(exp_ferm)
    
    # Scalar Wilson Loop (Constant Field)
    # W_scal = eBT / sinh(eBT)
    W_scal = (eB*T) / sinh(eB*T)
    
    # Integrand: (e^-m^2T / T^3) * [W_scal - 1 + 1/6 (eBT)^2]
    I_scal = exp(-m**2 * T) / T**3 * (W_scal - 1 + (eB*T)**2 / 6)
    
    # Series expansion around T=0
    exp_scal = I_scal.series(T, 6)
    print("\nScalar Integrand Expansion (T=0):")
    print(exp_scal)

if __name__ == "__main__":
    verify_small_t_expansion()
