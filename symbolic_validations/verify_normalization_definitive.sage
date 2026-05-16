from sage.all import *

def definitive_normalization_test():
    print("--- Definitive Normalization Test (Small Field Limit) ---")
    
    # HE small field (Scalar QED): L = (eB)^4 / (360 * pi^2 * m^4)
    # Spectral G sum small field:
    # Delta tr G_2D(s) approx -(eB^2 / 24*pi) * s + (7*eB^4 / 1440*pi) * s^3
    # Wait, let's derive Delta tr G_2D(s) to more terms.
    
    s, eB, m, chi = var('s eB m chi')
    # Delta tr G_2D(s) = (1/4*pi) * [eB/sinh(seB) - 1/s]
    # Expansion of [eB/sinh(seB) - 1/s] around s=0:
    # 1/sinh(x) = 1/x - x/6 + 7x^3/360 - 31x^5/15120 ...
    # [eB/sinh(seB) - 1/s] = -s*eB^2/6 + 7*s^3*eB^4/360 - 31*s^5*eB^6/15120
    
    # Term of order B^4: (1/4*pi) * (7*s^3*eB^4/360) = (7*eB^4 / 1440*pi) * s^3
    
    # Delta tr G_2D(chi) = integral ds e^(-s(chi^2+m^2)) * Delta tr G_2D(s)
    # Order B^4: (7*eB^4 / 1440*pi) * integral ds s^3 e^(-s(chi^2+m^2))
    # = (7*eB^4 / 1440*pi) * (6 / (chi^2+m^2)^4)
    # = (7*eB^4 / 240*pi) * (1 / (chi^2+m^2)^4)
    
    # Spectral integral: L = Norm_L * integral chi^3 dchi * (2*pi * Delta tr G_2D)
    # L_B4 = Norm_L * 2*pi * (7*eB^4 / 240*pi) * integral chi^3 dchi / (chi^2+m^2)^4
    
    # integral chi^3 dchi / (chi^2+m^2)^4:
    # Let x = chi^2. dx = 2 chi dchi.
    # 1/2 * integral x dx / (x+m^2)^4
    # = 1/2 * integral ( (x+m^2) - m^2 ) / (x+m^2)^4 dx
    # = 1/2 * [ -1/(2(x+m^2)^2) + m^2/(3(x+m^2)^3) ]_0^inf
    # = 1/2 * [ 1/(2m^4) - m^2/(3m^6) ] = 1/2 * [ 1/2 - 1/3 ] / m^4 = 1/2 * (1/6) / m^4 = 1/(12 m^4)
    
    # L_B4 = Norm_L * 2*pi * (7*eB^4 / 240*pi) * (1 / 12*m^4)
    # = Norm_L * (7*eB^4 / 120) * (1 / 12*m^4) = Norm_L * 7*eB^4 / (1440 * m^4)
    
    # Matching to HE_B4 = (eB^4 / 360*pi^2 * m^4):
    # Norm_L * 7 / 1440 = 1 / (360*pi^2)
    # Norm_L = 1440 / (7 * 360 * pi^2) = 4 / (7 * pi^2)
    
    # Wait, the 7/360 in HE is for Scalar.
    # HE_B4_scal = (7 * (eB)^4) / (360 * 32*pi^2 * m^4) ?
    # Let's check I_scal expansion in WLNumerics.tex eq (eqn:scalI):
    # I_scal(T) approx 7(eB)^4 T / 360
    # L_HE_scal = integral dT/T^3 e^-m^2T (7(eB)^4 T / 360)
    # = (7(eB)^4 / 360) * integral dT/T^2 e^-m^2T ... Divergent.
    # Ah! The renormalization term 1/6 (eBT)^2 was subtracted.
    # But for B^4, it's fine.
    # Wait, the 1/T^3 makes it integral dT/T^2. Still divergent.
    # Ah! The expansion in paper is for the integrad I(T).
    # L = 1/16pi^2 * integral dT I(T).
    # The B^4 term in HE is (eB)^4 / (360 * 16*pi^2 * m^4) * 7?
    # Standard HE for scalar is (eB)^4 / (2880 pi^2 m^4)?
    
    print("\nCorrect HE coefficients from literature (Scalar):")
    print("L = (eB)^4 / (2880 * pi^2 * m^4)")
    # Matching:
    # Norm_L * 7 / 1440 = 1 / (2880 * pi^2)
    # Norm_L = 1440 / (7 * 2880 * pi^2) = 1 / (14 * pi^2)
    
    print("\nMatches Norm_L = 1 / (14 * pi^2)? Still weird.")

if __name__ == "__main__":
    definitive_normalization_test()
