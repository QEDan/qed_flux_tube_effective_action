from src.python import constants
from src.python import constants
import torch
import numpy as np
from src.python.profiles import FieldProfile
from scipy.special import expi

class LatticeBumpProfile(FieldProfile):
    def __init__(self, rho: torch.Tensor, a: float, lambd: float, F: float, lambd_min: float = None, e: float = constants.ELECTRON_CHARGE) -> None:
        """
        Cylindrically symmetric flux tube lattice model from docs/periodic.tex.
        
        a: Lattice spacing (distance between flux tubes)
        lambd: Flux tube width parameter
        F: Flux parameter (dimensionless F = e/(2*pi) * Phi)
        lambd_min: Minimum width at which the field between tubes vanishes (default 0.1 * a)
        """
        super().__init__(rho)
        self.a = a
        self.lambd = lambd
        self.F = F
        self.lambd_min = lambd_min if lambd_min is not None else 0.1 * a
        self.e = e
        
        # Numerical constants from periodic.tex
        self.q1 = 0.443991
        self.q2 = 0.0742478
        self.q3 = 0.0187671
        
        self.update()

    def _psi(self, x: torch.Tensor) -> torch.Tensor:
        """Bump function Psi(x) = exp(-1/(1-x^2)) for |x|<1, else 0."""
        mask = torch.abs(x) < 1.0
        # Use a small eps to avoid division by zero in mask-out regions, 
        # though torch.where should handle it.
        val = torch.zeros_like(x)
        safe_x = torch.where(mask, x, torch.zeros_like(x))
        val[mask] = torch.exp(-1.0 / (1.0 - safe_x[mask]**2))
        return val

    def update(self) -> None:
        r_safe = torch.where(self.rho == 0, torch.tensor(1e-15, device=self.rho.device), self.rho)
        
        # Calculate A0 and B0 for rho <= a/2
        # F in the document is dimensionless flux parameter F = e/(2*pi) * Phi.
        # In our StepFunctionProfile, self.F is total flux Phi.
        # Let's clarify: The document says F = e/(2*pi) * Phi.
        # So we'll treat the input F as the total flux, and convert to cal_F = F * e / (2*pi).
        cal_F = self.F * self.e / (constants.TWO_PI)
        
        term_l = (self.lambd - self.lambd_min) / (self.a - self.lambd_min)
        
        # B0 (constant background in interior)
        B0 = (6.0 * cal_F) / (self.e * self.a**2) * term_l
        
        # A0 (amplitude of bump in interior)
        A0 = (4.0 * cal_F) / (self.lambd**2 * self.e * self.q2) * (1.0 - 0.75 * term_l)
        
        # B_z for rho <= a/2
        # We only implement the interior region for now as requested by Q5 
        # (comparison within central cell 0 < rho < a/2)
        inner_mask = self.rho <= (self.a / 2.0)
        
        b_field = torch.zeros_like(self.rho)
        b_field[inner_mask] = A0 * self._psi(2.0 * self.rho[inner_mask] / self.lambd) + B0
        
        # For rho > a/2, we follow Eq 10.16
        outer_mask = ~inner_mask
        if outer_mask.any():
            # n = floor((rho + a/2) / a)
            n = torch.floor((self.rho[outer_mask] + self.a / 2.0) / self.a)
            term_outer_bump = (12.0 * cal_F) / (self.q1 * self.e * self.a * self.lambd) * \
                              ((self.a - self.lambd) / (self.a - self.lambd_min))
            b_field[outer_mask] = B0 + term_outer_bump * self._psi(2.0 * (self.rho[outer_mask] - n * self.a) / self.lambd)

        # Vector potential A_phi = 1/rho * Integral_0^rho r' B(r') dr'
        # The integral must capture the total flux up to rho.
        if len(self.rho) > 1:
            dr = self.rho[1] - self.rho[0]
            integrand = b_field * self.rho
            # For a periodic lattice, the field B(r) is not purely localized.
            # We integrate cumulatively.
            flux_integral = torch.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dr, dim=0)
            flux_integral = torch.cat([torch.zeros(1, device=self.rho.device), flux_integral])
        else:
            flux_integral = 0.5 * (b_field * self.rho) * self.rho

        self.a_phi = flux_integral / r_safe

        # In a lattice, the background field B0 is non-zero. 
        # A_phi should really be A_phi = 1/rho * Integral (B - B_background)
        # to ensure the gauge A_phi -> 0 at infinity or is consistent with the vacuum solver.
        # Let's subtract the contribution of B0.
        # Contribution of B0 is B0 * rho^2 / 2.
        # A_phi_sub = A_phi - B0 * rho / 2
        self.a_phi -= B0 * self.rho / 2.0

        # Subtract B0 from B_vals to represent the field fluctuation B - B0.
        # This makes it consistent with the vacuum solver which assumes B=0 background.
        self.B_vals = b_field - B0
        
        # A_phi is already updated with B0 subtraction:
        # A_phi_sub = A_phi - B0 * rho / 2
        # Which is consistent with B_fluctuation.
        
        self.da_phi = self.B_vals - self.a_phi / r_safe


if __name__ == "__main__":
    # Quick sanity check plot
    import matplotlib.pyplot as plt
    rho = torch.linspace(0, 15, 500)
    # Fig 10.2: a = sqrt(8)*lambda_e, lambda_min = 0.1a. 
    # Use lambda_e = 1.0 (m=1)
    a_val = np.sqrt(8.0)
    profile = LatticeBumpProfile(rho, a=a_val, lambd=0.6*a_val, F=2.0*constants.PI)
    
    r, a_p, da_p = profile.get_arrays()
    B = profile.B_vals.detach().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(r, B)
    plt.axvline(a_val/2, color='k', linestyle='--', label='a/2')
    plt.axvline(a_val, color='r', linestyle=':', label='a')
    plt.title("B_z(rho)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(r, a_p, label='A_phi')
    plt.plot(r, da_p, label="A_phi'")
    plt.title("Vector Potential")
    plt.legend()
    plt.tight_layout()
    plt.savefig("debug/lattice_bump_check.png")
    print("Plot saved to debug/lattice_bump_check.png")
