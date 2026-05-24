from src.python import constants
import sys, os; sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src/python"))
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.python.orchestrator import Orchestrator
from src.python.delta_shell import DeltaFunctionShellProfile
from scipy.special import iv, kv, jv, yv

def get_analytic_delta_shell_g(rho, R, F, chi, ml, sigma3, m=constants.ELECTRON_MASS, e=constants.ELECTRON_CHARGE):
    """
    Compute analytic Green's function G(rho, rho) for a delta shell.
    Based on matching I/K (or J/Y) functions at rho=R with jump condition.
    """
    F_cal = e * F / (2.0 * np.pi)
    k2 = chi**2 - m**2
    n = ml - F_cal
    
    # Solver uses ml^2 centrifugal term (standard Bessel)
    nu = ml
    n_nu = n
    
    # Physical jump condition: du_ext - du_int = + (e * sigma3 * F / 2pi*R) * u(R)
    jump = sigma3 * F_cal / R
    
    # Use I/K if k2 < 0, J/Y if k2 > 0
    if k2.real < 0:
        kappa = np.sqrt(-k2 + 0j)
        
        def d_iv(v, z): return 0.5 * (iv(v-1, z) + iv(v+1, z))
        def d_kv(v, z): return -0.5 * (kv(v-1, z) + kv(v+1, z))
        
        u0_R = iv(nu, kappa * R)
        du0_in_R = kappa * d_iv(nu, kappa * R)
        du0_out_R = du0_in_R + jump * u0_R
        
        uinf_R = kv(n_nu, kappa * R)
        duinf_out_R = kappa * d_kv(n_nu, kappa * R)
        
        W0 = R * (du0_out_R * uinf_R - u0_R * duinf_out_R)
        
        g = np.zeros_like(rho, dtype=complex)
        for i, r in enumerate(rho):
            if r <= R:
                uinf_in_R = uinf_R
                duinf_in_R = duinf_out_R - jump * uinf_R
                mat = np.array([[iv(nu, kappa*R), kv(nu, kappa*R)], 
                                [d_iv(nu, kappa*R), d_kv(nu, kappa*R)]])
                vec = np.array([uinf_R, duinf_in_R / kappa])
                D = np.linalg.solve(mat, vec)
                uinf_val = D[0] * iv(nu, kappa*r) + D[1] * kv(nu, kappa*r)
                u0_val = iv(nu, kappa*r)
            else:
                u0_out_R = u0_R
                du0_out_R_scaled = du0_out_R / kappa
                mat = np.array([[iv(n_nu, kappa*R), kv(n_nu, kappa*R)],
                                [d_iv(n_nu, kappa*R), d_kv(n_nu, kappa*R)]])
                vec = np.array([u0_out_R, du0_out_R_scaled])
                C = np.linalg.solve(mat, vec)
                u0_val = C[0] * iv(n_nu, kappa*r) + C[1] * kv(n_nu, kappa*r)
                uinf_val = kv(n_nu, kappa*r)
            # The Green's function satisfies (L - E)G = (1/r)delta.
            # For L = (1/r)d/dr(r d/dr) - V, G = - u0*uinf / W0.
            # The result here is r*G to match the numerical solver.
            g[i] = - (r * u0_val * uinf_val) / W0
            
    else: # k2 > 0
        k = np.sqrt(k2 + 0j)
        
        def d_jv(v, z): return 0.5 * (jv(v-1, z) - jv(v+1, z))
        def d_yv(v, z): return 0.5 * (yv(v-1, z) - yv(v+1, z))
        
        u0_R = jv(nu, k * R)
        du0_in_R = k * d_jv(nu, k * R)
        du0_out_R = du0_in_R + jump * u0_R
        
        uinf_R = yv(n_nu, k * R)
        duinf_out_R = k * d_yv(n_nu, k * R)
        
        W0 = R * (du0_out_R * uinf_R - u0_R * duinf_out_R)
        
        g = np.zeros_like(rho, dtype=complex)
        for i, r in enumerate(rho):
            if r <= R:
                uinf_in_R = uinf_R
                duinf_in_R = duinf_out_R - jump * uinf_R
                mat = np.array([[jv(nu, k*R), yv(nu, k*R)], 
                                [d_jv(nu, k*R), d_yv(nu, k*R)]])
                vec = np.array([uinf_R, duinf_in_R / k])
                D = np.linalg.solve(mat, vec)
                uinf_val = D[0] * jv(nu, k*r) + D[1] * yv(nu, k*r)
                u0_val = jv(nu, k*r)
            else:
                u0_out_R = u0_R
                du0_out_R_scaled = du0_out_R / k
                mat = np.array([[jv(n_nu, k*R), yv(n_nu, k*R)],
                                [d_jv(n_nu, k*R), d_yv(n_nu, k*R)]])
                vec = np.array([u0_out_R, du0_out_R_scaled])
                C = np.linalg.solve(mat, vec)
                u0_val = C[0] * jv(n_nu, k*r) + C[1] * yv(n_nu, k*r)
                uinf_val = yv(n_nu, k*r)
            # The Green's function satisfies (L - E)G = (1/r)delta.
            # For L = (1/r)d/dr(r d/dr) - V, G = - u0*uinf / W0.
            # The result here is r*G to match the numerical solver.
            g[i] = - (r * u0_val * uinf_val) / W0
            
    return g

def run_delta_shell_validation():
    print("--- Running Delta-Function Shell Validation ---")
    
    R = 5.0
    F = 1.0 
    rho_np = np.linspace(0.01, 10.0, 500)
    profile = DeltaFunctionShellProfile(rho_np, R=R, F=F)
    
    orchestrator = Orchestrator(device='cpu')
    
    # Test for a single set of parameters
    chi = 0.5
    ml = 1
    sigma3 = 1
    m = 1.0
    
    params = [{'chi': complex(chi), 'ml': ml, 'sigma3': sigma3, 'm': m, 'e': 1.0}]
    
    print(f"Solving numerically for chi={chi}, ml={ml}, sigma3={sigma3}...")
    num_g_batch, _ = orchestrator.backend.solve_batch(params, profile)
    num_g = num_g_batch[0].cpu().numpy()
    
    print("Computing analytic Green's function...")
    ana_g = get_analytic_delta_shell_g(rho_np, R, F, chi, ml, sigma3, m)
    
    # Visualization: 3 subplots (Real, Imag, Residuals)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # 1. Real part
    axes[0].plot(rho_np, num_g.real, label='Numerical (Real)', color='blue')
    axes[0].plot(rho_np, ana_g.real, label='Analytic (Real)', linestyle='--', color='red')
    axes[0].axvline(x=R, color='gray', linestyle=':', label='Shell Radius R')
    axes[0].set_ylabel(r"$\mathfrak{Re}\{G(\rho, \rho)\}$")
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Imaginary part
    axes[1].plot(rho_np, num_g.imag, label='Numerical (Imag)', color='blue')
    axes[1].plot(rho_np, ana_g.imag, label='Analytic (Imag)', linestyle='--', color='red')
    axes[1].axvline(x=R, color='gray', linestyle=':', label='Shell Radius R')
    axes[1].set_ylabel(r"$\mathfrak{Im}\{G(\rho, \rho)\}$")
    axes[1].legend()
    axes[1].grid(True)

    # 3. Residuals
    residuals = np.abs(num_g - ana_g)
    axes[2].plot(rho_np, residuals, label='Residual Magnitude', color='black')
    axes[2].axvline(x=R, color='gray', linestyle=':', label='Shell Radius R')
    axes[2].set_xlabel('rho')
    axes[2].set_ylabel('Residual')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    output_path = "results/delta_shell_greens_function_comparison.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Compute error
    error = np.linalg.norm(num_g - ana_g) / np.linalg.norm(ana_g)
    print(f"Relative error: {error:.2e}")
    
    if error < 1e-3:
        print("✅ Delta-shell Green's function validation passed!")
    else:
        print("❌ Delta-shell Green's function validation failed: high error.")

if __name__ == "__main__":
    run_delta_shell_validation()
