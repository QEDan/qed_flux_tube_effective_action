import torch
import numpy as np
from src.python.orchestrator import Orchestrator
from src.python.profiles import Sech2Profile

def debug_solver():
    rho = torch.linspace(0.01, 10.0, 100, dtype=torch.float64)
    profile = Sech2Profile(rho, B=0.5, lambd=2.0)
    
    orc = Orchestrator(device="cpu")
    
    chi_vals = [1.1, 5.0, 10.0]
    ml_vals = [0, 50, 100, 200]
    s3_vals = [1, -1]
    
    for chi in chi_vals:
        for ml in ml_vals:
            for s3 in s3_vals:
                params = [{'chi': chi, 'ml': ml, 'sigma3': s3, 'm': 1.0, 'e': 1.0}]
                results, _ = orc.backend.solve_batch(params, profile)
                
                num_chi = torch.tensor([chi], device=orc.device, dtype=torch.complex128)
                num_ml = torch.tensor([ml], device=orc.device, dtype=torch.int32)
                
                g0 = orc.renormalizer.compute_g0(num_chi, num_ml, 1.0, rho)
                uv = orc.renormalizer.compute_uv_subtraction(num_chi, num_ml, 1.0, rho, profile)
                
                ren = results[0] - g0[0] + uv[0]
                
                if torch.isnan(ren).any():
                    print(f"❌ NaN found for chi={chi}, ml={ml}, s3={s3}")
                    if torch.isnan(results).any(): print("  Cause: results")
                    if torch.isnan(g0).any(): print("  Cause: g0")
                    if torch.isnan(uv).any(): print("  Cause: uv")
                else:
                    sum_rho = torch.sum(ren * (rho[1]-rho[0])).item()
                    print(f"✅ chi={chi:4.1f}, ml={ml:4d}, s3={s3:2d} OK. Sum={sum_rho:.2e}")

if __name__ == "__main__":
    debug_solver()
