"""
Audits the stability of the numerical Green's function solver by comparing it 
against the analytic benchmark across the parameter space (chi, ml, sigma3, F).
Identifies regimes where residuals are high or numerical errors (NaN, Inf) occur.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.python.orchestrator import Orchestrator
from src.python.profiles import StepFunctionProfile
from src.python.analytic_step_profile import get_full_analytic_solution

def audit_stability():
    print("--- Auditing Green's Function Stability ---")
    
    device = 'cpu'
    # Use a fixed lambda for simplicity in this audit
    lambd = 1.0
    m = 1.0
    rho = torch.linspace(0.01, 5.0, 100, device=device)
    rho_np = rho.numpy()
    
    orc = Orchestrator(device=device)
    
    # Ranges for parameters
    chi_values = [0.1, 1.0, 5.0, 10.0, 20.0]
    ml_values = [0, 5, 15, 30]
    sigma3_values = [1, -1]
    F_values = [0.1, 0.5, 1.0, 2.0]

    results = []

    print(f"{'Chi':<6} | {'ml':<4} | {'s3':<4} | {'F':<4} | {'Max Abs Err':<12} | {'Max Rel Err':<12} | {'Status'}")
    print("-" * 75)

    for chi in chi_values:
        # Using a small imaginary part as in the orchestrator
        chi_c = chi + 0.1j
        for ml in ml_values:
            for s3 in sigma3_values:
                for F in F_values:
                    p = StepFunctionProfile(rho, lambd=lambd, F=F)
                    
                    # 1. Numerical Solution
                    params = [{'chi': chi_c, 'ml': ml, 'sigma3': s3, 'm': m, 'e': 1.0}]
                    try:
                        g_num_batch, _ = orc.backend.solve_batch(params, p)
                        g_num = g_num_batch[0].detach().cpu().numpy()
                    except Exception as e:
                        print(f"{chi:<6} | {ml:<4} | {s3:<4} | {F:<4} | {'ERROR':<12} | {'ERROR':<12} | {str(e)[:20]}")
                        continue
                    
                    # 2. Analytic Solution
                    try:
                        g_ana = get_full_analytic_solution(rho_np, chi_c, ml, s3, m, lambd, F)
                    except Exception as e:
                        print(f"{chi:<6} | {ml:<4} | {s3:<4} | {F:<4} | {'ANA_ERR':<12} | {'ANA_ERR':<12} | {str(e)[:20]}")
                        continue

                    # 3. Residuals
                    abs_err = np.abs(g_num - g_ana)
                    # Use a small epsilon for relative error to avoid div by zero
                    rel_err = abs_err / (np.abs(g_ana) + 1e-15)
                    
                    max_abs = np.max(abs_err)
                    max_rel = np.max(rel_err)
                    
                    status = "✅ OK"
                    if np.isnan(max_abs) or np.isinf(max_abs):
                        status = "❌ NaN/Inf"
                    elif max_rel > 0.1:
                        status = "⚠️ HIGH ERR"
                    elif max_rel > 0.01:
                        status = "🟡 MED ERR"
                        
                    print(f"{chi:<6.1f} | {ml:<4} | {s3:<4} | {F:<4.1f} | {max_abs:<12.2e} | {max_rel:<12.2e} | {status}")
                    
                    results.append({
                        'chi': chi, 'ml': ml, 's3': s3, 'F': F,
                        'max_abs': max_abs, 'max_rel': max_rel, 'status': status
                    })

    # Summary of failures
    failures = [r for r in results if "❌" in r['status'] or "⚠️" in r['status']]
    if failures:
        print(f"\nAudit complete. Found {len(failures)} parameter sets with issues.")
        for f in failures:
            print(f"Problem set: chi={f['chi']}, ml={f['ml']}, s3={f['s3']}, F={f['F']} -> RelErr={f['max_rel']:.2e}")
    else:
        print("\nAudit complete. All parameter sets within expected bounds.")

if __name__ == "__main__":
    audit_stability()
