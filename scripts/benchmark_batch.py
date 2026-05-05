import torch
import numpy as np
import time
import sys
import os

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "python")))

from orchestrator import Orchestrator
from profiles import ZeroFluxProfile

def benchmark_batch_sizes():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on: {device}")
    
    # Test parameters
    lambd = 5.0
    B = 0.5
    rho = torch.linspace(0.01, 20.0, 500, device=device, dtype=torch.float64)
    profile = ZeroFluxProfile(rho, B=B, lambd=lambd)
    
    # Grid: small enough to run quickly, large enough to saturate GPU
    chi_values = [2.0 + 0.1j]
    ml_values = list(range(-50, 51))
    sigma3_values = [1, -1]
    
    # Batch sizes to test
    batch_sizes = [128, 512, 1024, 2048, 4096, 8192]
    
    results = {}
    
    for bs in batch_sizes:
        print(f"Testing batch_size={bs}...", end=" ", flush=True)
        try:
            orc = Orchestrator(device=device, batch_size=bs)
            
            # Warm-up run
            _ = orc.compute_effective_action(profile, chi_values, ml_values, sigma3_values, m=1.0)
            
            # Timed run
            start = time.time()
            _ = orc.compute_effective_action(profile, chi_values, ml_values, sigma3_values, m=1.0)
            end = time.time()
            
            duration = end - start
            results[bs] = duration
            print(f"took {duration:.4f}s")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("OOM!")
                break
            else:
                print(f"Error: {e}")
                break
                
    print("\n--- Benchmark Results ---")
    for bs, dur in results.items():
        print(f"Batch Size: {bs:5d} | Time: {dur:.4f}s")

if __name__ == "__main__":
    benchmark_batch_sizes()
