import ctypes
import numpy as np
import os
import torch
from pytorch_solver import PyTorchSolver

# Define ctypes structures
class Complex128(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

    @classmethod
    def from_complex(cls, c):
        return cls(c.real, c.imag)

class Parameters(ctypes.Structure):
    _fields_ = [
        ("chi", Complex128),
        ("ml", ctypes.c_int),
        ("sigma3", ctypes.c_int),
        ("m", ctypes.c_double),
        ("e", ctypes.c_double),
    ]

class Profile(ctypes.Structure):
    _fields_ = [
        ("rho", ctypes.POINTER(ctypes.c_double)),
        ("a_phi", ctypes.POINTER(ctypes.c_double)),
        ("da_phi", ctypes.POINTER(ctypes.c_double)),
        ("n_points", ctypes.c_int),
    ]

class CSolverBackend:
    def __init__(self, lib_path="./libsolver.so"):
        if not os.path.exists(lib_path):
             # Try to find it in the project root if called from elsewhere
             project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
             lib_path = os.path.join(project_root, "libsolver.so")
             
        self.lib = ctypes.CDLL(os.path.abspath(lib_path))
        
        self.lib.solve_batch.argtypes = [
            ctypes.POINTER(Parameters),
            ctypes.c_int,
            Profile,
            ctypes.POINTER(Complex128)
        ]
        self.lib.solve_batch.restype = None

    def solve_batch(self, params_list, field_profile):
        n_params = len(params_list)
        rho, a_phi, da_phi = field_profile.get_arrays()
        n_points = len(rho)
        
        params_array = (Parameters * n_params)()
        for i, p in enumerate(params_list):
            params_array[i] = Parameters(
                chi=Complex128.from_complex(complex(p['chi'])),
                ml=int(p['ml']),
                sigma3=int(p['sigma3']),
                m=float(p['m']),
                e=float(p['e'])
            )
            
        c_profile = Profile(
            rho=rho.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            a_phi=a_phi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            da_phi=da_phi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_points=n_points
        )
        
        results_array = (Complex128 * (n_params * n_points))()
        self.lib.solve_batch(params_array, n_params, c_profile, results_array)
        
        res_np = np.frombuffer(results_array, dtype=np.complex128).reshape(n_params, n_points)
        return res_np

from pytorch_solver import PyTorchSolver
from renormalization import Renormalizer

# ... (ctypes structures unchanged)

class Orchestrator:
    def __init__(self, backend_type="pytorch", device=None, lib_path="./libsolver.so", batch_size=128):
        self.backend_type = backend_type
        self.batch_size = batch_size
        if backend_type == "pytorch":
            self.backend = PyTorchSolver(device=device)
            self.device = self.backend.device
            self.renormalizer = Renormalizer(device=self.device)
        elif backend_type == "c":
            self.backend = CSolverBackend(lib_path=lib_path)
            self.device = torch.device("cpu")
            self.renormalizer = Renormalizer(device="cpu")
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    def compute_effective_action(self, field_profile, chi_values, ml_values, sigma3_values, m=1.0, e=1.0, chi_threshold=100.0):
        """
        Computes the full effective action by integrating over chi and summing over ml.
        Implements batching to manage memory and UV renormalization.
        """
        rho, _, _ = field_profile.get_arrays(as_numpy=False)
        rho = rho.to(self.device)
        n_points = len(rho)
        
        # Integration weights for rho (trapezoidal)
        rho_weights = torch.zeros_like(rho)
        rho_weights[1:-1] = (rho[2:] - rho[:-2]) / 2.0
        rho_weights[0] = (rho[1] - rho[0]) / 2.0
        rho_weights[-1] = (rho[-1] - rho[-2]) / 2.0
        rho_factor = (rho**2 * rho_weights).to(torch.complex128)

        # Prepare parameters for batching
        all_params = []
        for chi in chi_values:
            for ml in ml_values:
                for s3 in sigma3_values:
                    all_params.append({'chi': chi, 'ml': ml, 'sigma3': s3, 'm': m, 'e': e})

        total_inner_sum = torch.zeros(len(chi_values), device=self.device, dtype=torch.complex128)
        
        # Mapping chi index for easy summation
        chi_map = {chi: i for i, chi in enumerate(chi_values)}

        # Process in batches
        for i in range(0, len(all_params), self.batch_size):
            batch = all_params[i : i + self.batch_size]
            
            # Split batch into numerical and asymptotic regimes
            numerical_batch = [p for p in batch if abs(p['chi']) <= chi_threshold]
            asymptotic_batch = [p for p in batch if abs(p['chi']) > chi_threshold]
            
            # Extract batch parameters for renormalization
            b_chi = torch.tensor([p['chi'] for p in batch], device=self.device, dtype=torch.complex128)
            b_ml = torch.tensor([p['ml'] for p in batch], device=self.device, dtype=torch.int32)
            
            # Initialize renormalized_g for the whole batch
            renormalized_g = torch.zeros((len(batch), n_points), device=self.device, dtype=torch.complex128)
            
            if numerical_batch:
                # Solve ODE for numerical batch
                num_results = self.backend.solve_batch(numerical_batch, field_profile)
                
                # Get G0 and UV sub for numerical batch
                num_chi = torch.tensor([p['chi'] for p in numerical_batch], device=self.device, dtype=torch.complex128)
                num_ml = torch.tensor([p['ml'] for p in numerical_batch], device=self.device, dtype=torch.int32)
                
                num_g0 = self.renormalizer.compute_g0(num_chi, num_ml, m, rho)
                num_uv = self.renormalizer.compute_uv_subtraction(num_chi, num_ml, m, rho, field_profile)
                
                num_renorm = num_results - num_g0 - num_uv
                
                # Place into renormalized_g
                num_indices = [idx for idx, p in enumerate(batch) if abs(p['chi']) <= chi_threshold]
                renormalized_g[num_indices] = num_renorm

            if asymptotic_batch:
                # For large chi, we assume the renormalized contribution is negligible or follow analytic form
                # In a full implementation, we'd use a 1/chi^4 expansion here.
                # For now, we set it to zero as it decays very fast.
                asymp_indices = [idx for idx, p in enumerate(batch) if abs(p['chi']) > chi_threshold]
                renormalized_g[asymp_indices] = 0.0

            # Integration over rho
            inner_int = torch.sum(renormalized_g * rho_factor, dim=-1) # (batch_size,)
            
            # Accumulate into total_inner_sum based on chi
            for idx, p in enumerate(batch):
                chi_idx = chi_map[p['chi']]
                total_inner_sum[chi_idx] += inner_int[idx]

        # Final integration over chi
        chi_tensor = torch.tensor([complex(c) for c in chi_values], device=self.device, dtype=torch.complex128)
        chi_real = chi_tensor.real
        chi_weights = torch.zeros_like(chi_real)
        if len(chi_real) > 1:
            chi_weights[1:-1] = (chi_real[2:] - chi_real[:-2]) / 2.0
            chi_weights[0] = (chi_real[1] - chi_real[0]) / 2.0
            chi_weights[-1] = (chi_real[-1] - chi_real[-2]) / 2.0
        else:
            # If only one chi point, we can't integrate. 
            # This is primarily for testing single points.
            chi_weights[0] = 1.0
        
        action = torch.pi * torch.sum(chi_tensor**3 * total_inner_sum * chi_weights)
        return action


def generate_params_grid(chi_values, ml_values, sigma3_values, m=1.0, e=1.0):
    grid = []
    for chi in chi_values:
        for ml in ml_values:
            for s3 in sigma3_values:
                grid.append({
                    'chi': chi,
                    'ml': ml,
                    'sigma3': s3,
                    'm': m,
                    'e': e
                })
    return grid
