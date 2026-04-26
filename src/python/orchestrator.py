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

class Orchestrator:
    def __init__(self, backend_type="pytorch", device=None, lib_path="./libsolver.so"):
        self.backend_type = backend_type
        if backend_type == "pytorch":
            self.backend = PyTorchSolver(device=device)
            self.device = self.backend.device
        elif backend_type == "c":
            self.backend = CSolverBackend(lib_path=lib_path)
            self.device = torch.device("cpu") # Default for C backend
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    def compute_effective_action(self, field_profile, chi_values, ml_values, sigma3_values, m=1.0, e=1.0):
        """
        Computes the full effective action by integrating over chi and summing over ml.
        """
        params_grid = generate_params_grid(chi_values, ml_values, sigma3_values, m=m, e=e)
        
        if self.backend_type == "pytorch":
            # Direct tensor computation for gradients
            results = self.backend.solve_batch(params_grid, field_profile)
            
            # G0 (field-free) subtraction: results - G0
            # For simplicity in this step, we can compute G0 by passing a zero profile
            # or by implementing the analytic Bessel expression.
            # Let's use the analytic background Greens function from Eq 2.22
            rho, _, _ = field_profile.get_arrays(as_numpy=False)
            
            # Reshape results to (n_chi, n_ml, n_s3, n_points)
            n_chi = len(chi_values)
            n_ml = len(ml_values)
            n_s3 = len(sigma3_values)
            n_points = len(rho)
            res_tensor = results.view(n_chi, n_ml, n_s3, n_points)
            
            # Background G0 = -pi/2 * rho * J_ml(k*rho) * Y_ml(k*rho)
            # where k = sqrt(chi^2 - m^2) (Wait, chi is Wick rotated, so k is real?)
            # In Eq 2.21, chi was introduced after rotation.
            
            # For now, let's implement the integral sum:
            # action = sum_{s3, ml} \int chi^3 dchi \int rho^2 drho (G - G0)
            
            # Trapeze integration over rho
            rho_weights = torch.zeros_like(rho)
            rho_weights[1:-1] = (rho[2:] - rho[:-2]) / 2.0
            rho_weights[0] = (rho[1] - rho[0]) / 2.0
            rho_weights[-1] = (rho[-1] - rho[-2]) / 2.0
            
            # Inner integral over rho: \int rho^2 (G - G0) drho
            # Assuming G0 is handled or using a simpler subtraction for validation
            # Let's assume we want the raw integral for now to test auto-diff
            inner_int = torch.sum(res_tensor * (rho**2 * rho_weights), dim=-1)
            
            # Sum over ml and s3
            summed = torch.sum(inner_int, dim=(1, 2))
            
            # Integration over chi (using chi_values)
            chi_tensor = torch.tensor([complex(c) for c in chi_values], device=self.device, dtype=torch.complex128)
            chi_weights = torch.zeros_like(chi_tensor.real)
            # Simple trapezoid for chi
            chi_real = chi_tensor.real
            chi_weights[1:-1] = (chi_real[2:] - chi_real[:-2]) / 2.0
            chi_weights[0] = (chi_real[1] - chi_real[0]) / 2.0
            chi_weights[-1] = (chi_real[-1] - chi_real[-2]) / 2.0
            
            # Full action = \pi * \sum \int chi^3 * inner_int dchi
            action = torch.pi * torch.sum(chi_tensor**3 * summed * chi_weights)
            return action
        else:
            raise NotImplementedError("Effective action integration only implemented for PyTorch backend.")

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
