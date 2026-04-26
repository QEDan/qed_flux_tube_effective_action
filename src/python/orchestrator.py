import ctypes
import numpy as np
import os

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

class Orchestrator:
    def __init__(self, lib_path="./libsolver.so"):
        self.lib = ctypes.CDLL(os.path.abspath(lib_path))
        
        # solve_batch(Parameters* params_array, int n_params, Profile profile, double complex* results_array)
        self.lib.solve_batch.argtypes = [
            ctypes.POINTER(Parameters),
            ctypes.c_int,
            Profile,
            ctypes.POINTER(Complex128)
        ]
        self.lib.solve_batch.restype = None

    def compute_greens_function_batch(self, params_list, field_profile):
        n_params = len(params_list)
        rho, a_phi, da_phi = field_profile.get_arrays()
        n_points = len(rho)
        
        # Prepare parameters array
        params_array = (Parameters * n_params)()
        for i, p in enumerate(params_list):
            params_array[i] = Parameters(
                chi=Complex128.from_complex(complex(p['chi'])),
                ml=int(p['ml']),
                sigma3=int(p['sigma3']),
                m=float(p['m']),
                e=float(p['e'])
            )
            
        # Prepare profile
        c_profile = Profile(
            rho=rho.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            a_phi=a_phi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            da_phi=da_phi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_points=n_points
        )
        
        # Prepare results array
        results_array = (Complex128 * (n_params * n_points))()
        
        # Call C solver
        self.lib.solve_batch(params_array, n_params, c_profile, results_array)
        
        # Convert results to numpy complex128
        res_np = np.frombuffer(results_array, dtype=np.complex128).reshape(n_params, n_points)
        return res_np

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
