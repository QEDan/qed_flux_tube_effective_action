import numpy as np
import torch

class FieldProfile:
    def __init__(self, rho):
        if isinstance(rho, np.ndarray):
            self.rho = torch.from_numpy(rho).to(torch.float64)
        else:
            self.rho = rho
        self.a_phi = torch.zeros_like(self.rho)
        self.da_phi = torch.zeros_like(self.rho)

    def get_arrays(self, as_numpy=True):
        if as_numpy:
            return self.rho.detach().cpu().numpy(), \
                   self.a_phi.detach().cpu().numpy(), \
                   self.da_phi.detach().cpu().numpy()
        return self.rho, self.a_phi, self.da_phi

class StepFunctionProfile(FieldProfile):
    def __init__(self, rho, lambd, F, e=1.0):
        super().__init__(rho)
        self.lambd = lambd
        self.F = F
        self.e = e
        self.update()

    def update(self):
        # Aphi = F/(2*pi) * (rho/lambd^2 if rho < lambd else 1/rho)
        inner = self.rho < self.lambd
        outer = ~inner
        
        pre = self.F / (2 * np.pi)
        
        # Initialize tensors if not already
        self.a_phi = torch.zeros_like(self.rho)
        self.da_phi = torch.zeros_like(self.rho)
        
        # Fill a_phi
        self.a_phi = torch.where(inner, pre * self.rho / (self.lambd**2), pre / self.rho)
        
        # Fill da_phi
        self.da_phi = torch.where(inner, pre / (self.lambd**2), -pre / (self.rho**2))

class DifferentiableProfile(FieldProfile):
    """
    A profile where a_phi and da_phi are derived from differentiable parameters.
    """
    def __init__(self, rho, params):
        super().__init__(rho)
        self.params = params # e.g. spline coefficients or NN weights
        
    def forward(self):
        # This should be implemented by subclasses or provided via a function
        raise NotImplementedError
