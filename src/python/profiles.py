import numpy as np

class FieldProfile:
    def __init__(self, rho):
        self.rho = rho
        self.a_phi = np.zeros_like(rho)
        self.da_phi = np.zeros_like(rho)

    def get_arrays(self):
        return self.rho, self.a_phi, self.da_phi

class StepFunctionProfile(FieldProfile):
    def __init__(self, rho, lambd, F, e=1.0):
        super().__init__(rho)
        self.lambd = lambd
        self.F = F
        self.e = e
        
        # Aphi = F/(2*pi) * (rho/lambd^2 if rho < lambd else 1/rho)
        inner = self.rho < lambd
        outer = ~inner
        
        pre = F / (2 * np.pi)
        self.a_phi[inner] = pre * self.rho[inner] / (lambd**2)
        self.a_phi[outer] = pre / self.rho[outer]
        
        # da_phi = F/(2*pi) * (1/lambd^2 if rho < lambd else -1/rho^2)
        self.da_phi[inner] = pre / (lambd**2)
        self.da_phi[outer] = -pre / (self.rho[outer]**2)
