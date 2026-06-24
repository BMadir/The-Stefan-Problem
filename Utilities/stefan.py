import numpy as np
import scipy
from scipy.special import erf, erfc
from scipy.optimize import newton

__all__ = ["Stefan1D"]

class Stefan1D:
    """ Implements the solution to the one-dimensional melting Stefan problem.

    Args:   are dimensional parameters
        Th: the hot temperature
        Tc: the cold temperature
        Tf: the fusion temperature
        L: the latent heat
        alpha: the material thermal diffusivity
        c: the specific heat
        xref: the reference length
        tref: the reference time
    """
    def __init__(self, Th, Tc, Tf, L, alpha, c, xref, tref):

        # convert to non-dimensional parameters
        dT = max(Th - Tf, Tf - Tc)
        self.Th = (Th - Tf) / dT
        self.Tc = (Tc - Tf) / dT
        self.Ste = (c* dT) / L
        self.Fo = (alpha * tref) / xref**2 # or maybe 1/Pe

        # solution to the nonlinear Stefan condition
        self.lam = newton(self._F_lam, 0.01)

    def _F_lam(self, x):
        F = x - (self.Ste / np.sqrt(np.pi)) * np.exp(- x ** 2)* (self.Tc / erfc(x) + self.Th / erf(x))
        return F

    def _T_liquid(self, t, x):
        N = erf(x / (2* np.sqrt(self.Fo * t)))
        D = erf(self.lam)
        return self.Th* (1 - N/D)

    def _T_solid(self, t, x):
        N = erfc(x / (2 * np.sqrt(self.Fo * t)))
        D = erfc(self.lam)
        return self.Tc* (1 - N/D)
    
    def S(self, t): 
        return 2* self.lam* np.sqrt(self.Fo * t)
    
    def T(self, t, x) :
        return np.where(x < self.S(t), self._T_liquid(t, x), self._T_solid(t, x))