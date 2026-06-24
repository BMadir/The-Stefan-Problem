import numpy as np
import scipy

class Stefan:
    
    def __init__(self, T_h, T_c, T_f, L_f, alpha, c, L_ref, T_ref):
        
        self.T_h = T_h
        self.T_c = T_c
        self.T_f = T_f
        self.L_f = L_f
        self.alpha = alpha
        self.c = c
        self.L_ref = L_ref
        self.T_ref = T_ref
        
        self.delta_t = max(T_h - T_f, T_f - T_c)
        self.theta_h = (T_h - T_f)/self.delta_t
        self.theta_c = (T_c - T_f)/self.delta_t
        
        self.Pe = L_ref**2/(T_ref* alpha) # which is 1/Fo, with Fo is the Fourier number.
        self.Ste = (self.delta_t* c)/L_f
        
        self.erf = lambda x : scipy.special.erf(x)
        self.erfc = lambda x : scipy.special.erfc(x)
        
        self.lam = scipy.optimize.newton(self.F_lam, 0.01)
        
    def F_lam(self, x):
        coef1 = self.c/(self.L_f* np.sqrt(np.pi))
        coef2 = self.T_c - self.T_f
        coef3 = self.T_h - self.T_f
        
        return x - coef1* np.exp(-x**2)* (coef2/self.erfc(x) + coef3/self.erf(x))
    
    def theta_l(self, x, t):
        N = self.erf(np.sqrt(self.Pe)* x/(2* np.sqrt(t)))
        D = self.erf(self.lam)
        
        return self.theta_h* (1 - N/D)
    
    def theta_s(self, x, t):
        N = self.erfc(np.sqrt(self.Pe)* x/(2* np.sqrt(t)))
        D = self.erfc(self.lam)
        
        return self.theta_c* (1 - N/D)
    
    def S(self, t): 
        return 2* self.lam* np.sqrt(self.alpha)* np.sqrt(t)
    
    def theta(self, x, t):
        Sol_liq = self.theta_l(x, t)
        Sol_sol = self.theta_s(x, t)
        Sol = np.where(x < self.S(t), Sol_liq, Sol_sol)
        return Sol