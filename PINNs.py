import torch
import numpy as np
from neural_networks import feedforward, ResNet, sf_net, ff_net
from AutoDiff import AG_grad, args_requires_grad

__all__ = ["Stefan_pinn"]

class Stefan_pinn(feedforward):
    def __init__(self, layers_dim, activations, Ste, Fo, delta, dim=1, mu=0., sigma=1., **kwargs):
        super().__init__(layers_dim, activations, **kwargs)
        self.Ste = Ste
        self.Fo = Fo
        self.delta = delta
        self.dim = dim
        self.mu = mu
        self.sigma = sigma

    def net(self, *args):
        inputs = torch.cat(args, dim=1)
        inputs_ = (inputs - self.mu)/self.sigma
        outputs = self.__call__(inputs_)
        return outputs

    def fn_net(self, params, *args):
        params_dict = _vect_parameters_to_dic(
            params,
            dict(self.named_parameters())
        )

        inputs = torch.cat(args, dim=1)
        outputs = functional_call(
            self,
            params_dict,
            inputs
        )
        return outputs

    @staticmethod
    def phi(T, delta):
        return (1 / 2) * (1 + torch.tanh(T / delta))

    @staticmethod
    def phi_der(T, delta):
        return (1 / (2 * delta)) * (1 - torch.tanh(T / delta) ** 2)

    @args_requires_grad
    def net_grad(self, *args):
        T = self.net(*args)
        grad = AG_grad(T, args)
        return grad

    @args_requires_grad
    def net_res(self, *args):
        if self.dim == 1:
            return self.net_res_1D(*args)
        elif self.dim == 2:
            return  self.net_res_2D(*args)
        else:
            raise NotImplementedError(f"Stefan pb for dim = {self.dim} is not implemented")

    def net_res_1D(self, t, x):
        T = self.net(t, x)
        Tt, Tx = AG_grad(T, (t, x))
        Txx = AG_grad(Tx, x)
        res = (1. + (1. / self.Ste) * self.phi_der(T, self.delta)) * Tt - (self.Fo) * Txx
        return res
    
    def net_res_2D(self, t, x, y):
        T = self.net(t, x, y)
        Tt = AG_grad(T, t)
        Txx, Tyy = AG_grad(T, (x, y), order=2)
        res = (1. + (1. / self.Ste) * self.phi_der(T, self.delta)) * Tt - (self.Fo) * (Txx + Tyy)
        return res