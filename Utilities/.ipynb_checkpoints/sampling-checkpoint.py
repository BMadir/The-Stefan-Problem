import numpy as np
import torch
import skopt


class Sampler:

    def __init__(self, sampler, l_bounds, u_bounds, func, seed=1234):

        assert len(l_bounds) == len(u_bounds)
        self.dim = len(l_bounds)
        self.bounds = l_bounds, u_bounds

        self.sampler = self.__sampler_fn(sampler.lower())

        self.space = [(float(l), float(u)) for l, u in zip(l_bounds, u_bounds)]
        self.unit_sq = [(0., 1.) for l, u in zip(l_bounds, u_bounds)]

        self.func = func
        self.seed = seed

    @staticmethod
    def __sampler_fn(sampler):

        if sampler == 'lhs':
            sampler_fn = skopt.sampler.Lhs(lhs_type='centered', criterion='maximin', iterations=0)

        elif sampler == 'sobol':
            sampler_fn = skopt.sampler.Sobol(skip=0, randomize=False)

        elif sampler == 'halton':
            sampler_fn = skopt.sampler.Halton(min_skip=-1, max_skip=-1)

        elif sampler == 'hammersley':
            sampler_fn = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)

        else:
            raise Exception('unsupported sampler')

        return sampler_fn

    def scale(self, sample):

        lower = np.broadcast_to(self.bounds[0], self.dim)
        upper = np.broadcast_to(self.bounds[1], self.dim)

        return sample * (upper - lower) + lower

    @staticmethod
    def __Array_to_Tensor(list_A, device):
        
        list_T = []
        for a in list_A:
            t = torch.Tensor(a).to(device)
            list_T.append(t)
        return list_T
    
    def sample(self, n, tensor=False, device='cpu'):

        x = self.sampler.generate(dimensions=self.unit_sq, n_samples=n, random_state=self.seed)
        x = np.array(x)
        x = self.scale(x)
        x = np.split(x, self.dim, 1)

        sample = *x, self.func(*x)

        if tensor:
            return self.__Array_to_Tensor(sample, device)
        else:
            return sample