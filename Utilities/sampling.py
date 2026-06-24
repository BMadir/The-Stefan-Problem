import numpy as np
import torch
import skopt

__all__ = ["Sampler", "Join_samplers", "SeqToSeq", "Adaptor", "_rar_fn", "_rad_fn", "data_sampler"]
class Sampler:
    def __init__(self, l_bounds, u_bounds, func=None, sampler='lhs', seed=1234):

        assert len(l_bounds) == len(u_bounds)
        self.dim = len(l_bounds)
        self.bounds = l_bounds, u_bounds
        self.sampler = self.__sampler_fn(sampler.lower())
        self.unit_sq = [(0., 1.) for l, u in zip(l_bounds, u_bounds)]
        if func is None:
            func = lambda *args: np.zeros_like(args[0])
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
        elif sampler == 'hammersly':
            sampler_fn = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
        else:
            raise Exception('unsupported sampler')
        return sampler_fn

    def scale(self, sample):
        lower = np.broadcast_to(self.bounds[0], self.dim)
        upper = np.broadcast_to(self.bounds[1], self.dim)
        return sample * (upper - lower) + lower

    @staticmethod
    def _Array_to_Tensor(list_A, device):
        list_T = []
        for a in list_A:
            t = torch.Tensor(a).to(device)
            list_T.append(t)
        return tuple(list_T)
    
    def sample(self, n, tensor=True, device='cpu'):
        x = self.sampler.generate(dimensions=self.unit_sq, n_samples=n, random_state=self.seed)
        x = np.array(x)
        x = self.scale(x)
        x = np.split(x, self.dim, 1)
        
        if isinstance(self.func, list):
            sample = *x, *[f(*x) for f in self.func]
        else:
            sample = *x, self.func(*x)
        if tensor:
            return self._Array_to_Tensor(sample, device)
        else:
            return sample
        
class Join_samplers:
    def __init__(self, samplers, percentages=None):
        self.samplers = samplers
        if percentages is not None:
            assert len(percentages) == len(samplers)
            assert sum(percentages) == 1
        else:
            percentages = len(samplers)* [1/len(samplers)]
        self.percentages = percentages

    def sample(self, n, **kwargs):
        # q, r = divmod(n, len(self.samplers))
        l = [int(n* p) for p in self.percentages]
        l[0] += n - sum(l)

        samples = list(self.samplers[0].sample(l[0], **kwargs))
        for j, sampler in enumerate(self.samplers[1:]):
            sample = sampler.sample(l[j+1], **kwargs)
            for i, s in enumerate(sample):
                if torch.is_tensor(s):
                    samples[i] = torch.vstack([samples[i], s])
                else:
                    samples[i] = np.vstack([samples[i], s])
        return tuple(samples)


class SeqToSeq(Sampler):
    def __init__(self, n_seq, n_col, n_iter, l_bounds, u_bounds, func=None, sampler='lhs', **kwargs):
        super().__init__(l_bounds, u_bounds, func, sampler, **kwargs)
        self.n_seq = n_seq
        self.n_col = n_col
        self.seq_bounds = self._seq_bounds(n_seq, self.bounds, 0)
        self.seq_sizes = self._seq_sizes(n_seq, n_col)

        self.n_iter = n_iter
        iterations = [n_iter for _ in self.seq_sizes]
        self.iterations = [sum(iterations[:i]) for i in range(len(iterations))]

        self.samples = self._samples(self.seq_sizes)

    def _seq_bounds(self, n_seq, bounds, indice):
        seq = lambda k: bounds[0][indice] + k * (bounds[1][indice] - bounds[0][indice]) / n_seq
        seq_bounds = []
        for k in range(1, n_seq + 1):
            _bounds = bounds[1][:]
            _bounds[indice] = seq(k)
            _bounds = [bounds[0], _bounds]
            seq_bounds.append(_bounds)
        return seq_bounds

    def _seq_sizes(self, n_seq, size):
        size_0 = int(0.1 * size)
        size_k = int((size - size_0) / (n_seq - 1)) + 1
        tot = size_0 + (n_seq - 1) * size_k
        size_0 = size_0 - (tot - size)
        return [size_0 + k * size_k for k in range(n_seq)]

    def __scale(self, sample, bounds):
        dim = len(bounds[0])
        lower = np.broadcast_to(bounds[0], dim)
        upper = np.broadcast_to(bounds[1], dim)
        return sample * (upper - lower) + lower

    def __sample(self, n, bounds, tensor=True, device='cpu'):
        x = self.sampler.generate(dimensions=self.unit_sq, n_samples=n, random_state=self.seed)
        x = np.array(x)
        x = self.__scale(x, bounds)
        x = np.split(x, self.dim, 1)

        if isinstance(self.func, list):
            sample = *x, *[f(*x) for f in self.func]
        else:
            sample = *x, self.func(*x)

        if tensor:
            return self._Array_to_Tensor(sample, device)
        else:
            return sample

    def _samples(self, seq_sizes):
        samples = []
        for (size, bounds) in zip(seq_sizes, self.seq_bounds):
            sample = self.__sample(size, bounds, device="cuda")
            samples.append(sample)

        self.samples = samples
        return samples


def _rar_fn(sample_fn, func, num_in, num_out, **kwargs):
    """ Residual-based adaptive refinement """
    *sample, output = sample_fn(num_in, **kwargs)
    abs_value = func(*sample).abs()
    with torch.no_grad():
        values, indices = torch.topk(abs_value, num_out, dim=0)
        indices = indices.flatten()
        return *[s[indices] for s in sample], output[indices], abs_value[indices]



def _rad_fn(sample_fn, func, num_in, num_out, k, c, seed=1234, **kwargs):
    generator = torch.Generator("cuda").manual_seed(seed)

    """ Residual-based adaptive distribution """
    *sample, output = sample_fn(num_in, **kwargs)
    abs_value = func(*sample).abs().flatten()
    with torch.no_grad():
        density = (abs_value ** k) / (abs_value ** k).mean() + c
        density /= density.sum()
        indices = torch.multinomial(density, num_out, generator=generator)
        indices = indices.flatten()
        return *[s[indices] for s in sample], output[indices], density[indices]

# Adaptor
class Adaptor:
    def __init__(self, sample_fn, func, method=None, num_in=None, num_out=None, k=0.5, c=1., seed=1234, **kwargs):
        self.method = method
        self.sample_fn = sample_fn
        self.func = func
        self.num_in = num_in
        self.num_out = num_out
        self.kwargs = kwargs
        self.state_dict = None
        self.seed = seed

        self.k = k
        self.c = c

    def _update(self, sample, set_2=False, set_1_size=None, save=False):
        if self.method.lower() == "rar":
            adapted = self.rar_fn()
        elif self.method.lower() == "rad":
            adapted = self.rad_fn()
        else:
            raise Exception("unsupported adaptor")
        if save:
            self._save(adapted)
        if set_2 and set_1_size:
            return [torch.vstack([s[0:set_1_size, :], adapted[i]]) for i, s in enumerate(sample)]
        else:
            return adapted

    def _save(self, sample):
        if isinstance(sample, list):
            pass
        else:
            sample = list(sample)

        if self.state_dict is None:
            keys = ["sample_" + str(i) for i in range(len(sample))]
            values = [[] for i in range(len(sample))]
            self.state_dict = dict(zip(keys, values))

        num = len(sample)
        for i in range(num):
            self.state_dict["sample_" + str(i)].append(sample[i].detach().cpu())

    def rar_fn(self):
        return _rar_fn(self.sample_fn, self.func, self.num_in, self.num_out, **self.kwargs)
    
    def rad_fn(self):
        return _rad_fn(self.sample_fn, self.func, self.num_in, self.num_out, self.k, self.c, seed=self.seed, **self.kwargs)

    def sample(self, num_out, **kwargs):
        if self.method.lower() == "rar":
            return _rar_fn(self.sample_fn, self.func, self.num_in, num_out, **kwargs)
        elif self.method.lower() == "rad":
            k = kwargs.get("k")
            c = kwargs.get("c")
            if k is None:
                k = self.k
            else:
                kwargs.pop("k")
            if c is None:
                c = self.c
            else:
                kwargs.pop("c")
            return _rad_fn(self.sample_fn, self.func, self.num_in, num_out, k, c, seed=self.seed, **kwargs)


class data_sampler:
    def __init__(self, data, device="cuda"):
        n, m = data.shape
        self.data = np.hsplit(data, m)
        self.device = device
        self.n = n

    def __fit(self, n_in, n_out, inputs=None):
        if inputs is None:
            inputs = (1 / n_in) * torch.ones(n_in)
        outputs = torch.multinomial(inputs, n_out)
        return outputs

    @staticmethod
    def _Array_to_Tensor(list_A, device):
        list_T = []
        for a in list_A:
            t = torch.Tensor(a).to(device)
            list_T.append(t)
        return tuple(list_T)

    def sample(self, n, **kwargs):
        sample = self._Array_to_Tensor(self.data, self.device)
        idx = self.__fit(self.n, n)
        return tuple(s[idx] for s in sample)