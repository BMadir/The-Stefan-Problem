import torch
import numpy as np
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from torch.func import functional_call
from functools import wraps


__all__ = ["vect_parameters_to_dic", "vector_to_parameters", "parameters_to_vector", "functional_call", "evaluate"]


def _Array_to_Tensor(list_A, device, half):
    list_T = []
    for a in list_A:
        t = torch.Tensor(a).to(device)
        if half:
            t = t.half()
        list_T.append(t)
    return tuple(list_T)

def torch_out(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        outputs = func(*args, **kwargs)
        if isinstance(outputs, np.ndarray):
            return torch.Tensor(outputs)
        elif isinstance(outputs, list | tuple):
            return tuple(torch.Tensor(out) if isinstance(out, np.ndarray) else out for out in outputs)
    return wrapper

def torch_in(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(outputs, np.ndarray):
            return torch.Tensor(outputs)
        elif isinstance(outputs, list | tuple):
            return tuple(torch.Tensor(out) if isinstance(out, np.ndarray) else out for out in outputs)
    return wrapper


def split_fn(self, samples, in_dim):
    return samples[:in_dim], samples[in_dim:]

def vect_parameters_to_dic(vect_parameters, dict_parameters):
    if not torch.is_tensor(vect_parameters):
        vect_parameters = torch.Tensor(vect_parameters)
    offset = 0
    for n, p in dict_parameters.items():
        numel = p.numel()
        dict_parameters[n] = vect_parameters[offset : offset + numel].reshape(p.shape)
        offset += numel
    assert offset == vect_parameters.numel(), 'invalid size for vect_parameters'
    return dict_parameters
    
    

def evaluate(self, *args):
    inputs = list()
    for arg in args:
        if not torch.is_tensor(arg):
            arg = torch.Tensor(arg)
        inputs.append(arg.to(self.device))
    outputs = self.net(*inputs)
    if torch.is_tensor(outputs):
        return outputs.detach().cpu().numpy()
    else:
        return tuple(out.detach().cpu().numpy() for out in outputs)
