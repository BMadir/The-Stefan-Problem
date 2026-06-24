import torch
from AutoDiff import AG_grad

__all__ = ["lr_annealing", "_new_params", "penalty", "augmented_lagrangian", "soft_attention", "rad_loss"]

# Learning rate annealing algorithme:
def _grad_losses(losses, params, func=lambda x: x):
    params = tuple(params)
    grads = list()
    for loss in losses:
        grad = AG_grad(loss, params)
        grad = torch.cat([g.detach().view(-1) for g in grad])
        grads.append(func(grad))
    return grads

#"""
def lr_annealing(
        losses,
        params,
        weights=None,
        res_indices=None,
        alpha=0.6,
        sup=1e5,
        **kwargs
):
    if weights is None:
        weights = [1. for _ in losses]
    if res_indices is None:
        res_indices = [len(losses) - 1]
    grads = _grad_losses(losses, params)
    res_grads = [weights[i] * grads[i] for i in res_indices]
    max_res = sum(res_grads).abs().max().item()
    new_weights = weights[:]
    eps = 0
    for i, (w, grad) in enumerate(zip(weights, grads)):
        if i in res_indices:
            pass
        else:
            m = grad.abs().mean().item()
            w = (1. - alpha) * w + alpha * (max_res / (w * m + eps))
            new_weights[i] = min(sup, w)
    return new_weights

# define the Lagrange multiplier (weights adjusted using GD):
def _new_params(data, device):
    params = torch.nn.ParameterList()
    for d in data:
        if torch.is_tensor(d):
            p = torch.nn.Parameter(d.to(device))
        else:
            p = torch.tensor([1.]).to(device).requires_grad_(False)
        params.append(p)
    return params

# Penalty method:
def penalty(
        errors,
        weights,
        indices=None,
        copy=None,
        **kwargs
):
    if indices is None:
        indices = [len(errors) - 1]
    loss = sum(errors[i].pow(2).mean() for i in indices)
    errors = [errors[i] for i in set(range(len(errors))) - set(indices)]
    if copy is not None:
        _weights = []
        for i, p in enumerate(LM):
            for _ in range(copy[i]):
                _weights.append(p)
    else:
        _weights = weights

    for i, e in enumerate(errors):
        c = _weights[i] * e.pow(2)
        loss += c.mean()
    return loss

# Augmented Lagrangian method:
def augmented_lagrangian(
        errors,
        weights,
        indices=None,
        beta=1.,
        copy=None,
        **kwargs
):
    if indices is None:
        indices = [len(errors) - 1]
    loss = sum(errors[i].pow(2).mean() for i in indices)
    errors = [errors[i] for i in set(range(len(errors))) - set(indices)]
    if copy is not None:
        _weights = []
        for i, p in enumerate(weights):
            for _ in range(copy[i]):
                _weights.append(p)
    else:
        _weights = weights

    for i, e in enumerate(errors):
        c = beta * e.pow(2) + _weights[i] * e
        loss += c.mean()
    return loss

# Soft Attention mechanism:
def soft_attention(
        errors,
        weights,
        mask_fns,
        copy=None,
        no_grad=None,
        **kwargs
):

    if copy is not None:
        _weights = []
        for i, p in enumerate(weights):
            for _ in range(copy[i]):
                _weights.append(p)
    else:
        _weights = weights
    if callable(mask_fns):
        fn = len(_weights) *  [mask_fns]
    else:
        fn = mask_fns

    if no_grad is not None:
        for i in no_grad:
            weights[i].requires_grad_(False)
            fn[i] = lambda x:x
    loss = 0.
    for i, (w, e) in enumerate(zip(_weights, errors)):
        c = fn[i](w) * e.pow(2)
        loss += c.mean()
    return loss

# NTK weighting: Soon!
# EW_functions:

def rad_loss(errors, weights):
    loss = 0
    for i, error in enumerate(errors):
        loss += (weights[i] * error.pow(2)).mean()
    return loss