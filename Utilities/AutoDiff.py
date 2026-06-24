import torch
from functools import wraps

__all__ = ["AG_grad", "args_requires_grad"]

def __AG_grad(outputs, inputs):
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )

    if torch.is_tensor(inputs):
        return grad[0]
    else:
        return grad

def AG_grad(outputs, inputs, order=1, mixed=False):
    if order == 0:
        return outputs
    else:
        outputs = __AG_grad(outputs, inputs)
        if torch.is_tensor(inputs):
            return AG_grad(outputs, inputs, order-1)
        else:
            if mixed:
                return tuple(AG_grad(out, inputs, order-1) for out in outputs)
            else:
                return tuple(AG_grad(out, inputs[i], order-1) for i, out in enumerate(outputs))


def args_requires_grad(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if torch.is_tensor(arg) and not arg.requires_grad:
                arg.requires_grad_()
        output = func(*args, **kwargs)
        return output
    return wrapper
