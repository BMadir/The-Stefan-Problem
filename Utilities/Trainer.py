import torch
import numpy as np
import copy
from AutoDiff import AG_grad
from pinns_weights import*

__all__ = ["Trainer"]

class Trainer:
    def __init__(self, model, validation_data, in_validation_data=None, *args, **kwargs):
        
        self._model = model
        self._weights = model.weights
        self.parameters = model.parameters
        self.zero_grad = model.zero_grad
        self.device = model.device
        self.net = model.net
        self.net_grad = model.net_grad
        self.net_res = model.net_res
        self.state_dict = model.state_dict

        self.optimizer = None
        self.lr_scheduler = None
        self.loss_weights = None
        self.error_weights = None

        self.validation_data_is_available = False
        if validation_data is not None:
            n, m = validation_data.shape
            validation_data = torch.Tensor(validation_data).to(self.device)
            splitted_vd = torch.hsplit(validation_data, m)

            if in_validation_data is not  None:
                self.val_in, self.val_out = splitted_vd[:in_validation_data], splitted_vd[in_validation_data:]
            else:
                self.val_in, self.val_out = splitted_vd[:m-1], splitted_vd[-1]
            self.validation_data_is_available = True

    @torch.no_grad
    def l2_error(self, inputs, targets, dims=None):
        if dims is not None:
            outputs = self.net(*inputs)[dims]
        else:
            outputs = self.net(*inputs)
        assert type(outputs) == type(targets)
        if torch.is_tensor(outputs):
            error = torch.linalg.norm(targets - outputs, 2) / torch.linalg.norm(targets, 2)
            return error.item()
        else:
            error = tuple((torch.linalg.norm(targets[i] - out, 2) / torch.linalg.norm(targets[i], 2)).item() for i, out in enumerate(outputs))
            return error

    def validation_error(self, epoch):
        error = self.l2_error(self.val_in, self.val_out)
        self.losses_dict["l2"].append(error)
        if error == min(self.losses_dict["l2"]):
            self.save_checkpoint(epoch)

    @staticmethod
    def mse_error(outputs, targets=0):
        error = (targets - outputs)
        loss = error.pow(2).mean()
        return loss, error

    # Compute the loss weights
    def compute_weights(self, method, losses, weights, **kwargs):
        if method == "lr_anneling":
            return lr_annealing(losses=losses, params=self._weights(), weights=weights, **kwargs)
        else:
            return weights
        
    # Compute the total loss
    def compute_loss(self, method, losses, errors, weights, **kwargs):
        if (method is None) or (method == "lr_anneling"):
            return sum(w * l for w, l in zip(weights, losses))
        elif method == "penalty":
            return penalty(errors=errors, weights=self.error_weights, **kwargs)
        elif method == "AL":
            return augmented_lagrangian(errors=errors, weights=self.error_weights, **kwargs)
        elif method == "SA":
            return soft_attention(errors=errors, weights=self.error_weights, **kwargs)
        elif method == "rad":
            return rad_loss(errors=errors, weights=weights)
        else:
            raise RuntimeError(f"{method} is not supported")

    # Closure function
    def closure_fn(self, epoch, weighting_dict, **kwargs):
        frequency = weighting_dict.get("frequency")
        if frequency is None:
            frequency = 1
        self.zero_grad()
        losses, errors = self.loss_fn()
        if epoch % frequency == 0:
            self.loss_weights = self.compute_weights(losses=losses, weights=self.loss_weights, **weighting_dict)
        total_loss = self.compute_loss(losses=losses, errors=errors, weights=self.loss_weights, **weighting_dict, **kwargs)
        total_loss.backward()
        # save last values of the losses:
        self._last_losses = [loss.item() for loss in losses]
        self._last_losses.append(total_loss.item())
        return total_loss

    def _set_weights(self, weighting_dict):
        """ helper function initializes the weights """
        weighting_dict_ = copy.deepcopy(weighting_dict)
        # Loss weights
        try:
            self.loss_weights = weighting_dict_.pop("weights")
        except:
            self.loss_weights = [1. for i in self.loss_fn()[0]]

        # Error weights
        method = weighting_dict.get("method")
        if method in ("penalty", "AL", "SA"):
            params_data = weighting_dict.get("params_data")
            if self.error_weights is None:
                self.error_weights = _new_params(params_data, device=self.device)
        else:
            self.error_weights = None
        return weighting_dict_

    def _set_opt(self, weighting_dict, optim_dict):
        optim_dict_ = copy.deepcopy(optim_dict)

        # Parameters
        lr = weighting_dict.get('lr')
        if lr is None:
            lr = 1e-2
        if self.error_weights is not None:
            params_list = [
                {'params': self.parameters()},
                {'params': self.error_weights, 'lr': lr, 'maximize': True}
            ]
        else:
            params_list = self.parameters()

        # Optimizer & lr_scheduler
        try:
            optimizer = optim_dict_.pop('optimizer')
        except:
            optimizer = "Adam"
        try:
            exp_lr = optim_dict_.pop('exp_lr')
        except:
            exp_lr = None
        try:
            exp_dr = optim_dict_.pop('exp_dr')
        except:
            exp_dr = .9
        try:
            exp_ds = optim_dict_.pop('exp_ds')
        except:
            exp_ds = 8000
        self.optimizer = getattr(torch.optim, optimizer)(params_list, **optim_dict_)
        self.zero_grad = self.optimizer.zero_grad
        if exp_lr is not None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda t: exp_dr ** (t / exp_ds))

    def step(self, closure):
        if self.optimizer is not None:
            loss = self.optimizer.step(closure)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return loss
    
    @torch.no_grad
    def weights_norm(self, *args):
        _norm = list()
        for w in args:
            _norm.append(torch.linalg.norm(w).item())
        return  _norm
    
    # SAVE
    def save(self, losses=True, weights=False, optimizer=False, file='', name=''):
        if losses:
            for key, val in self.losses_dict.items():
                np.savetxt(file + name + 'loss_' + key, val)
        if weights:
            for key, val in self.weights_dict.items():
                np.savetxt(file + name + 'weight_' + key, val)
        if optimizer:
            optimizer_dict = {
                'optimizer': self.optimizer.state_dict(),
                'lr_sched': None,
            }
            torch.save(optimizer_dict, "optim.pth")

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model': self.state_dict()
        }
        torch.save(checkpoint, f'checkpoint.pth')

    def check_if_best(val):
        if val == min(self.losses['l2']):
            best_model = copy.deepcopy(self.state_dict())
