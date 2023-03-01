## Reference code: https://github.com/moskomule/shampoo.pytorch
## More reference: https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/shampoo.py
## Ultimate official reference:
##  https://github.com/google-research/google-research/blob/master/scalable_shampoo/pytorch/shampoo.py

import math
from typing import Any, Dict, Iterable, Optional, Union

import torch
from torch.optim.optimizer import Optimizer

from . import positive_matrix_functions as mf
from .third_party.shampoo import matrix_functions as smf

class PreconditionedGradient(Optimizer):
    r"""Implements Preconditioned Gradient Optimizer Algorithm.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        epsilon: epsilon for damping (default: 1e-4)
        update_freq: interval between successive preconditioner updates (default: 1)
    Example:
        >>> import preconditioned_gradient as pg
        >>> optimizer = pg.KradagradPP(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 1,
    ):

        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {} <= 0.0'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum value: {} < 0.0'.format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {} < 0.0'.format(weight_decay)
            )
        if epsilon < 0.0:
            raise ValueError('Invalid epsilon value: {} < 0.0'.format(epsilon))
        if update_freq < 1:
            raise ValueError('Invalid update_freq value: {} < 1'.format(update_freq))
        self._epsilon = epsilon
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
        )
        super(PreconditionedGradient, self).__init__(params, defaults)

    @torch.no_grad()
    def _compute_ix(self, shape, step):
        """
        return dimension to update or None to update all dimensions
        """
        raise NotImplementedError

    @torch.no_grad()
    def _apply_precond_to_all_but(self, state, grad, ix):
        """
        in place apply precond to all dimensions of grad except that specified by `ix`
        """
        raise NotImplementedError

    @torch.no_grad()
    def _stat_init(self, dim, grad):
        """
        return the initial value for stat
        """
        raise NotImplementedError

    @torch.no_grad()
    def _stat_update(self, stat, grad_mat):
        """
        in place update stat from matricized gradient tensor
        """
        raise NotImplementedError

    @torch.no_grad()
    def _precond_update(self, precond, stat, order):
        """
        in place update preconditioner
        """
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group['momentum']
                weight_decay = group['weight_decay']
                if len(state) == 0:
                    state['step'] = 0
                    if momentum > 0:
                        state['momentum_buffer'] = torch.zeros_like(grad.data, device=grad.device)
                    for dim_id, dim in enumerate(original_size):
                        # initialize preconditioner matrices
                        state['stat_{}'.format(dim_id)] = self._stat_init(dim, grad)
                        state['precond_{}'.format(dim_id)] = torch.eye(dim, out=grad.new(dim, dim))
                        state['need_update_{}'.format(dim_id)] = False


                # Compute dim to be updated for Kradagrad
                # not for Shampoo or Kradagrad star, so ix should be None for those algo's
                ix = self._compute_ix(grad.size(), state['step'])
                if ix is not None:
                    # Precondittion all other dims for Kradagrad
                    self._apply_precond_to_all_but(state, grad, ix)
                # Compute statistics and preconditioners
                for dim_id, dim in enumerate(original_size):
                    # Update only one dim for Kradagrad
                    # all dims for Shampoo and Kradagrad star
                    if ix is None or ix == dim_id:
                        # mat_{dim_id}(grad)
                        grad = grad.transpose(0, dim_id).contiguous()
                        transposed_size = grad.size()
                        grad_mat = grad.view(dim, -1)

                        stat = state['stat_{}'.format(dim_id)]
                        precond = state['precond_{}'.format(dim_id)]
                        need_update = state['need_update_{}'.format(dim_id)]

                        self._stat_update(stat, grad_mat)
                        grad = grad.view(transposed_size).transpose(0, dim_id)
                        state['need_update_{}'.format(dim_id)] = True
                    if state['step'] % group['update_freq'] == 0:
                        # save some computation if `update_freq < dim`
                        if state['need_update_{}'.format(dim_id)]:
                            self._precond_update(precond, stat, order)

                # Apply preconditionining
                for dim_id, dim in enumerate(original_size):
                    # Precondition remaining dim for Kradagrad
                    # all dims for Shampoo and Kradagrad star
                    if ix is None or ix == dim_id:
                        stat = state['stat_{}'.format(dim_id)]
                        precond = state['precond_{}'.format(dim_id)]
                        need_update = state['need_update_{}'.format(dim_id)]

                        grad = grad.transpose(0, dim_id).contiguous()
                        transposed_size = grad.size()
                        grad = grad.view(dim, -1)
                        grad = precond @ grad
                        grad = grad.view(transposed_size)

                if ix is None:
                    grad = grad.moveaxis(0, -1).view(original_size)
                else:
                    grad = grad.transpose(0, ix).view(original_size)

                if weight_decay > 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                if momentum > 0:
                    state['momentum_buffer'].mul_(momentum).add(grad)
                momentum_update = state['momentum_buffer']
                wd_update = grad

                state['step'] += 1
                # Nesterov
                momentum_update.mul_(momentum).add_(wd_update)
                p.data.add_(momentum_update, alpha=-group['lr'])

        return loss


class Kradagrad(PreconditionedGradient):
    """
    implement kradagrad as a preconditioned gradient method
    """
    @torch.no_grad()
    def _compute_ix(self, shape, step):
        order = len(shape)
        if step < order ** 2:
            # simple equal updating schedule:
            ix = step % order
        else:
            # O(k^0.5) for everything except largest dim
            max_dim = max(shape)
            max_ix = [j for (j, x) in enumerate(shape) if x == max_dim][0]
            step_sqrt_floor = int(math.sqrt(step))
            exc = step - step_sqrt_floor ** 2
            ix = exc if exc < order else max_ix
        return ix

    @torch.no_grad()
    def _apply_precond_to_all_but(self, state, grad, ix):
        original_size = grad.size()
        grad_ = grad
        for dim_id, dim in enumerate(original_size):
            if ix != dim_id:
                precond = state['precond_{}'.format(dim_id)]
                grad = grad.transpose(0, dim_id).contiguous()
                transposed_size = grad.size()
                grad = grad.view(dim, -1)
                grad = precond @ grad
                grad = grad.view(transposed_size).transpose(0, dim_id)
        grad_.copy_(grad)

    @torch.no_grad()
    def _stat_init(self, dim, grad):
        return torch.eye(dim, out=grad.new(dim, dim))

    @torch.no_grad()
    def _stat_update(self, stat, grad_mat):
        stat_2_eps = stat.mul(-self._epsilon).mm(stat.T)
        stat.add_(stat_2_eps)
        ggt = grad_mat @ grad_mat.t()
        ggtl = ggt.type_as(stat).mm(stat.T)
        t_k = -(1 + mf.matrices_norm(ggtl, 'fro'))
        ggtl.mul_(1 / t_k)
        lggtl = stat.mm(ggtl)
        stat.add_(lggtl)

    @torch.no_grad()
    def _precond_update(self, precond, stat, order):
        precond.copy_(mf.matrix_power_svd(stat + 1e-12 * torch.eye(stat.size()[0], device=stat.device), 2))
        #precond.copy_(mf.mat_root(stat, 2))


class KradagradStar(Kradagrad):
    """
    implement kradagrad star as a preconditioned gradient method
    """
    @torch.no_grad()
    def _compute_ix(self, shape, step):
        return None

    @torch.no_grad()
    def _precond_update(self, precond, stat, order):
        precond.copy_(mf.matrix_power_svd(stat + 1e-12 * torch.eye(stat.size()[0], device=stat.device), 2 * order))
        #precond.copy_(mf.mat_root(stat, 2 * order))


class Shampoo(PreconditionedGradient):
    """
    implement shampoo as a preconditioned gradient method
    """
    @torch.no_grad()
    def _compute_ix(self, shape, step):
        return None

    @torch.no_grad()
    def _stat_init(self, dim, grad):
        return self._epsilon * torch.eye(dim, out=grad.new(dim, dim))

    @torch.no_grad()
    def _stat_update(self, stat, grad_mat):
        stat.add_(grad_mat @ grad_mat.t())

    @torch.no_grad()
    def _precond_update(self, precond, stat, order):
        precond.copy_(mf.matrix_power_svd(stat + 1e-12 * torch.eye(stat.size()[0], device=stat.device), -1 / (2 * order)))
        #precond.copy_(smf.ComputePower(stat, 2 * order, ridge_epsilon=1e-12))

