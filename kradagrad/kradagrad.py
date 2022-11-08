## Reference code: https://github.com/moskomule/shampoo.pytorch
## More reference: https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/shampoo.py

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

import matrix_sqrt


class Kradagrad(Optimizer):
    r"""Implements KrADagrad Optimizer Algorithm.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        epsilon: epsilon added to each mat_gbar_j for numerical stability
            (default: 1e-4)
        update_sched: function that takes as input iteration number and outputs
            a `tuple` of `bool` of whether or not to update L or R in that
            iteration. If both are `True`, L will take precedence over R
        exponent: `alpha`, if 1 use ONS-style, if 1/2 use Adagrad-style
    Example:
        >>> import kradagrad as kg
        >>> optimizer = kg.Kradagrad(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: float = 1e-1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: Callable[[int], Tuple[bool, bool]] = lambda x: (x % 2 == 0, x % 2 == 1),
        exponent: float = 1,
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
            raise ValueError('Invalid momentum value: {} < 0.0'.format(momentum))
        if update_freq < 1:
            raise ValueError('Invalid momentum value: {} < 1'.format(momentum))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
        )
        super(Kradagrad, self).__init__(params, defaults)
    
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


        ### TODO: turn this from shampoo into kradagrad and then dupe it to pp
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
                        state['momentum_buffer'] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        # precondition matrices
                        state['precond_{}'.format(dim_id)] = group[
                            'epsilon'
                        ] * torch.eye(dim, out=grad.new(dim, dim))
                        state[
                            'inv_precond_{dim_id}'.format(dim_id=dim_id)
                        ] = grad.new(dim, dim).zero_()

                if momentum > 0:
                    grad.mul_(1 - momentum).add_(
                        state['momentum_buffer'], alpha=momentum
                    )

                if weight_decay > 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # See Algorithm 2 for detail
                for dim_id, dim in enumerate(grad.size()):
                    precond = state['precond_{}'.format(dim_id)]
                    inv_precond = state['inv_precond_{}'.format(dim_id)]

                    # mat_{dim_id}(grad)
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()
                    precond.add_(grad @ grad_t)
                    if state['step'] % group['update_freq'] == 0:
                        inv_precond.copy_(_matrix_power(precond, -1 / order))

                    if dim_id == order - 1:
                        # finally
                        grad = grad_t @ inv_precond
                        # grad: (-1, last_dim)
                        grad = grad.view(original_size)
                    else:
                        # if not final
                        grad = inv_precond @ grad
                        # grad (dim, -1)
                        grad = grad.view(transposed_size)

                state['step'] += 1
                state['momentum_buffer'] = grad
                p.data.add_(grad, alpha=-group['lr'])

        return loss

