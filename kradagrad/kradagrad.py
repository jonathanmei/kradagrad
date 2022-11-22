## Reference code: https://github.com/moskomule/shampoo.pytorch
## More reference: https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/shampoo.py

import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

import matrix_root as mr


def roundup(p, m):
    # how many groups of m are in p?
    # p: int or float
    # m: int
    return (p // m) + int(p % m > 0)

class KradagradPPBatched(Optimizer):
    r"""Implements KrADagrad++ Optimizer Algorithm, batched statistics updates.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-2)
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        epsilon: initialization, for numerical stability of original algo (default: 1e-4)
        update_freq: number of optimization steps in between updating preconditioner (default: 100)
        max_size: preconditioner kronecker factor dimension (default: 64)
        batches: how many preconditioners to update in parallel (default: 8)
        root: 1/`alpha`, if 1 use ONS-style, if 2 use Adagrad-style (default: 2)
        inv_tol: error threshhold to terminate inner Newton update on matrix inverse (default: 1e-4)
        root_tol: error threshhold to terminate outer Newton update on matrix root (default: 1e-4)
    Example:
        >>> import kradagrad as kg
        >>> optimizer = kg.Kradagrad(model.parameters())
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
        update_freq: int = 100,
        max_size: int = 64,
        batches: int = 8,
        root: int = 2,
        inv_tol: float = 1e-4,
        root_tol: float = 1e-4,
    ):

        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {} <= 0.0'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum value: {} < 0.0'.format(momentum))
        if weight_decay < 0.0:
            raise ValueError('Invalid weight_decay value: {} < 0.0'.format(weight_decay))
        if epsilon < 0.0:
            raise ValueError('Invalid epsilon value: {} < 0.0'.format(epsilon))
        if update_freq < 1:
            raise ValueError('Invalid update_freq value: {} < 1'.format(update_freq))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
            max_size=max_size,
            batches=batches,
            root=root,
            inv_tol=inv_tol,
            root_tol=root_tol
        )
        super(Kradagrad, self).__init__(params, defaults)
        # Initialize
        for group in self.param_groups:
            epsilon = group['epsilon']
            max_size = group['max_size']
            root = group['root']

            # # Assume requires_grad doesn't change during training:
            # param_sizes = [p_.size() for p_ in group['params'] if p_.requires_grad]
            # def matricize_size(size):
            #     if len(size) == 1:  # e.g. bias vectors
            #        mat_sz = (roundup(math.sqrt(size[0]), 1), ) * 2
            #     elif len(size) > 2:  # higher dim tensors
            #         split = 
            #         math.prod()
            #     return mat_sz
            # param_sizes = [matricize_size(sz_) for sz_ in param_sizes if len(sz_) != 2]
            # # break params into chunks of (max_size, max_size), padding along the way
            # num_blocks_per_param = [
            #     roundup(sz[0], max_size) * roundup(sz[1], max_size)
            #     for sz in param_sizes
            # ]
            # num_blocks = sum(num_blocks_per_param)

            # Less wasteful but prob won't result in as good a precond matrix:
            # 1) vectorize and concat all params
            # 2) make square chunks
            param_Ns = [p_.numel() for p_ in group['params'] if p_.requires_grad]
            num_blocks = roundup(sum(param_Ns), max_size * max_size)

            # bookkeeping for where each square chunk comes from
            param_chunks = []
            param_num_0 = 0
            param_ind_0 = 0
            remaining = 0  # how many params in queue to be chunked
            for i_ in range(num_blocks):
                filled = False
                param_num_1, param_ind_1 = param_num_0, param_ind_0
                while not filled:
                    if remaining < max_size * max_size:
                        remaining += group['params'][param_num_1].numel()
                        param_num_1 += 1
                    else:
                        filled = True
                        overflow = remaining - max_size * max_size
                        param_ind_1 = group['params'][param_num_1 - 1].numel() - overflow
                        remaining = overflow
                param_chunks.append((param_num_0, param_ind_0), (param_num_1 - 1, param_ind_1))
                param_num_0, param_ind_1 = param_num_1, param_ind_1
            
            self.param_chunks = param_chunks
            
            init_mag = 1 / epsilon
            init_rt_mag = torch.pow(init_mag, 0.5 / root)
            group_state_init = {}

            group_state_init.update({
                'L_{}'.format(i_): torch.eye(max_size) * init_mag
                for i_ in range(num_blocks)
            })
            group_state_init.update({
                'R_{}'.format(i_): torch.eye(max_size) * init_mag
                for i_ in range(num_blocks)
            })
            group_state_init.update({
                'L_rt_{}'.format(i_): torch.eye(max_size) * init_rt_mag
                for i_ in range(num_blocks)
            })
            group_state_init.update({
                'R_rt_{}'.format(i_): torch.eye(max_size) * init_rt_mag
                for i_ in range(num_blocks)
            })

            self.state[tuple(group['params'])] = group_state_init
    
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
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            max_size = group['max_size']
            
            # individual gradients
            for p_ in group['params']:
                if p_.grad is None:
                    continue
                state = self.state[p_]
                if len(state) == 0:
                    state['step'] = 0
                    grad = p_.grad.data
                    if momentum > 0:
                        state['momentum_buffer'] = grad.clone()
                else:
                    grad = p_.grad.data
                    if momentum > 0:
                        grad.mul_(1 - momentum).add_(
                            state['momentum_buffer'], alpha=momentum
                        )
                    if weight_decay > 0:
                        grad.add_(p_.data, alpha=group['weight_decay'])

            # preconditioners and main loop
            batches = group['batches']
            root = group['root']
            group_state = self.state[tuple(group['params'])]
            for index_ in range(roundup(len(group_state), batches)):
                g = _slice_grad(group['params'], batches, max_size, index_)
                G = g.view([batches, max_size, max_size])

                precond_range = range(max_size * index_, max_size * (index_ + 1))
                
                L_s = torch.stack([group_state[key_] for i_ in precon_range if (key_ := 'L_{}'.format(i_)) in group_state])
                R_s = torch.stack([group_state[key_] for i_ in precon_range if (key_ := 'R_{}'.format(i_)) in group_state])
                L_rt_s = torch.stack([group_state[key_] for i_ in precon_range if (key_ := 'L_rt_{}'.format(i_)) in group_state])
                R_rt_s = torch.stack([group_state[key_] for i_ in precon_range if (key_ := 'R_rt_{}'.format(i_)) in group_state])

                M_L = L.bmm(G.bmm(G_T)))
                t_kL_inv = 1 / (1 + torch.linalg.matrix_norm(M_L, ord='fro'))
                Delta_L = M_L @ L.transpose(-2, -1)

                M_R = R.bmm(G_T.bmm(G))
                t_kR_inv = 1 / (1 + torch.linalg.matrix_norm(M_R, ord='fro'))
                Delta_R = M_R @ R.transpose(-2, -1)

                L_s.sub_(Delta_L, t_kL_inv)
                R_s.sub_(Delta_R, t_kR_inv)
                if group_state['step'] % group['update_freq'] == 0:
                    L_rt_s.copy_(mr.matrix_even_root_warm(2 * root, L_s, L_rt_s))
                    R_rt_s.copy_(mr.matrix_even_root_warm(2 * root, R_s, R_rt_s))

                grad = L_rt_s.bmm(G.bmm(R_rt_s)).view([-1])
                _apply_grad(group['params'], grad, lr, batches, max_size, index_)

            state['step'] += 1

        return loss

    @torch.no_grad()
    def _slice_grad(self, param_list, batches, max_size, index):
        N = batches * max_size * max_size
        chunk = torch.Tensor.zeros([N])

        chunk_beg, chunk_end = self.param_chunks[index]
        param_num_end, param_ind_end = chunk_end
        param_num, param_ind = chunk_beg
        chunk_pointer = 0
        while chunk_pointer < N:
            if param_num < param_num_end:
                addition_size = param_list[param_num].numel() - param_ind
            else:
                addition_size = param_ind_end - param_ind
            chunk[chunk_pointer:chunk_pointer + addition_size] = param_list[param_num].view([-1])[param_ind:param_ind + addition_size]
            chunk_pointer += addition_size
            param_num += 1
            param_ind = 0
        return chunk.view([batches, max_size, max_size])

    @torch.no_grad()
    def _apply_grad(self, param_list, grad, lr, batches, max_size, index):
        N = batches * max_size * max_size
        chunk_beg, chunk_end = self.param_chunks[index]
        param_num_end, param_ind_end = chunk_end
        param_num, param_ind = chunk_beg
        chunk_pointer = 0
        while chunk_pointer < N:
            this_param = param_list[param_num]
            if param_num < param_num_end:
                addition_size = this_param.numel() - param_ind
            else:
                addition_size = param_ind_end - param_ind
            this_param.data.view([-1])[param_ind:param_ind + addition_size].add_(grad[chunk_pointer:chunk_pointer + addition_size], alpha=-lr)
            if momentum > 0:  # TODO: does this go here or before preconditioning...?
                state = self.state[this_param]
                state['momentum_buffer'].view([-1])[param_ind:param_ind + addition_size] = grad[chunk_pointer:chunk_pointer + addition_size]
            chunk_pointer += addition_size
            param_num += 1
            param_ind = 0

