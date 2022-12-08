## Simple Kradagrad++ that extends official unoptimized Shampoo implementation

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

import positive_matrix_functions as mf
from shampoo import (
    Shampoo, Preconditioner,
    STEP, MOMENTUM, PRECONDITIONER, GRAFT,
    LayerwiseGrafting, AdamGraft, AdagradGraft, SGDGraft, Graft
)


class KrADPreconditioner(Preconditioner):

    @torch.no_grad()
    def __init__(self, var, hps, eps_override=1.0, debug=False):
        """
        debug mode print tensor values likely to be non-finite, raises exceptions
        eps_override changes initialization of preconditioner diagonal (usually 1.0)
        """
        super().__init__(var, hps, eps_override=eps_override)
        self.debug = debug
        
    @torch.no_grad()
    def exponent_for_preconditioner(self):
        """Returns exponent to use for pth root M^{1/p}.
        If `inverse_exponent_override` is set, use ONS-style update
        """
        if self._hps.inverse_exponent_override > 0:
            return len(self._transformed_shape)
        return 2 * len(self._transformed_shape)

    @torch.no_grad()
    def add_statistics(self, grad):
        """Compute inverse KrAD statistics from gradients and add to the correct state entries.

        Args:
          grad: Gradient to compute statistics from.
        """
        if not self.statistics: return
        partitioned_grads = self.partition_grad(grad)
        w1 = self._hps.beta2
        damping = self._hps.matrix_eps > 0
        rank = len(self._transformed_shape)
        for j, grad in enumerate(partitioned_grads):
            for i in range(rank):
                axes = list(range(i)) + list(range(i + 1, rank))
                stat = self.statistics[j*rank + i]
                if w1 < 1:
                    stat.mul_(1/w1)

                if damping:
                    stat_2_eps = stat.mul(-self._hps.matrix_eps).mm(stat.T)
                    if self.debug and not stat_2_eps.isfinite().all():
                        print('stat2eps', stat_2_eps, '\nstat', stat)
                        raise ValueError('stat2eps broke')
                    stat.add_(stat_2_eps)

                ggt = torch.tensordot(grad, grad, [axes, axes])
                if self.debug and not ggt.isfinite().all():
                    print('ggt', ggt, '\ngrad', grad)
                    raise ValueError('ggt broke')

                geom_fact = ggt.mm(stat.T)

                t_k = - (1 + mf.matrices_norm(geom_fact, 'fro'))
                geom_fact.mul_(1/t_k)
                DX = stat.mm(geom_fact)
                self.statistics[j*rank + i].add_(DX)


    @torch.no_grad()
    def compute_preconditioners(self, **kwargs):
        """Compute L^{1/exp} for each statistics matrix L"""
        exp = self.exponent_for_preconditioner()
        for i, stat in enumerate(self.statistics):
            if stat.device.type == 'cpu':
                try:
                    self.preconditioners[i] = mf.matrix_power_svd(stat, 1 / exp) if exp > 1 else stat
                except Exception as err:
                    if self.debug:
                        print('stat', stat)
                    raise err
            else:
                self.preconditioners[i] = mf.mat_root(
                    stat[None, ...], exp,
                    self.preconditioners[i][None, ...],
                    iters=8, tol=1e-4, inner_iters=4, inner_tol=1e-3
                )[0]  # mf.mat_root operates on batches

    @torch.no_grad()
    def partition_grad(self, grad):
        """Partition gradient.
        Args:
          grad: Gradient to partition.
        """
        reshaped_grad = torch.reshape(grad.detach(), self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        return partitioned_grads

class KradagradPP(Shampoo):
    r"""Implements a simple version of Kradagrad++ Optimizer Algorithm.
    Extends the unoptimized official pytorch implementation of Shampoo
    """
    @torch.no_grad()
    def __init__(self, params, **kwargs):
        # extra accouting for batch processing:
        # group by exponent, sort by size
        # group similar sizes together to minimize padding
        self.cpu = kwargs.get('cpu', False)
        self.initialized = False
        self._step = 0
        self.tensor_batch_size = kwargs.get('tensor_batch_size', 16)
        
        # store references to Tensors for batch processing
        self.param_master_dict = defaultdict(list)
        self.stat_master_dict = defaultdict(list)
        self.cond_master_dict = defaultdict(list)
        
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def init_var_state(self, var, state):
        """Initialize the PyTorch state of for a single variable."""
        # group by exponent for batching
        # we can further improve by sorting by size, but that requires advanced bookkeeping
        prec = KrADPreconditioner(var, self.hps, eps_override=1.0)
        exp = prec.exponent_for_preconditioner()
        self.param_master_dict[exp].append(var)
        self.stat_master_dict[exp].append(prec.statistics)
        self.cond_master_dict[exp].append(prec.preconditioners)

        # original, pared down        
        var = var.detach()
        state[MOMENTUM] = torch.zeros_like(var.data, device=var.device)
        state[PRECONDITIONER] = prec
        if self.cpu:
            state[STEP] = 0
            if self.hps.graft_type == LayerwiseGrafting.ADAM:
                state[GRAFT] = AdamGraft(self.hps, var)
            else:
                state[GRAFT] = Graft(self.hps, var)

    # reimplementation for batch processing on gpu
    @torch.no_grad()
    def step(self, closure=None):
        if self.cpu:
            super().step(closure)
            return
        hps = self.hps
        # GPU batch processing, currently no grafting
        # we can still initialize individually
        if not self.initialized:
            for group in self.param_groups:
                lr = group['lr']
                for p in group['params']:
                    if p.grad is None: continue
                    if not (state := self.state[p]):
                        self.init_var_state(p, state)
            self.initialized = True
        for group in self.param_groups:
            lr = group['lr']
            partitioned_grads_dict = defaultdict(list)
            for p in group['params']:
                if p.grad is None: continue
                prec = self.state[p]
                exp = prec.exponent_for_preconditioner()
                partitioned_grads_dict[exp].append(prec.partition_grads(p.grad.detach()))
                
            self._step += 1

            # batch processing
            if self._step % hps.statistics_compute_steps == 0:
                self.batch_add_statistics(partitioned_grads_dict)
            if self._step % hps.preconditioning_compute_steps == 0:
                self.batch_compute_preconditioners(step=self.step)

            # apply them in batch too
            krad_partitioned_grads_dict = partitioned_grads_dict
            if self._step >= self.hps.start_preconditioning_step:
                krad_partitioned_grads_dict = self.batch_preconditioned_grad(partitioned_grads_dict)
            krad_grad_dict = self.batch_merge_partitions(krad_partitioned_grads_dict)
            krad_grad_list = self.unbatch(krad_grad_dict)


            ## Do in series (easier, not as much gain from batched)
            for ix, p in enumerate(group['params']):
                state = self.state[p]
                # Weight decay
                if self.hps.weight_decay != 0.0:
                    krad_grad_list[ix].add_(p.data, alpha=hps.weight_decay)

                # Momentum and Nesterov momentum, if needed
                state[MOMENTUM].mul_(group['momentum']).add_(krad_grad_list[ix])

                momentum_update = state[MOMENTUM]
                wd_update = krad_grad_list[ix]

                if hps.nesterov:
                    momentum_update.mul_(group['momentum']).add_(wd_update)

                # Final update
                p.data.add_(momentum_update, alpha=-lr)

    def batch_add_statistics(self, partitioned_grads_dict):
        pass
    
    # TODO batch processing
    def batch_compute_preconditioners(self, step):
        pass
    
    def batch_preconditioned_grad():
        pass
    
    def batch_merge_partitions():
        pass
    
    def unbatch():
        pass