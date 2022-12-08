## Simple Kradagrad-- that extends official unoptimized Shampoo implementation

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


class KrADmmPreconditioner(Preconditioner):
    """Inherit from Preconditioner mainly for the block partitioning
    Otherwise treat as abstract and override everything else
    """
    @torch.no_grad()
    def __init__(self, var, hps, eps_override=1.0, debug=False):
        """
        debug mode print tensor values likely to be non-finite, raises exceptions
        eps_override changes initialization of preconditioner diagonal (usually 1.0)
        """
        super().__init__(var, hps, eps_override=eps_override)
        self.debug = debug
        self.updated = set()

    @torch.no_grad()
    def exponent_for_preconditioner(self):
        """Returns exponent to use for pth root M^{1/p}.
        """
        return 2

    @torch.no_grad()
    def add_statistic(self, grad, i):
        """Compute inverse KrAD statistic from gradients and add to the correct state entries.

        Args:
          grad: Gradient to compute statistics from.
          i: rank for which to add statistic to
        """
        if not self.statistics: return
        grad = grad.type(torch.float32)
        partitioned_grads = self.partition_grad(grad)
        partitioned_precon_grads = self.preconditioned_grad(grad, statistics_unmerged=True)
        w1 = self._hps.beta2
        damping = self._hps.matrix_eps > 0
        rank = len(self._transformed_shape)
        self.updated |= {i}
        for j, (grad_, precon_grad_) in enumerate(zip(partitioned_grads, partitioned_precon_grads)):
            axes = list(range(i)) + list(range(i + 1, rank))
            stat = self.statistics[j*rank + i]
            if w1 < 1:
                stat.mul_(1/w1)

            lgrgt = torch.tensordot(precon_grad_, grad_, [axes, axes])
            if self.debug and not lgrgt.isfinite().all():
                print('lgrgt', lgrgt, '\nprecon_grad', precon_grad_, '\nstat', stat_, '\ngrad', grad_)
                raise ValueError('lgrgt broke')

            if damping:
                I_epsL = torch.eye(stat.size()[0], device=stat.device) - self._hps.matrix_eps * stat
                
                lgrgt_prime = I_epsL.mm(lgrgt)
                stat_prime = I_epsL.mm(stat)
                if self.debug and (not lgrgt_prime.isfinite().all() or not stat_prime.isfinite().all()):
                    print('lgrgt_prime', lgrgt_prime, '\nI_epsL', I_epsL, '\nstat_prime', stat_prime)
                    raise ValueError('lgrgt_prime broke')
                lgrgt = lgrgt_prime
                stat = stat_prime

            # damping
            t_k = - (1 + mf.matrices_norm(lgrgt, 'fro'))
            lgrgt.mul_(1/t_k)
            DX = lgrgt.mm(stat.T)
            if self.debug and not DX.isfinite().all():
                print('DX', DX, '\nlgrgt', lgrgt)
                raise ValueError('DX broke')
            DX = mf.symmetrize(DX)
            self.statistics[j*rank + i] = stat.add(DX)

    @torch.no_grad()
    def compute_preconditioners(self, **kwargs):
        """Compute L^{1/exp} for each statistics matrix L"""
        exp = self.exponent_for_preconditioner()
        for i in list(self.updated):
            stat = self.statistics[i]
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
                    iters=8, tol=1e-4,
                )[0]  # mf.mat_root operates on batches
        self.updated = set()

    @torch.no_grad()
    def partition_grad(self, grad):
        """Partition gradient.
        Args:
          grad: Gradient to partition.
        """
        reshaped_grad = torch.reshape(grad.detach(), self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        return partitioned_grads

class KradagradMM(Shampoo):
    r"""Implements a simple version of Kradagrad-- Optimizer Algorithm.
    Extends the unoptimized official pytorch implementation of Shampoo
    """
    @torch.no_grad()
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.debug = kwargs.get('debug')
        self.initialized = False
        self._step = 0

    @torch.no_grad()
    def init_var_state(self, var, state):
        """Initialize the PyTorch state of for a single variable."""
        # group by exponent for batching
        # we can further improve by sorting by size, but that requires advanced bookkeeping
        prec = KrADmmPreconditioner(var, self.hps, eps_override=1.0, debug=self.debug)

        # original, pared down        
        var = var.detach()
        state[MOMENTUM] = torch.zeros_like(var.data, device=var.device)
        state[PRECONDITIONER] = prec

    # reimplementation for single statistc update
    @torch.no_grad()
    def step(self, closure=None):
        hps = self.hps
        # currently no grafting
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
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.detach()
                state = self.state[p]
                prec = state[PRECONDITIONER]

                # Compute stats and preconditioners
                if self._step % hps.statistics_compute_steps == 0:
                    ix = self._step % len(prec._transformed_shape)
                    prec.add_statistic(grad, ix)
                if self._step % hps.preconditioning_compute_steps == 0:
                    prec.compute_preconditioners()

                # Precondition
                krad_grad = grad
                if self._step >= self.hps.start_preconditioning_step:
                    krad_grad = prec.preconditioned_grad(
                        grad.type(torch.float32)).type_as(grad)

                # Weight decay
                if self.hps.weight_decay != 0.0:
                    krad_grad.add_(p.data, alpha=hps.weight_decay)

                # Momentum and Nesterov momentum, if needed
                state[MOMENTUM].mul_(momentum).add_(krad_grad)

                momentum_update = state[MOMENTUM]
                wd_update = krad_grad

                if hps.nesterov:
                    momentum_update.mul_(momentum).add_(wd_update)

                # Final update
                p.data.add_(momentum_update, alpha=-lr)

        self._step += 1