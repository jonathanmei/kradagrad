## Simple Kradagrad-- that extends official unoptimized Shampoo implementation

from collections import defaultdict
import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

from . import positive_matrix_functions as mf
from .third_party.shampoo import (
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
        If `inverse_exponent_override` is set, use ONS-style update
        """
        return 2 if self._hps.inverse_exponent_override == 0 else 1

    @torch.no_grad()
    def add_statistic(self, grad, i):
        """Compute inverse KrAD statistic from gradients, add to the correct state entries, and
            cache preconditioned gradient (skipping dimension i).

        Args:
          grad: Gradient to compute statistics from.
          i: rank for which to add statistic to
        """
        if not self.statistics: return
        grad_orig = grad
        grad = grad.type(torch.float32)
        partitioned_grads = self.partition_grad(grad)
        partitioned_precon_grads = self.preconditioned_grad(grad, skip=[i], unmerged=True)
        w1 = self._hps.beta2
        damping = self._hps.matrix_eps > 0
        rank = len(self._transformed_shape)
        self.updated |= {i}
        for j, (grad_, precon_grad_) in enumerate(zip(partitioned_grads, partitioned_precon_grads)):
            axes = list(range(i)) + list(range(i + 1, rank))
            stat = self.statistics[j*rank + i]
            precon = self.preconditioners[j*rank + i]
            if w1 < 1:
                stat.mul_(1/w1)

            grgt = torch.tensordot(precon_grad_, precon_grad_, [axes, axes])
            if self.debug and not grgt.isfinite().all():
                print('self.updated', self.updated, '\ngrgt', grgt, '\nprecon_grad', precon_grad_, '\nstat', stat_)
                raise ValueError('grgt broke')

            if damping:
                eps = self._hps.matrix_eps

                stat2eps = stat.mm(stat.T.mul(-eps))
                stat_prime = stat + stat2eps

                if self.debug and not stat_prime.isfinite().all():
                    print('self.updated', self.updated, '\nstat_prime', stat_prime, '\nstat', stat, '\nprecon', precon, '\ngrad', grad_, '\nprecon_grad', precon_grad_)
                    raise ValueError('damping broke')
                stat = stat_prime
            lgrgt = stat.mm(grgt)

            # damping
            t_k = - (1 + mf.matrices_norm(lgrgt, 'fro'))
            lgrgt.mul_(1/t_k)
            DX = lgrgt.mm(stat.T)
            if self.debug and not DX.isfinite().all():
                print('DX', DX, '\nlgrgt', lgrgt)
                raise ValueError('DX broke')
            self.statistics[j*rank + i] = stat + DX
        grad_orig.copy_(
            self._partitioner.merge_partitions(
                partitioned_precon_grads
            ).reshape(self._original_shape).type_as(grad_orig)
        )

    @torch.no_grad()
    def compute_preconditioners(self, **kwargs):
        """Compute L^{1/exp} for each statistics matrix L"""
        if not self.statistics: return  # sgd
        # kradagradmm
        exp = self.exponent_for_preconditioner()
        for i in list(self.updated):
            stat = self.statistics[i]
            try:
                if stat.device.type == 'cpu':
                    self.preconditioners[i] = mf.matrix_power_svd(stat, 1 / exp) if exp > 1 else stat
                else:
                    self.preconditioners[i] = mf.mat_root(
                        stat, exp,
                        self.preconditioners[i],
                        iters=12, tol=1e-4,# debug=True
                    )
            except Exception as err:
                if self.debug:
                    print('stat', stat, '\nmat_root broke')
                raise err
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

    @torch.no_grad()
    def precondition_grad_single(self, grad, ix):
        """Precondition the gradient.

        Args:
          grad: A mostly preconditioned tensor
          ix: the dimension to precondition

        Returns:
          A fully preconditioned gradient.
        """
        if not self.preconditioners: return grad  # sgd
        # precondition gradient
        reshaped_grad = torch.reshape(grad.detach(), self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        preconditioned_partitioned_grads = []
        num_splits = self._partitioner.num_splits()
        for i, grad in enumerate(partitioned_grads):
            mats = self.preconditioners
            preconditioners_for_grad = mats[i * num_splits:(i + 1) * num_splits]
            rank = len(grad.shape)
            orig_type = grad.type()
            precond_grad = grad.type(torch.float32)

            preconditioner = preconditioners_for_grad[ix]
            precond_grad = torch.tensordot(precond_grad, preconditioner, [[ix], [0]])
            precond_grad = precond_grad.moveaxis(-1, ix)

            preconditioned_partitioned_grads.append(precond_grad.type(orig_type))
        merged_grad = self._partitioner.merge_partitions(
            preconditioned_partitioned_grads)
        return torch.reshape(merged_grad, self._original_shape)

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
        var = var.detach()
        prec = KrADmmPreconditioner(var, self.hps, eps_override=1.0, debug=self.debug)

        # original, pared down        
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
                    sh_ = prec._transformed_shape
                    nd_ = len(sh_)
                    if self._step < nd_ ** 2:
                        # simple equal updating schedule:
                        ix = self._step % nd_
                    else:
                        # O(k^0.5) for everything except largest dim
                        max_dim = max(sh_)
                        max_ix = [j for (j, x) in enumerate(sh_) if x == max_dim][0]
                        step_sqrt_floor = int(math.sqrt(self._step))
                        exc = self._step - step_sqrt_floor ** 2
                        ix = exc if exc < nd_ else max_ix

                    # also modifies grad in place to be preconditioned in all dims except ix:
                    prec.add_statistic(grad, ix)
                if self._step % hps.preconditioning_compute_steps == 0:
                    prec.compute_preconditioners()

                # Precondition
                krad_grad = grad
                if self._step >= self.hps.start_preconditioning_step:
                    if self._step % hps.statistics_compute_steps == 0:
                        krad_grad = prec.precondition_grad_single(grad, ix)
                    else:
                        krad_grad = prec.preconditioned_grad(grad)

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
