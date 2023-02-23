## Simple Kradagrad++ that extends official unoptimized Shampoo implementation

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

from .. import positive_matrix_functions as mf
from ..third_party.shampoo import (
    Shampoo, Preconditioner,
    STEP, MOMENTUM, PRECONDITIONER, GRAFT,
    LayerwiseGrafting, AdamGraft, Graft
)


class ShamppPreconditioner(Preconditioner):

    @torch.no_grad()
    def __init__(self, var, hps, debug=False):
        """
        debug mode print tensor values likely to be non-finite, raises exceptions
        eps_override changes initialization of preconditioner diagonal (usually 1.0)
        """
        super().__init__(var, hps)
        device = var.device
        shapes = self._partitioner.shapes_for_preconditioners()

        self.debug = debug
        rank = len(self._transformed_shape)
        if rank < 1:
            self.krad_stats = []
        else:
            self.krad_stats = [(1/hps.matrix_eps) * torch.eye(s[0], device=device) for s in shapes]

    @torch.no_grad()
    def add_statistics(self, grad):
        """Compute inverse KrAD statistics from gradients and add to the correct state entries.

        Args:
          grad: Gradient to compute statistics from.
        """
        if not self.statistics: return  # sgd
        # kradagrad
        reshaped_grad = torch.reshape(grad.detach(), self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        w1 = self._hps.beta2
        w2 = 1.0 if w1 == 1.0 else (1.0 - w1)
        rank = len(self._transformed_shape)
        for j, grad in enumerate(partitioned_grads):
            if self._hps.double:
                grad = grad.double()
            if self._hps.bf16:
                grad = grad.bfloat16()
            for i in range(rank):
                axes = list(range(i)) + list(range(i + 1, rank))
                krad_stat = self.krad_stats[j*rank + i]
                if w1 < 1:
                    krad_stat.mul_(1/w1)

                ggt = torch.tensordot(grad, grad, [axes, axes])
                if self.debug and not ggt.isfinite().all():
                    print('ggt', ggt, '\ngrad', grad)
                    raise ValueError('ggt broke')
                self.statistics[j*rank+i].mul_(w1).add_(ggt, alpha=w2)

                ggtl = ggt.mm(krad_stat.type_as(ggt).T)

                t_k = -(1 + mf.matrices_norm(ggtl, 'tr'))
                ggtl.mul_(1/t_k)
                lggtl = krad_stat.type_as(ggtl).mm(ggtl)
                krad_stat.add_(lggtl)

    @torch.no_grad()
    def compute_preconditioners(self, **kwargs):
        """Compute L^{-1/exp} for each statistics matrix L"""
        if not self.statistics: return  # sgd
        # kradagrad
        exp = self.exponent_for_preconditioner()
        for i, (stat, krad) in enumerate(zip(self.statistics, self.krad_stats)):
            if self._hps.double:
                krad = krad.double()
            eps = self._hps.matrix_eps
            precon = mf.matrix_power_svd(krad, 1/exp, double=self._hps.double, matrix_eps=eps, eig_eps=eps)
            if self._hps.double:
                precon = precon.double()
            self.preconditioners[i] = precon
            if self.debug:
                if not precon.isfinite().all():
                    print('stat', stat)
                    print('krad', krad)
                    raise ValueError('precon broke')

    @torch.no_grad()
    def update_krad_from_shampoo(self, **kwargs):
        """Replace running Kradagrad estimate with Shampoo statistic
        """
        if not self.statistics: return  # sgd
        eps = self._hps.matrix_eps
        for i, stat in enumerate(self.statistics):
            self.krad_stats[i] = mf.matrix_power_svd(stat, -1, double=self._hps.double, matrix_eps=eps, eig_eps=eps)


class Shampp(Shampoo):
    r"""Implements a simple version of Shampoo++ Optimizer Algorithm.
    Extends the unoptimized official pytorch implementation of Shampoo
    more frequent krad updates in between more "accurate" shampoo updates
    """
    @torch.no_grad()
    def __init__(self, params, **kwargs):
        self.debug = kwargs.get('debug', False)
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def init_var_state(self, var, state):
        """Initialize the PyTorch state of for a single variable."""
        prec = ShamppPreconditioner(var, self.hps, debug=self.debug)

        # original, pared down        
        var = var.detach()
        state[MOMENTUM] = torch.zeros_like(var.data, device=var.device)
        state[PRECONDITIONER] = prec
        state[STEP] = 0
        state[GRAFT] = Graft(self.hps, var)

    @torch.no_grad()
    def step(self, closure=None):
        hps = self.hps
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError('Shampoo does not support sparse yet')
                state = self.state[p]
                if not state:
                    self.init_var_state(p, state)
                state[STEP] += 1

                preconditioner = state[PRECONDITIONER]
                graft = state[GRAFT]

                # Gather statistics, compute preconditioners
            
                graft.add_statistics(grad)
            
                if state[STEP] % hps.statistics_compute_steps == 0:
                    preconditioner.add_statistics(grad)
                if state[STEP] % hps.replace_preconditioner_steps == 0:
                    preconditioner.update_krad_from_shampoo()
                if state[STEP] % hps.preconditioning_compute_steps == 0:
                    preconditioner.compute_preconditioners()
                # Precondition gradients
                graft_grad = graft.precondition_gradient(grad)
                shampoo_grad = grad
                if state[STEP] >= self.hps.start_preconditioning_step:
                    shampoo_grad = preconditioner.preconditioned_grad(grad)

                # Grafting
                graft_norm = torch.norm(graft_grad)
                shampoo_norm = torch.norm(shampoo_grad)
                shampoo_grad.mul_(graft_norm / (shampoo_norm + 1e-16))

                # Weight decay
                if self.hps.weight_decay != 0.0:
                    shampoo_grad.add_(p.data, alpha=self.hps.weight_decay)
                    graft_grad.add_(p.data, alpha=self.hps.weight_decay)

                # Momentum and Nesterov momentum, if needed
                state[MOMENTUM].mul_(group['momentum']).add_(shampoo_grad)
                graft_momentum = graft.update_momentum(grad, group['momentum'])

                if state[STEP] >= self.hps.start_preconditioning_step:
                    momentum_update = state[MOMENTUM]
                    wd_update = shampoo_grad
                else:
                    momentum_update = graft_momentum
                    wd_update = graft_grad

                if hps.nesterov:
                    momentum_update.mul_(group['momentum']).add_(wd_update)

                # Final update
                p.data.add_(momentum_update, alpha=-lr)