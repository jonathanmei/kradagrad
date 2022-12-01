## Simple Kradagrad++ that extends official unoptimized Shampoo implementation

import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer
import torch.multiprocessing as mp

import matrix_root as mr
from shampoo import (
    Shampoo, Preconditioner,
    STEP, MOMENTUM, PRECONDITIONER, GRAFT,
    LayerwiseGrafting, AdamGraft, AdagradGraft, SGDGraft, Graft
)


class KrADPreconditioner(Preconditioner):
    @torch.no_grad()
    def __init__(self, var, hps):
        super(KrADPreconditioner, self).__init__(var, hps)

    @torch.no_grad()
    def add_statistics(self, grad):
        """Compute inverse KrAD statistics from gradients and add to the correct state entries.

        Args:
          grad: Gradient to compute statistics from.
        """
        if not self.statistics: return
        reshaped_grad = torch.reshape(grad.detach(), self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        w1 = self._hps.beta2
        rank = len(self._transformed_shape)
        for j, grad in enumerate(partitioned_grads):
            for i in range(rank):
                axes = list(range(i)) + list(range(i + 1, rank))
                stat = self.statistics[j*rank + i]
                if w1 < 1:
                    stat.mul_(1/w1)

                stat_2_eps = stat.mul(-self._hps.diagonal_eps).mm(stat.T)
                if not stat_2_eps.isfinite().all():
                    print('stat2eps', stat_2_eps)
                    print('stat', stat)
                    print('-eps*stat', stat.mul(-self._hps.diagonal_eps))
                    raise ValueError('stat2eps broke')
                stat.add_(stat_2_eps)

                ggt = torch.tensordot(grad, grad, [axes, axes])
                if not ggt.isfinite().all():
                    print('ggt', ggt)
                    print('grad', grad)
                    raise ValueError('ggt broke')

                geom_fact = ggt.mm(stat.T)

                t_k = - (1 + mr._matrices_norm(geom_fact, 'fro'))
                geom_fact.mul_(1/t_k)
                DX = stat.mm(geom_fact)
                self.statistics[j*rank + i].add_(DX)

                # We could do another iteration of binomial for inverse here...
                #X = self.statistics[j*rank + i]
                #self.statistics[j*rank + i] = X + geom_fact.mm(X).mm(geom_fact.T)

    @torch.no_grad()
    def compute_preconditioners(self, **kwargs):
        """Compute L^{1/exp} for each statistics matrix L"""
        exp = self.exponent_for_preconditioner()
        eps = self._hps.matrix_eps
        for i, stat in enumerate(self.statistics):
            if stat.device.type == 'cpu':
                try:
                    self.preconditioners[i] = mr.matrix_power_svd(stat, 1 / exp) if exp > 1 else stat
                except Exception as err:
                    print('stat', stat)
                    raise err
            else:
                self.preconditioners[i] = mr.mat_root(
                    stat[None, ...], exp,
                    self.preconditioners[i][None, ...],
                    iters=10, tol=1e-4, inner_iters=5, inner_tol=1e-3
                )[0]  # mr.mat_root operates on batches
            #if self._hps.beta2 < 1:
            #    self.preconditioners[i] *= (1-self._hps.beta2)**(1/exp)


class KradagradPP(Shampoo):
    r"""Implements a simple version of Kradagrad++ Optimizer Algorithm.
    Extends the unoptimized official pytorch implementation of Shampoo
    """
    @torch.no_grad()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def init_var_state(self, var, state):
        """Initialize the PyTorch state of for a single variable."""
        var = var.detach()
        state[STEP] = 0
        state[MOMENTUM] = torch.zeros_like(var.data, device=var.device)
        state[PRECONDITIONER] = KrADPreconditioner(var, self.hps)
        if self.hps.graft_type == LayerwiseGrafting.ADAM:
            state[GRAFT] = AdamGraft(self.hps, var)
        else:
            state[GRAFT] = Graft(self.hps, var)
