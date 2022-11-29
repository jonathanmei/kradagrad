## Simple Kradagrad++ that extends official unoptimized Shampoo implementation

import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

import matrix_root as mr
from shampoo import (
    Shampoo, Preconditioner,
    STEP, MOMENTUM, PRECONDITIONER, GRAFT,
    LayerwiseGrafting, AdagradGraft, SGDGraft, Graft
)


class KrADPreconditioner(Preconditioner):
    @torch.no_grad()
    def __init__(self, var, hps):
        super().__init__(var, hps)

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
        w2 = 1.0 if w1 == 1.0 else (1.0 - w1)
        rank = len(self._transformed_shape)
        for j, grad in enumerate(partitioned_grads):
            for i in range(rank):
                axes = list(range(i)) + list(range(i + 1, rank))
                stat = torch.tensordot(grad, grad, [axes, axes])
                stat = stat.mm(self.statistics[j*rank + i].T)
                t_k_inv = -1.0 / (1.0 + mr._matrices_norm(stat, 'fro'))
                stat = self.statistics[j*rank + i].mm(stat)
                stat.mul_(t_k_inv)
                self.statistics[j*rank + i].mul_(w1).add_(stat, alpha=w2)

    @torch.no_grad()
    def compute_preconditioners(self, **kwargs):
        """Compute L^{1/exp} for each statistics matrix L"""
        exp = self.exponent_for_preconditioner()
        eps = self._hps.matrix_eps
        for i, stat in enumerate(self.statistics):
            if stat.device.type == 'cpu':
                self.preconditioners[i] = mr.matrix_power_svd(stat, 1 / exp) if exp > 1 else stat
                if kwargs.get('step', 0) > 32*280 and kwargs.get('step', 1) % (32*90) == 0:
                    print(i, kwargs['step'], mr._matrices_norm(stat[None, ...])[0])
            else:
                self.preconditioners[i] = mr.mat_root(
                    stat[None, ...], exp,
                    self.preconditioners[i][None, ...],
                    iters=12, tol=1e-4, inner_iters=8, inner_tol=1e-3
                )[0]  # mr.mat_root operates on batches

    @torch.no_grad()
    def preconditioned_grad(self, grad):
        return super().preconditioned_grad(grad)


class KradagradPP(Shampoo):
    r"""Implements a simple version of Kradagrad++ Optimizer Algorithm.
    Extends the unoptimized official pytorch implementation of Shampoo
    """
    @torch.no_grad()
    def init_var_state(self, var, state):
        """Initialize the PyTorch state of for a single variable."""
        state[STEP] = 0
        state[MOMENTUM] = torch.zeros_like(var.data, device=var.device)
        state[PRECONDITIONER] = KrADPreconditioner(var, self.hps)
        if self.hps.graft_type == LayerwiseGrafting.ADAGRAD:
            state[GRAFT] = AdagradGraft(self.hps, var)
        elif self.hps.graft_type == LayerwiseGrafting.SGD:
            state[GRAFT] = SGDGraft(self.hps, var)
        else:
            state[GRAFT] = Graft(self.hps, var)

    @torch.no_grad()
    def step(self, closure=None):
        super(KradagradPP, self).step(closure=closure)

