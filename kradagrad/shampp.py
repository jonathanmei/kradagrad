## Simple Kradagrad++ that extends official unoptimized Shampoo implementation

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

from . import riccati as ricc
from . import positive_matrix_functions as mf
from .third_party.shampoo import (
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
        super().__init__(var, hps, partition=False)
        device = var.device
        shapes = self._partitioner.shapes_for_preconditioners()

        self.debug = debug
        rank = len(self._transformed_shape)
        # for rank==0, we're fine as long as `self.statistics` is empty, which
        # is handled in `super()`
        if rank > 0:
            eps = hps.matrix_eps
            precision = torch.float64 if hps.double else torch.float32
            self.hist = torch.zeros((var.nelement(), hps.preconditioning_compute_steps // hps.statistics_compute_steps), device=device, dtype=precision)
            self.hist_pointer = 0
            self.low_rank = [torch.zeros((var.nelement(), hps.rank), device=device, dtype=precision)]
            self.next_preconditioners = self.preconditioners
            # statistic to the 1/p power, should be cheap to compute in addition to -1/2p
            self.statistics_p = [eps * torch.eye(s[0], device=device, dtype=precision) for s in shapes]

    @torch.no_grad()
    def add_statistics(self, grad):
        """Compute statistics from and store gradients.

        Args:
          grad: Gradient to compute statistics from.
        """
        super().add_statistics(grad)
        self.hist[0][:, self.hist_pointer] = grad.detach().ravel()
        self.hist_pointer += 1

    @torch.no_grad()
    def compute_preconditioners(self, **kwargs):
        """Compute L^{-1/exp} for each statistics matrix L"""
        if not self.statistics: return  # sgd
        # kradagrad
        exp = self.exponent_for_preconditioner()
        for i, stat in enumerate(self.statistics):
            eps = self._hps.matrix_eps
            precon, stat_p = mf.matrix_power_svd(stat, [-1/exp, 2/exp], double=self._hps.double, matrix_eps=eps, eig_eps=eps)
            if self._hps.double:
                stat_p = stat_p.double()
                precon = precon.double()
            self.preconditioners[i] = self.next_preconditioners[i]
            self.next_preconditioners[i] = precon
            self.statistics_p[i] = stat_p
            if self.debug:
                if not precon.isfinite().all():
                    print('stat', stat)
                    raise ValueError('precon broke')
        # assume single block, unpartitioned
        A = ricc.Kron(self.preconditioners)
        C = self.hist[..., :self.hist_pointer]
        self.low_rank = ricc.riccati_kron_solve(A, C, rank=self._hps.low_rank)
        self.hist.zero_()
        self.hist_pointer = 0

    @torch.no_grad()
    def preconditioned_grad(self, grad, **kwargs):
        """Replace running Kradagrad estimate with Shampoo statistic
        """
        kronecker_part = super.preconditioned_grad(grad, **kwargs)
        low_rank_part = (self.low_rank @ (self.low_rank.T @ grad.ravel())).reshape(grad.shape)
        return kronecker_part + low_rank_part


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
    