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


class RadPreconditioner(Preconditioner):

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
        # for rank==0, we're fine as long as `self.statistics` is empty, which
        # is handled in `super()`
        eps = hps.matrix_eps
        self.statistic = torch.zeros_like(var)
        self.preconditioner = torch.ones_like(var)
        self.next_preconditioner = torch.zeros_like(var)
        precision = torch.float64 if hps.double else torch.float32
        self.hist = torch.zeros((var.nelement(), hps.preconditioning_compute_steps), device=device, dtype=precision)
        self.hist_pointer = 0
        self.low_rank = torch.zeros((var.nelement(), hps.low_rank), device=device, dtype=precision)

    @torch.no_grad()
    def add_statistics(self, grad):
        """Compute statistics from and store gradients.

        Args:
          grad: Gradient to compute statistics from.
        """
        self.statistic += grad ** 2
        self.hist[:, self.hist_pointer] = grad.detach().ravel()
        self.hist_pointer += 1

    def compute_preconditioners(self, **kwargs):
        """Compute L^{-1/exp} for each statistics matrix L"""
        # single block, unpartitioned due to using IdentityPartitioner
        self.preconditioner = self.next_preconditioner
        self.next_preconditioner = (self.statistic + self._hps.diagonal_eps) ** -0.5
        A = self.preconditioner.ravel()[:, None]
        C = self.hist[..., :self.hist_pointer]
        #Z = ricc.riccati_diag_solve(A, C, rank=self._hps.low_rank)
        Z = ricc.diag_riccati_lr_greedy_nm(A, C, rank=self._hps.low_rank)
        Aih = 1 / (A + self._hps.diagonal_eps)
        woodbury_mid_sqrt = mf.matrix_power_svd(
            torch.eye(Z.shape[-1], device=Z.device, dtype=Z.dtype) \
                + Z.T @ (Aih * Z), -1/2, eig_eps=self._hps.matrix_eps, double=self._hps.double
            )
        self.low_rank = Aih * Z @ woodbury_mid_sqrt
        self.hist.zero_()
        self.hist_pointer = 0

    @torch.no_grad()
    def preconditioned_grad(self, grad, **kwargs):
        """Replace running Kradagrad estimate with Shampoo statistic
        """
        grad_dtype = grad.dtype
        precision = torch.float64 if self._hps.double else torch.float32
        diag_part = self.preconditioner.reshape(grad.shape) * grad
        low_rank_part = (self.low_rank @ (self.low_rank.T @ grad.ravel().type(precision))).reshape(grad.shape)
        return (diag_part - low_rank_part).type(grad_dtype)


class Radagrad(Shampoo):
    r"""Implements a simple version of Radagrad Optimizer Algorithm.
    Extends the unoptimized official pytorch implementation of Shampoo
    """
    @torch.no_grad()
    def __init__(self, params, **kwargs):
        self.debug = kwargs.get('debug', False)
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def init_var_state(self, var, state):
        """Initialize the PyTorch state of for a single variable."""
        prec = RadPreconditioner(var, self.hps, debug=self.debug)

        # original, pared down        
        var = var.detach()
        state[MOMENTUM] = torch.zeros_like(var.data, device=var.device)
        state[PRECONDITIONER] = prec
        state[STEP] = 0
        state[GRAFT] = Graft(self.hps, var)
    