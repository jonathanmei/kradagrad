## Simple Kradagrad++ that extends official unoptimized Shampoo implementation

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

from . import positive_matrix_functions as mf
from .third_party.shampoo import (
    Shampoo, Preconditioner,
    STEP, MOMENTUM, PRECONDITIONER, GRAFT,
    LayerwiseGrafting, AdamGraft, Graft
)


class KradelayPreconditioner(Preconditioner):

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
        precision = torch.float64 if hps.double else torch.float32
        if rank < 1:
            self.hist = []
        else:
            self.hist = [torch.zeros(var.numel(), device=device, dtype=precision)] * hps.preconditioning_comute_steps
            self._hist_pointer = 0

    @torch.no_grad()
    def add_statistics(self, grad):
        """Compute inverse KrAD statistics from gradients and add to the correct state entries.

        Args:
          grad: Gradient to compute statistics from.
        """
        if not self.statistics: return  # sgd
        # kradagrad
        self.hist[self._hist_pointer] = grad.detach().ravel()
        self._hist_pointer += 1


    @torch.no_grad()
    def compute_preconditioners(self, **kwargs):
        """Compute L^{-1/exp} for each statistics matrix L"""
        if not self.statistics: return  # sgd
        # kradagrad
        exp = self.exponent_for_preconditioner()
        #preprocess history first
        h_len = len(self.hist)
        hist = torch.stack(self.hist, 0)
        w1 = self._hps.beta2
        rank = len(self._transformed_shape)
        partitioned_hists = [self._paritioner.partition(h_.reshape(self._transformed_shape)) for h_ in hist]
        for i in range(rank):
            axes = list(range(i)) + list(range(i + 1, rank))
            def contr(tens1, tens2):
                return torch.tensordot(tens1, tens2, [axes, axes])
            def mul_i(mat, tens):
                return torch.tensordot(mat, tens, [[1], [i]]).moveaxis(0, i)
            QG_s = [[mul_i(self.statistics[j*rank+i], hist_[j]) for j in range(len(hist_))] for hist_ in partitioned_hists]
            # TODO
            
            
            QG = mul_j_batch(hist)
            # L LT = I+GT Q G
            L = torch.linalg.cholesky(torch.eye(h_len, device=hist.device, dtype=hist.dtype) + hist.T @ QG)
            # U = Q G L^{-T}
            V = torch.solve_triangular(L, hist, upper=False, left=False)
            U = self.preconditioned_grad_batch(V)
            reshaped_Us = [torch.reshape(u_, self._transformed_shape) for u_ in U.T]
            partitioned_Us = [self._partitioner.partition(u_) for u_ in reshaped_Us]
            for i in range(partitioned_Us[0])
            stat_u = torch.stack([mul_j(stat, u_).ravel() for u_ in u_v_mats], -1)
                for k in range(len(partitioned_Us)):
                    self.statistics[j*rank+i] += contr(partitioned_Us[k][j*rank+i], partitioned_Us[k][i*rank+j])
        for i, stat in enumerate(self.statistics):
            self.preconditioners[i] = mf.matrix_power_svd(stat, 1/exp)
        # reset history:
        for h_ in self.hist:
            h_.zero_()
        self._hist_pointer = 0


class Kradelay(Shampoo):
    r"""Implements a simple version of Shampoo++ Optimizer Algorithm.
    Extends the unoptimized official pytorch implementation of Shampoo
    """
    @torch.no_grad()
    def __init__(self, params, **kwargs):
        self.debug = kwargs.get('debug', False)
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def init_var_state(self, var, state):
        """Initialize the PyTorch state of for a single variable."""
        prec = KradelayPreconditioner(var, self.hps, debug=self.debug)

        # original, pared down        
        var = var.detach()
        state[MOMENTUM] = torch.zeros_like(var.data, device=var.device)
        state[PRECONDITIONER] = prec
        state[STEP] = 0
        state[GRAFT] = Graft(self.hps, var)
