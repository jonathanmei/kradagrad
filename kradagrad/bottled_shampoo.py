## Bottlenecked Shampoo that extends official unoptimized Shampoo implementation
# Uses the smallest k eigenvalues of fully accumulated GGT

import torch

from . import math_utils as mu
from . import positive_matrix_functions as mf
from .third_party.shampoo import (
    Shampoo, Preconditioner,
    STEP, MOMENTUM, PRECONDITIONER, GRAFT,
    Graft, shampoo
)


class BSPreconditioner(Preconditioner):

    @torch.no_grad()
    def __init__(self, var, hps, debug=False, **unused):
        """
        debug mode print tensor values likely to be non-finite, raises exceptions
        """
        self._hps = hps
        self._original_shape = var.shape
        self._transformed_shape = var.shape
        if hps.best_effort_shape_interpretation:
            self._transformed_shape = shampoo._merge_small_dims(
                self._original_shape, hps.block_size)

        reshaped_var = torch.reshape(var.detach(), self._transformed_shape)
        self._partitioner = shampoo.IdentityPartitioner(reshaped_var)
        shapes = self._partitioner.shapes_for_preconditioners()
        rank = len(self._transformed_shape)
        device = var.device

        n = torch.Tensor([var.numel()])
        target_rank = min(hps.low_rank, mu.roundup(n / hps.low_rank_factor, 1))

        if rank < 1:
            self.rs = []  # compute proportional ranks for each dim of tensor
            self.ss = []  # store eigvals^{-1/p}
            self.statistics = []  # accum G G^T across all time
            self.statistics_buffer = []  # accum G G^T for interval between updates
            self.Vs = []  # store eigvectors
        else:
            precision = torch.float64 if hps.double else torch.float32
            logn = torch.log(n)
            self.rs = [mu.roundup(target_rank ** (torch.log(torch.Tensor([s[0]])) / logn), 1) for s in shapes]
            self.ss = [torch.zeros((r,), device=device, dtype=precision) for r in self.rs]
            self.statistics = [torch.zeros(s, device=device, dtype=precision) for s in shapes]
            self.statistics_buffer = [torch.zeros(s, device=device, dtype=precision) for s in shapes]
            self.Vs = [torch.randn((s[0], r), device=device, dtype=precision) for s,r in zip(shapes, self.rs)]

        self.debug = debug

    @torch.no_grad()
    def add_statistics(self, grad):
        """Compute statistics from and store gradients.

        Args:
          grad: Gradient to compute statistics from.
        """
        if not self.ss: return  # sgd
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

                ggt = torch.tensordot(grad, grad, [axes, axes])
                if self.debug and not ggt.isfinite().all():
                    print('ggt', ggt, '\ngrad', grad)
                    raise ValueError('ggt broke')
                self.statistics_buffer[j*rank+i].mul_(w1).add_(ggt, alpha=w2)

    def compute_preconditioners(self, **kwargs):
        """Compute L^{-1/exp} for each statistics matrix L"""
        # single block, unpartitioned due to using IdentityPartitioner
        if not self.ss: return  # sgd
        exp = self.exponent_for_preconditioner()
        eps = self._hps.matrix_eps
        w1 = self._hps.beta2
        w2 = 1.0 if w1 == 1.0 else (1.0 - w1)
        for i, (r, stat, stat_accum, V) in enumerate(zip(self.rs, self.statistics, self.statistics_buffer, self.Vs)):
            stat.mul_(w1).add_(stat_accum, alpha=w2)
            #try:
            #    # will fail if r is too large relative to shape of stat_new, among other reasons
            #    s_new, V_new = torch.lobpcg(stat, k=r, X=V, largest=False)
            #except:
            #    s_new, V_new = torch.linalg.eigh(stat)
            #    s_new, V_new = s_new[:r], V_new[:, :r]
            s_new, V_new = torch.linalg.eigh(stat)
            s_new, V_new = s_new[:r], V_new[:, :r]
            s_new = torch.maximum(s_new, torch.Tensor([eps]).to(s_new.device))
            self.Vs[i] = V_new
            # precompute power
            self.ss[i] = s_new ** (-1/exp)
        

    @torch.no_grad()
    def preconditioned_grad(self, grad, **kwargs):
        """
        Low rank stuff
        """
        exp = self.exponent_for_preconditioner()
        eps = self._hps.matrix_eps
        grad_dtype = grad.dtype
        precision = torch.float64 if self._hps.double else torch.float32
        precond_grad = grad.type(precision)
        for j, V in enumerate(self.Vs):
            pg_ = torch.tensordot(precond_grad, V.type(precision), [[0], [0]])
            if 'debug' in self.__dict__ and self.debug and not pg_.isfinite().all():
                print(j, 'V', V, '\nprecond_grad (before)', precond_grad, '\ngrad (after)', pg_)
                raise ValueError('precond_grad broke in 1st matmul')
            precond_grad = pg_
        for j, s in enumerate(self.ss):
            slc = (None, ) * j + (slice(None), ) + (None, ) * (len(self.ss) - 1 - j)
            precond_grad *= s[slc]
        for j, V in enumerate(self.Vs):
            pg_ = torch.tensordot(precond_grad, V.type(precision), [[0], [1]])
            if 'debug' in self.__dict__ and self.debug and not pg_.isfinite().all():
                print(j, 'V', V, '\nprecond_grad (before)', precond_grad, '\ngrad (after)', pg_)
                raise ValueError('precond_grad broke in 2nd matmul')
            precond_grad = pg_
        precond_grad.add_(grad.type(precision), alpha=eps ** (-1 / exp))

        return precond_grad.type(grad_dtype)


class Bottled(Shampoo):
    r"""Implements a simple version of Bottled Shampoo Optimizer Algorithm.
    Extends the unoptimized official pytorch implementation of Shampoo
    """
    @torch.no_grad()
    def __init__(self, params, **kwargs):
        self.debug = kwargs.get('debug', False)
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def init_var_state(self, var, state):
        """Initialize the PyTorch state of for a single variable."""
        prec = BSPreconditioner(var, self.hps, debug=self.debug)

        # original, pared down        
        var = var.detach()
        state[MOMENTUM] = torch.zeros_like(var.data, device=var.device)
        state[PRECONDITIONER] = prec
        state[STEP] = 0
        state[GRAFT] = Graft(self.hps, var)
    