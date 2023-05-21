## Simple Kradagrad++ that extends official unoptimized Shampoo implementation


import torch

from .math_utils import positive_matrix_functions as mf
from .third_party.shampoo import (
    Shampoo, Preconditioner,
    STEP, MOMENTUM, PRECONDITIONER, GRAFT,
    Graft
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
            self.krad_stats = []
        else:
            self.krad_stats = [(1/hps.matrix_eps) * torch.eye(s[0], device=device, dtype=precision) for s in shapes]

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

                ggt = torch.tensordot(grad, grad, [axes, axes])
                if self.debug and not ggt.isfinite().all():
                    print('ggt', ggt, '\ngrad', grad)
                    raise ValueError('ggt broke')
                self.statistics[j*rank+i].mul_(w1).add_(ggt, alpha=w2)


    @torch.no_grad()
    def compute_preconditioners(self, **kwargs):
        """Compute L^{-1/exp} for each statistics matrix L"""
        if not self.statistics: return  # sgd
        # kradagrad
        exp = self.exponent_for_preconditioner()
        w1 = self._hps.beta2
        for i, (stat, krad) in enumerate(zip(self.statistics, self.krad_stats)):
            if w1 < 1:
                krad.mul_(1/w1)
            ggtl = stat.mm(krad.type_as(stat).T)

            t_k = -(1 + mf.matrices_norm(ggtl, 'tr'))
            ggtl.mul_(1/t_k)
            lggtl = stat.type_as(ggtl).mm(ggtl)
            krad.add_(lggtl)
            stat.zero_()
            precon = mf.matrix_power_svd(krad, 1/exp, double=self._hps.double)
            self.preconditioners[i] = precon
            if self.debug:
                if not precon.isfinite().all():
                    print('stat', stat)
                    print('krad', krad)
                    raise ValueError('precon broke')


class Kradelay(Shampoo):
    r"""Implements a simple version of Kradelay Optimizer Algorithm.
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
