## Simple Kradagrad++ that extends official unoptimized Shampoo implementation

from collections import defaultdict
import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

from .. import positive_matrix_functions as mf
from ..third_party.shampoo import (
    Shampoo, Preconditioner,
    STEP, MOMENTUM, PRECONDITIONER, GRAFT,
    LayerwiseGrafting, AdamGraft, AdagradGraft, SGDGraft, Graft
)


class KrADapooPreconditioner(Preconditioner):

    @torch.no_grad()
    def __init__(self, var, hps, eps_override=1.0, debug=False):
        """
        debug mode print tensor values likely to be non-finite, raises exceptions
        eps_override changes initialization of preconditioner diagonal (usually 1.0)
        """
        super().__init__(var, hps, eps_override=eps_override)
        device = var.device
        precision = torch.float64 if hps.double else torch.float32
        shapes = self._partitioner.shapes_for_preconditioners()

        self.debug = debug
        rank = len(self._transformed_shape)
        if rank < 1:
            self.sham_stats = []
        else:
            eps = eps_override if eps_override is not None else 1/hps.matrix_eps
            self.sham_stats = [eps * torch.eye(s[0], device=device, dtype=precision) for s in shapes]
        
    @torch.no_grad()
    def exponent_for_preconditioner(self):
        """Returns exponent to use for pth root M^{1/p}.
        If `inverse_exponent_override` is set, use ONS-style update
        """
        if self._hps.inverse_exponent_override > 0:
            return len(self._transformed_shape)
        return 2 * len(self._transformed_shape)

    @torch.no_grad()
    def add_statistics(self, grad):
        """Compute inverse KrAD statistics from gradients and add to the correct state entries.

        Args:
          grad: Gradient to compute statistics from.
        """
        if not self.statistics: return  # sgd
        # kradagrad
        partitioned_grads = self.partition_grad(grad)
        w1 = self._hps.beta2
        damping = self._hps.matrix_eps > 0
        rank = len(self._transformed_shape)
        for j, grad in enumerate(partitioned_grads):
            if self._hps.double:
                grad = grad.double()
            for i in range(rank):
                axes = list(range(i)) + list(range(i + 1, rank))
                stat = self.statistics[j*rank + i]
                sham_stat = self.sham_stats[j*rank + i]
                if w1 < 1:
                    stat.mul_(1/w1)
                    sham_stat.mul_(1/w1)

                if damping:
                    stat_2_eps = stat.mul(-self._hps.matrix_eps).mm(stat.T)
                    if self.debug and not stat_2_eps.isfinite().all():
                        print('stat2eps', stat_2_eps, '\nstat', stat)
                        raise ValueError('stat2eps broke')
                    self.statistics[j*rank + i].add_(stat_2_eps)
                    stat = self.statistics[j*rank + i]

                ggt = torch.tensordot(grad, grad, [axes, axes])
                sham_stat.add_(ggt.diag())
                
                if self.debug and not ggt.isfinite().all():
                    print('ggt', ggt, '\ngrad', grad)
                    raise ValueError('ggt broke')

                ggtl = ggt.mm(stat.type_as(ggt).T)

                t_k = -(1 + mf.matrices_norm(ggtl, 'fro'))
                ggtl.mul_(1/t_k)
                lggtl = stat.type_as(ggtl).mm(ggtl)
                self.statistics[j*rank + i].add_(lggtl)


    @torch.no_grad()
    def compute_preconditioners(self, **kwargs):
        """Compute L^{1/exp} for each statistics matrix L"""
        if not self.statistics: return  # sgd
        # kradagrad
        exp = self.exponent_for_preconditioner()
        hps = self._hps
        for i, (stat, sham_stat) in enumerate(zip(self.statistics, self.sham_stats)):
            sham_diag_rt = torch.pow(sham_stat, (exp-1)/(2*exp))
            precon = sham_diag_rt * stat * sham_diag_rt
                ## about the same as type0 but slower:
                # precon = mf.matrix_power_svd(stat/2 + (1/sham_stat.diag()).diag()/2, 1/exp)
                ## bad and slow:
                # sham_rt = mf.matrix_power_svd(sham_stat, (exp-1)/(2*exp))
                # precon = sham_rt @ stat @ sham_rt.T
                ## works similar to krad, just slow:
                # sham_diag_inv_rt = torch.pow(sham_stat.diag(), -1/(2*exp))
                # sham_diag_rt = torch.pow(sham_stat.diag(), 1/(2*exp))
                # precon = sham_diag_inv_rt * mf.matrix_power_svd(
                #     sham_diag_rt * mf.matrix_power_svd(stat, 1/exp) * sham_diag_rt, 0.5
                # ) * sham_diag_inv_rt
                ## fast but doesn't work
                # stat_diag_inv_rt = torch.pow(stat.diag(), -(exp-1)/(2*exp))
                # precon = stat_diag_inv_rt * sham_stat * stat_diag_inv_rt
                ##
                #stat_R = torch.linalg.cholesky(stat, upper=True)
                #V = torch.linalg.solve_triangular(
                #    stat_R.T, torch.linalg.solve_triangular(
                #        stat_R, sham_stat, upper=True, left=False
                #    ), upper=False, left=True,
                #)
                #precon = torch.linalg.solve_triangular(
                #    stat_R, torch.linalg.solve_triangular(
                #        stat_R.T, mf.matrix_power_svd(V, -(exp+1)/(2*exp), eps=math.sqrt(self._hps.matrix_eps)), upper=False, left=False, 
                #    ), upper = True, left=True,
                #)
                ## slow and diverges
                #stat_rt_m2 = mf.matrix_power_svd(stat, -0.5)
                #midd = mf.symmetrize(stat_rt_m2 @ sham_stat @ stat_rt_m2.T)
                #precon = mf.symmetrize(stat_rt_m2 @ mf.matrix_power_svd(midd, -(exp + 1)/(2*exp), eps=math.sqrt(hps.matrix_eps)) @ stat_rt_m2.T)
            self.preconditioners[i] = precon
            if self.debug:
                if not precon.isfinite().all():
                    print('stat', stat)
                    raise ValueError('precon broke')


    @torch.no_grad()
    def partition_grad(self, grad):
        """Partition gradient.
        Args:
          grad: Gradient to partition.
        """
        reshaped_grad = torch.reshape(grad.detach(), self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        return partitioned_grads

class Kradapoo(Shampoo):
    r"""Implements a simple version of Kradagrad++ Optimizer Algorithm.
    Extends the unoptimized official pytorch implementation of Shampoo
    """
    @torch.no_grad()
    def __init__(self, params, **kwargs):
        self.debug = kwargs.get('debug', False)
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def init_var_state(self, var, state):
        """Initialize the PyTorch state of for a single variable."""
        prec = KrADapooPreconditioner(var, self.hps, eps_override=1/self.hps.matrix_eps, debug=self.debug)

        # original, pared down        
        var = var.detach()
        state[MOMENTUM] = torch.zeros_like(var.data, device=var.device)
        state[PRECONDITIONER] = prec
        state[STEP] = 0
        if self.hps.graft_type == LayerwiseGrafting.ADAM:
            state[GRAFT] = AdamGraft(self.hps, var)
        else:
            state[GRAFT] = Graft(self.hps, var)
