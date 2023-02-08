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


class KradaKryPreconditioner(Preconditioner):

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
            self.grad = []
            self.precon_invs = []
        else:
            eps = hps.matrix_eps
            eps_rt = eps **(1/self.exponent_for_preconditioner())
            precision = torch.float64 if hps.double else torch.float32
            self.grad = [torch.zeros(var.numel(), device=device, dtype=precision)]
            self.precon_invs = [eps_rt * torch.eye(s[0], device=device, dtype=precision) for s in shapes]

    @torch.no_grad()
    def add_statistics(self, grad):
        """Compute inverse KrAD statistics from gradients and add to the correct state entries.

        Args:
          grad: Gradient to compute statistics from.
        """
        if not self.grad: return  # sgd
        self.grad[0] = grad.detach()

    @torch.no_grad()
    def compute_preconditioners(self, **kwargs):
        """Compute L^{1/exp} for each statistics matrix L. In this case also updates the statistics"""
        if not self.statistics: return  # sgd
        # kradagrad
        reshaped_grad = torch.reshape(self.grad[0], self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        rank = len(self._transformed_shape)
        for j, grad in enumerate(partitioned_grads):
            if self._hps.double:
                grad = grad.double()
            if self._hps.bf16:
                grad = grad.bfloat16()

            # rank of krylov subspace
            exp = self.exponent_for_preconditioner()
            kry = min(4 * exp, grad.numel() // 4)
            for i in range(rank):
                stat = self.statistics[j*rank + i]
                precon_inv = self.precon_invs[j*rank+i]
                axes = list(range(i)) + list(range(i + 1, rank))
                def contr(tens1, tens2):
                    return torch.tensordot(tens1, tens2, [axes, axes])
                def mul_i(mat, tens):
                    return torch.tensordot(mat, tens, [[1], [i]]).moveaxis(0, i)
                ### Krylov subspace method to find f(A+bb^T) - f(A) for Positive Definite A
                if self._hps.kry_qr:
                    ## QR decomp to build basis
                    # Ref: Alg 1 in https://arxiv.org/pdf/2008.11501.pdf
                    u_v = [0] * kry
                    u_v[0] = grad 
                    for kk in range(1, kry):
                        u_v[kk] = mul_i(stat, u_v[kk-1])
                    u_v = torch.stack([u_.ravel() for u_ in u_v], -1)
                    u_v, _ = torch.linalg.qr(u_v)
                    u_v_mats = [u_.reshape(grad.size()) for u_ in u_v.T]
                    #do stuff in reduced subspace
                    stat_u = torch.stack([mul_i(stat, u_).ravel() for u_ in u_v_mats], -1)
                    G_m = u_v.T @ stat_u
                    UTg = torch.stack([(u_v_mats[i] * grad).sum() for i in range(kry)])[:, None]
                    f_G = mf.matrix_power_svd(G_m, 1/exp, double=True)
                    G_p = G_m + UTg @ UTg.T  # U^T D U
                    try:
                        f_Gp = mf.matrix_power_svd(G_p, 1/exp, double=True) 
                    except Exception as err:
                        print('G_m', G_m)
                        print('G_p', G_p)
                        print('f_G', f_G)
                        print('matsvd failed')
                        raise err
                else:
                    ## Lanczos w/o reorthogonalization
                    # Ref: Alg 1 in https://sma.epfl.ch/~anchpcommon/publications/rank_one.pdf
                    # Nb: _v denotes vector, _s scalar, _m matrix
                    #     (mat-vec muls are kept in kronecker form)
                    u_v = [0]*(kry+1)
                    u_v[0] = torch.zeros_like(grad)
                    grad_norm = mf.matrices_norm(grad, norm='fro')
                    u_v[1] = grad * (1 / grad_norm)
                    G_m = torch.zeros((kry, kry), device=grad.device, dtype=grad.dtype)
                    b_s = 0
                    for kk in range(1, kry+1):
                        w_v = mul_i(stat, u_v[kk]) - b_s * u_v[kk-1]
                        a_s = (u_v[kk] * w_v).sum()
                        G_m[kk-1, kk-1] = a_s
                        w_v = w_v - a_s * u_v[kk]
                        b_s = mf.matrices_norm(w_v, norm='fro')
                        if kk < kry-1:
                            G_m[kk-1, kk] = b_s
                            G_m[kk, kk-1] = b_s
                        if kk < kry:
                            u_v[kk+1] = w_v * (1 / b_s)
                    u_v_mats = u_v[1:]
                    u_v = torch.stack([u_.ravel() for u_ in u_v_mats], -1)
                    f_G = mf.matrix_power_svd(G_m, 1/exp, double=True)
                    G_m[0,0] += grad_norm**2
                    f_Gp = mf.matrix_power_svd(G_m, 1/exp, double=True)
                X = mf.symmetrize(mf.symmetrize(f_Gp) - mf.symmetrize(f_G))
                # We expect X[0,0] to dominate:
                if 1-(X[0,0]/X.sum()) < self._hps.matrix_eps:
                    precon_inv.add_(X[0,0] * contr(u_v_mats[0], u_v_mats[0]))
                else:
                    # unstable:
                    #X_cho = torch.linalg.cholesky(X)
                    X_cho = mf.matrix_power_svd(X, 0.5, double=True)
                    ## apply KrAD
                    UXch = [u_.reshape(grad.size()) for u_ in (u_v @ X_cho).T]
                    if self.debug and not all(u_.isfinite().all() for u_ in UXch):
                        print('X_cho', X_cho)
                        print('X', X)
                        print('f_Gp', f_Gp)
                        print('f_G', f_G)
                        raise ValueError('U X_chol broke')
                    ## apply KrAD
                    delta_L = torch.zeros_like(precon_inv)
                    for kk in range(kry):
                        delta_L += contr(UXch[kk], UXch[kk])
                    precon_inv.add_(delta_L)
                    # delta_L = torch.zeros_like(precon_inv)
                    # for kk in range(kry-1, -1, -1):
                    #     delta_L += X[kk, kk] * contr(u_v_mats[kk], u_v_mats[kk])
                    #     for ll in range(kry-1, kk, -1):
                    #         delta_L += 2 * X[kk, ll] * contr(u_v_mats[kk], u_v_mats[ll])
                    # precon_inv.add_(delta_L)
                self.statistics[j*rank+i] = mf.symmetrize(mf.mat_pow(precon_inv, exp))
                eps_rt = self._hps.matrix_eps ** (1/exp)
                self.preconditioners[j*rank+i] = mf.symmetrize(mf.matrix_power_svd(precon_inv, -1, double=self._hps.double, matrix_eps=eps_rt, eig_eps=eps_rt))
                if self.debug:
                    if not precon_inv.isfinite().all():
                        print('stat', stat)
                        #print('delta_L', delta_L)
                        raise ValueError('precon_inv broke')


class KradaKry(Shampoo):
    r"""Implements a simple version of KradaKry Optimizer Algorithm.
    Extends the unoptimized official pytorch implementation of Shampoo
    """
    @torch.no_grad()
    def __init__(self, params, **kwargs):
        self.debug = kwargs.get('debug', False)
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def init_var_state(self, var, state):
        """Initialize the PyTorch state of for a single variable."""
        prec = KradaKryPreconditioner(var, self.hps, debug=self.debug)

        # original, pared down        
        var = var.detach()
        state[MOMENTUM] = torch.zeros_like(var.data, device=var.device)
        state[PRECONDITIONER] = prec
        state[STEP] = 0
        state[GRAFT] = Graft(self.hps, var)
