## Simple Kradagrad-- that extends official unoptimized Shampoo implementation

import math
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

from . import positive_matrix_functions as mf
from .third_party.shampoo import (
    GRAFT,
    MOMENTUM,
    PRECONDITIONER,
    STEP,
    AdagradGraft,
    AdamGraft,
    Graft,
    LayerwiseGrafting,
    Preconditioner,
    SGDGraft,
    Shampoo,
)


class KrADmmPreconditioner(Preconditioner):
    """Inherit from Preconditioner mainly for the block partitioning
    Otherwise treat as abstract and override everything else
    """

    @torch.no_grad()
    def __init__(self, var, hps, eps_override=1.0, debug=False):
        """
        debug mode print tensor values likely to be non-finite, raises exceptions
        eps_override changes initialization of preconditioner diagonal (usually 1.0)
        """
        super().__init__(var, hps, eps_override=eps_override)
        self.debug = debug
        self.updated = set()

    @torch.no_grad()
    def exponent_for_preconditioner(self):
        """Returns exponent to use for pth root M^{1/p}.
        If `inverse_exponent_override` is set, use ONS-style update
        """
        return 2 if self._hps.inverse_exponent_override == 0 else 1

    @torch.no_grad()
    def add_statistic(self, grad, i):
        """Compute inverse KrAD statistic from gradients and add to the correct state entries.

        Args:
          grad: Gradient to compute statistics from.
          i: rank for which to add statistic to
        """
        if not self.statistics:
            return
        grad = grad.type(torch.float32)
        partitioned_grads = self.partition_grad(grad)
        partitioned_precon_grads = self.preconditioned_grad(
            grad, statistics=True, unmerged=True
        )
        w1 = self._hps.beta2
        damping = self._hps.matrix_eps > 0
        rank = len(self._transformed_shape)
        self.updated |= {i}
        for j, (grad_, precon_grad_) in enumerate(
            zip(partitioned_grads, partitioned_precon_grads)
        ):
            axes = list(range(i)) + list(range(i + 1, rank))
            stat = self.statistics[j * rank + i]
            precon = self.preconditioners[j * rank + i]
            if w1 < 1:
                stat.mul_(1 / w1)

            lgrgt = torch.tensordot(precon_grad_, grad_, [axes, axes])
            if self.debug and not lgrgt.isfinite().all():
                print(
                    "self.updated",
                    self.updated,
                    "\nlgrgt",
                    lgrgt,
                    "\nprecon_grad",
                    precon_grad_,
                    "\nstat",
                    stat_,
                )
                raise ValueError("lgrgt broke")

            if damping:
                eps = self._hps.matrix_eps

                stat2eps = stat.mm(stat.T.mul(-eps))
                stat_prime = stat + stat2eps

                lgrgt_2eps = stat.mm(lgrgt.mul(-eps))
                lgrgt_prime = lgrgt + lgrgt_2eps

                if self.debug and (
                    not lgrgt_prime.isfinite().all() or not stat_prime.isfinite().all()
                ):
                    print(
                        "self.updated",
                        self.updated,
                        "\nlgrgt_prime",
                        lgrgt_prime,
                        "\nstat_prime",
                        stat_prime,
                        "\nstat",
                        stat,
                        "\nprecon",
                        precon,
                        "\ngrad",
                        grad_,
                        "\nprecon_grad",
                        precon_grad_,
                    )
                    raise ValueError("damping broke")
                lgrgt = lgrgt_prime
                # stat = mf.symmetrize(stat_prime)
                stat = stat_prime

            # damping
            t_k = -(1 + mf.matrices_norm(lgrgt, "fro"))
            lgrgt.mul_(1 / t_k)
            DX = lgrgt.mm(stat.T)
            if self.debug and not DX.isfinite().all():
                print("DX", DX, "\nlgrgt", lgrgt)
                raise ValueError("DX broke")
            self.statistics[j * rank + i] = stat + DX

    @torch.no_grad()
    def compute_preconditioners(self, **kwargs):
        """Compute L^{1/exp} for each statistics matrix L"""
        if not self.statistics:
            return  # sgd
        # kradagradmm
        exp = self.exponent_for_preconditioner()
        for i in list(self.updated):
            stat = self.statistics[i]
            try:
                if stat.device.type == "cpu":
                    self.preconditioners[i] = (
                        mf.matrix_power_svd(stat, 1 / exp) if exp > 1 else stat
                    )
                else:
                    self.preconditioners[i] = mf.mat_root(
                        stat[None, ...],
                        exp,
                        self.preconditioners[i][None, ...],
                        iters=12,
                        tol=1e-4,  # debug=True
                    )[
                        0
                    ]  # mf.mat_root operates on batches
            except Exception as err:
                if self.debug:
                    print("stat", stat, "\nmat_root broke")
                raise err
        self.updated = set()

    @torch.no_grad()
    def partition_grad(self, grad):
        """Partition gradient.
        Args:
          grad: Gradient to partition.
        """
        reshaped_grad = torch.reshape(grad.detach(), self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        return partitioned_grads


class KradagradMM(Shampoo):
    r"""Implements a simple version of Kradagrad-- Optimizer Algorithm.
    Extends the unoptimized official pytorch implementation of Shampoo
    """

    @torch.no_grad()
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.debug = kwargs.get("debug")
        self.initialized = False
        self._step = 0

    @torch.no_grad()
    def init_var_state(self, var, state):
        """Initialize the PyTorch state of for a single variable."""
        var = var.detach()
        prec = KrADmmPreconditioner(var, self.hps, eps_override=1.0, debug=self.debug)

        # original, pared down
        state[MOMENTUM] = torch.zeros_like(var.data, device=var.device)
        state[PRECONDITIONER] = prec

    # reimplementation for single statistc update
    @torch.no_grad()
    def step(self, closure=None):
        hps = self.hps
        # currently no grafting
        if not self.initialized:
            for group in self.param_groups:
                lr = group["lr"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if not (state := self.state[p]):
                        self.init_var_state(p, state)
            self.initialized = True
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                state = self.state[p]
                if "preconditioner" not in state.keys():
                    import ipdb

                    ipdb.set_trace()
                    dumb = 1
                prec = state[PRECONDITIONER]

                # Compute stats and preconditioners
                if self._step % hps.statistics_compute_steps == 0:
                    sh_ = prec._transformed_shape
                    nd_ = len(sh_)
                    if self._step < nd_**2:
                        # simple equal updating schedule:
                        ix = self._step % nd_
                    else:
                        # O(k^0.5) for everything except largest dim
                        max_dim = max(sh_)
                        max_ix = [j for (j, x) in enumerate(sh_) if x == max_dim][0]
                        step_sqrt_floor = int(math.sqrt(self._step))
                        exc = self._step - step_sqrt_floor**2
                        ix = exc if exc < nd_ else max_ix

                    prec.add_statistic(grad, ix)
                if self._step % hps.preconditioning_compute_steps == 0:
                    prec.compute_preconditioners()

                # Precondition
                krad_grad = grad
                if self._step >= self.hps.start_preconditioning_step:
                    krad_grad = prec.preconditioned_grad(grad)

                # Weight decay
                if self.hps.weight_decay != 0.0:
                    krad_grad.add_(p.data, alpha=hps.weight_decay)

                # Momentum and Nesterov momentum, if needed
                state[MOMENTUM].mul_(momentum).add_(krad_grad)

                momentum_update = state[MOMENTUM]
                wd_update = krad_grad

                if hps.nesterov:
                    momentum_update.mul_(momentum).add_(wd_update)

                # Final update
                p.data.add_(momentum_update, alpha=-lr)

        self._step += 1
