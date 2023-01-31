""" Real Harmonic mean
"""
from __future__ import print_function

import torch

from ..third_party.shampoo import (
  Graft, Preconditioner, Shampoo,
  GRAFT, MOMENTUM, PRECONDITIONER, STEP
)
from .. import positive_matrix_functions as mf

class ShampooHMPreconditioner(Preconditioner):
  """Compute statistics/shape from gradients for preconditioning."""

  @torch.no_grad()
  def __init__(self, var, hps, eps_override=None, **kwargs):
    super().__init__(var, hps, eps_override=eps_override, **kwargs)
    rank = len(self._transformed_shape)
    self._len_partitioned_grads = 1
    for i, sz in self._partitioner._split_sizes:
      self._len_partitioned_grads *= len(sz)
    if rank < 1:
      self.eigs = []
    else:
      precision = torch.float64 if hps.double else torch.float32
      self.eigs = [torch.zeros([1]*rank, device=var.device, dtype=precision) for _ in range(self._len_partitioned_grads)]

  @torch.no_grad()
  def add_statistics(self, grad):
    """Compute statistics from gradients and add to the correct state entries.

    Args:
      grad: Gradient to compute statistics from.
    """
    if not self.statistics: return  # sgd
    rank = len(self._transformed_shape)
    w1 = self._hps.beta2
    w2 = 1.0 if w1 == 1.0 else (1.0 - w1)
    # shampoo
    reshaped_grad = torch.reshape(grad.detach(), self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    for j, grad in enumerate(partitioned_grads):
      if self._hps.double:
          grad = grad.double()
      if self._hps.bf16:
          grad = grad.bfloat16()
      for i in range(rank):
        axes = list(range(i)) + list(range(i + 1, rank))
        stat = torch.tensordot(grad, grad, [axes, axes])
        self.statistics[j*rank + i].mul_(w1).add_(stat, alpha=w2)


  @torch.no_grad()
  def compute_preconditioners(self, **kwargs):
    """Compute L^{-1/exp} for each stats matrix L."""
    if not self.statistics: return  # sgd
    eps = self._hps.matrix_eps
    precision = torch.float64 if self._hps.double else torch.float32
    # shampoo using harmonic mean instead of geometric mean
    rank = len(self._transformed_shape)
    for j in range(self._len_partitioned_grads):
      self.eigs[j].zero_()
      for i in range(rank):
        stat = self.statistics[j*rank + i]
        device = stat.device
        mat = stat.type(precision) + eps*torch.eye(stat.size()[0], device=device)
        L, U = torch.linalg.eigh(mat)
        self.preconditioners[i] = U
        expands = [None,] * rank
        expands[i] = slice(None)
        self.eigs[j] = self.eigs[j] + (
          1 / torch.maximum(L, eps * torch.ones(1, device=device))
        )[tuple(expands)]
      self.eigs[j] = (self.eigs[j] / rank) ** 0.5

  @torch.no_grad()
  def preconditioned_grad(self, grad, skip=[], unmerged=False):
    """Precondition the gradient.

    Args:
      grad: A gradient tensor to precondition.
      skip: list of `int` in [0, grad.ndim] for the dimensions to not precondition
      unmerged: if `True`, return unmerged

    Returns:
      A preconditioned gradient.
    """
    if not self.preconditioners: return grad  # sgd
    # precondition gradient
    reshaped_grad = torch.reshape(grad.detach(), self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    preconditioned_partitioned_grads = []
    num_splits = self._partitioner.num_splits()
    for i, grad in enumerate(partitioned_grads):
      mats = self.preconditioners
      preconditioners_for_grad = mats[i * num_splits:(i + 1) * num_splits]
      rank = len(grad.shape)
      orig_type = grad.type()
      precision = torch.float64 if self._hps.double else torch.float32
      precond_grad = grad.type(precision)
      # multiply by U^T
      for j in range(rank):
        preconditioner = preconditioners_for_grad[j]
        if j in skip:
          pg_ = precond_grad.moveaxis(0, -1)
        else:
          pg_ = torch.tensordot(preconditioner.T.type(precision), precond_grad, [[0], [0]])
          pg_ = pg_.moveaxis(0, -1)
        if 'debug' in self.__dict__ and self.debug and not precond_grad.isfinite().all():
          print(i, j, 'precon', preconditioner, '\nprecond_grad (before)', precond_grad, '\nprecond_grad (after)', pg_)
          raise ValueError('precond_grad broke')
        precond_grad = pg_
      # multiply by Sigma
      precond_grad *= self.eigs[i]
      # multiply by U
      for j in range(rank):
        preconditioner = preconditioners_for_grad[j]
        if j in skip:
          pg_ = precond_grad.moveaxis(0, -1)
        else:
          pg_ = torch.tensordot(preconditioner.type(precision), precond_grad, [[0], [0]])
          pg_ = pg_.moveaxis(0, -1)
        if 'debug' in self.__dict__ and self.debug and not precond_grad.isfinite().all():
          print(i, j, 'precon', preconditioner, '\nprecond_grad (before)', precond_grad, '\nprecond_grad (after)', pg_)
          raise ValueError('precond_grad broke')
        precond_grad = pg_
      
      preconditioned_partitioned_grads.append(precond_grad.type(orig_type))
    if not unmerged:
      merged_grad = self._partitioner.merge_partitions(
          preconditioned_partitioned_grads)
      return torch.reshape(merged_grad, self._original_shape)
    else:
        return preconditioned_partitioned_grads


class ShampooHM(Shampoo):
  """The Shampoo optimizer using harmonic mean instead of geometric."""

  @torch.no_grad()
  def init_var_state(self, var, state):
    """Initialize the PyTorch state of for a single variable."""
    var = var.detach()
    state[STEP] = 0
    state[MOMENTUM] = torch.zeros_like(var.data, device=var.device)
    state[PRECONDITIONER] = ShampooHMPreconditioner(var, self.hps)
    state[GRAFT] = Graft(self.hps, var)