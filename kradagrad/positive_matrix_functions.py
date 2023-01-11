import sys

import numpy as np
import torch

from .math_utils import cubic

# Note: these are unbatched, intended for Tensors sized (N, N)
def _mat_fro_norm(A):  # torch's version is incorrect
    return torch.sqrt((A ** 2).sum())

def _mat_inf_norm(A):
    return torch.abs(A).sum(dim=-1).max(dim=-1)[0]

def _mat_l1_norm(A):
    return torch.abs(A).sum()

def matrices_norm(A, norm: str='tr'):
    if norm == 'inf':  # decently tight bound on largest eig for diag dom
        A_norm = _mat_inf_norm(A)
    elif norm == 'fro':  # looser bound on largest eig
        A_norm = _mat_fro_norm(A)
    elif norm == 'l1':  # looser bound on largest eig
        A_norm = _mat_l1_norm(A)
    else:  # trace, even looser bound on largest eig
        A_norm = torch.trace(A)
    return A_norm

def symmetrize(A):
    return A / 2 + A.t() / 2

def _matrices_info(A: torch.Tensor, norm: str='tr'):
    # check if A is 3D, return size and norm
    A_sz = A.size()
    A_dim = A.dim()
    if A_dim > 2:
        caller = sys._getframe(1).f_code.co_name
        raise ValueError("{} supports matrices only! Input dim: {}".format(caller, A_dim))
    A_norm = matrices_norm(A, norm=norm)
    A_norm = A_norm[None, None]
    A_dev = A.device

    return A_sz, A_dim, A_norm, A_dev

def mat_pow(A: torch.Tensor, p: int):
    # performs A^p on symmetric matrices using O(log_2(p)) matmuls
    if p < 1:
        raise ValueError("mat_pow only valid for positive integer powers!")
    if p == 1:
        return A
    # string:
    pb = bin(p)[2:]
    X_prev = A  # tmp var tracking A^(2^i))
    Xpow = None  # accumulator
    for i, save in enumerate(pb[::-1]):
        if save == '1':
            if Xpow is None:
                Xpow = X_prev
            else:
                Xpow = Xpow.mm(X_prev)
        if i < len(pb) - 1:
            # stays more symmetric:
            X_prev = X_prev.mm(X_prev.transpose(-2, -1))
    return Xpow

def mat_root(A, p, A_p=None, **kwargs):
    """
    performs A^(1/p) using the "best" Newton iterations
    main interface to other methods in this file
    optionally with initialization

    A: matrix to take root
    p: root to take
    A_p: initial guess
    iters: num of iterations
    tol: error threshold to early terminate
    inner_iters: if using Newton to compute inverses, iterations for those loops
    inner_tol: if using Newton to compute inverses, threshold for early terminating those loops
    norm: which normalization to use: {'inf', 'fro', 'tr'}
    double: use 64-bit instead of 32-bit precision
    debug: print debugging info
    """
    if p == 1:
        return A
    elif p == 2:
        # ignore the warm, NS is great
        X = matrix_sqrt_NS(A, **kwargs)
    elif p % 2 == 0:
        X = matrix_even_root_N_warm(p, A, **kwargs)
    else:
        X = matrix_even_root_N_warm(2 * p, A.mm(A.t()), **kwargs)
    return X


def matrix_inv_warm(A: torch.Tensor, A_p: torch.Tensor=None, tol: float=1e-6, iters: int=10, norm='inf', debug: bool=False, double: bool=False, **kwargs) -> torch.Tensor:
    # Newton method for inverting A
    orig_type = A.dtype
    A = A.type(torch.float64 if double else torch.float32)
    A_sz, A_dim, A_norm, A_dev = _matrices_info(A, norm=norm)

    Z = A
    I = torch.eye(*A_sz[-2:], device=A_dev).type_as(A)
    I_norm = matrices_norm(I, norm='l1')
    I2 = 2 * I
    if A_p is None:
        A_p = A.transpose(-2, -1)
    X = A_p / A_norm
    last_norm = float('inf')
    for it_ in range(iters):
        Y = Z.mm(X)
        if Y.isnan().any():
            if debug:
                print('warning: inv iterations unstable')
            break
        #X = X.mm(I2 - Y)
        X.addmm_(X, Y, beta=2, alpha=-1)
        this_norm = matrices_norm(Y - I, norm='l1')
        if this_norm / I_norm < tol or (this_norm > last_norm and it_ > 15):
            if debug:
                print('inv quit after {} iter'.format(it_ + 1))
            break
        last_norm = this_norm
    return X.type(orig_type)

def matrix_even_root_N_warm(p: int, A: torch.Tensor, A_p: torch.Tensor=None, tol: float=1e-6, iters: int=20, norm: str='inf', inner_tol: float=1e-6, inner_iters: int=10, debug: bool=False, double: bool=False, **kwargs) -> torch.Tensor:
    # Symmetric coupled Newton iterations to compute A^(1/p) for even p
    # X_p: initial guess of A^(1/p). If `None`, use I
    # Uses Newton iterations to compute inverse in an inner loop
    if p % 2 != 0:
        raise ValueError("matrix_even_root_N_warm only supports even roots! p: {}".format(p))
    orig_type = A.dtype
    A = A.type(torch.float64 if double else torch.float32)
    A_sz, A_dim, A_norm, A_dev = _matrices_info(A, norm=norm)

    Z = A / A_norm
    I = torch.eye(*A_sz[-2:], device=A_dev).type_as(A)
    I_norm = matrices_norm(I, norm='l1')
    if A_p is None:
        A_p = I

    p_2 = p // 2
    A_norm_p = A_norm ** (1 / p)
    X = A_p / A_norm_p
    Xp2 = mat_pow(X, p_2)
    Xp2_inv = matrix_inv_warm(Xp2, I, inner_tol, inner_iters, norm=norm, debug=debug, double=double)
    M = Xp2_inv.mm(Z.mm(Xp2_inv))
    IM_pp2_inv = I
    Ip_1 = I * (p - 1)
    last_norm = float('inf')
    for i_ in range(iters):
        IM_p = (Ip_1 + M) / p
        X = X.mm(IM_p) if i_ % 2 == 0 else IM_p.mm(X)
        X = symmetrize(X)
        this_norm = matrices_norm(IM_p - I, norm='l1')
        if this_norm / I_norm < tol or (this_norm > last_norm and i_ > 15):
            if debug:
                print('root exit early after {} iter'.format(i_ + 1))
            break
        if i_ < iters:
            IM_pp2 = mat_pow(IM_p, p_2)
            IM_pp2_inv = matrix_inv_warm(IM_pp2, IM_pp2_inv, inner_tol, inner_iters, norm=norm, debug=debug, double=double)
            M_ = IM_pp2_inv.mm(M.mm(IM_pp2_inv))

            M = M_
            last_norm = this_norm
    return (X * A_norm_p).type(orig_type)

def matrix_sqrt_NS(A: torch.Tensor, iters: int=25, tol: float=1e-5, batched: bool=False, norm: str='inf', debug: bool=False, verbose: bool=False, double: bool=False, **kwargs) -> torch.Tensor:
    # 3D batch of matrices only
    # Newton Schulz iterations; converge quadratically, but no warm start
    orig_type = A.dtype
    A = A.type(torch.float64 if double else torch.float32)
    A_sz, A_dim, A_norm, A_dev = _matrices_info(A, norm=norm)

    Y = A / A_norm
    Z = torch.eye(*A_sz[-2:], device=A_dev).type_as(A)
    I = torch.eye(*A_sz[-2:], device=A_dev).type_as(A)
    eye3 = 3 * I

    I_norm = matrices_norm(I, 'l1')
    last_norm = float('inf')
    for it_ in range(iters):
        #X = 0.5 * eye3 - Z.mm(0.5 * Y)
        X = torch.addmm(eye3, Z, Y, beta=0.5, alpha=-0.5)
        if it_ < iters - 1:
            Z = X.mm(Z)
            Z = symmetrize(Z)
        Y = Y.mm(X)
        Y = symmetrize(Y)
        if debug and (not Y.isfinite().all() or not Z.isfinite().all()):
            print(it_, '||A||', A_norm, '\nX', X, '\nZ', Z, '\nY', Y)
            raise ValueError('matrix_sqrt_NS broke')
        this_norm = matrices_norm(X - I, 'l1')
        if this_norm / I_norm < tol or (this_norm > last_norm and it_ > 15):
            if verbose:
                print('mat sqrt NS exit early after {} iters'.format(it_ + 1))
            break
        last_norm = this_norm
    Y = Y * (A_norm ** (0.5))
    return Y.type(orig_type)

def matrix_power_svd(matrix: torch.Tensor, power: float, double: bool=False, eps=0) -> torch.Tensor:
    # Really, use hermitian eigenvalue decomposition
    orig_type = matrix.dtype
    L, V = torch.linalg.eigh(matrix.type(torch.float64 if double else torch.float32))
    L = torch.maximum(L, torch.zeros(1, device=L.device)) + eps
    return ((V * L.pow(power)) @ V.T).type(orig_type)
