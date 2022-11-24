import sys

#import mpmath
import numpy as np
import torch

import cubic

### Support matrix even and odd roots separately
# Note: most of these are batched, intended for Tensors sized (B, N, N)

def _matrices_info(A: torch.Tensor, norm: str='fro'):
    # check if A is 3D, return size and norm
    # `trace` is True, use trace instead of spectral norm
    A_sz = A.size()
    A_dim = A.dim()
    if A_dim < 3:
        caller = sys._getframe(1).f_code.co_name
        raise ValueError("{} supports batches of matrices only! Input dim: {}".format(caller, A_dim))

    if norm == 'sing':
        A_norm = (1 + 1e-6) * torch.stack([torch.linalg.matrix_norm(a, ord=2) for a in A])
    elif norm == 'fro':
        A_norm = torch.stack([torch.linalg.matrix_norm(a) for a in A])
    else:  # trace
        A_norm = torch.stack([torch.trace(a) for a in A])
    A_norm = A_norm[..., None, None]
    A_dev = A.device

    return A_sz, A_dim, A_norm, A_dev

def _batcher(batch_size: int, M: torch.Tensor):
    return M.unsqueeze(0).repeat([batch_size, 1, 1])

def mat_pow(A: torch.Tensor, p: int):
    # performs A^p using O(log_2(p)) matmuls
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
                Xpow = Xpow.bmm(X_prev)
        if i < len(pb) - 1:
            # stays more symmetric:
            X_prev = X_prev.bmm(X_prev.transpose(-2, -1))
    return Xpow


def matrix_inv_warm(A: torch.Tensor, A_p:torch.Tensor, tol: float=1e-4, iters: int=10, norm: str='fro') -> torch.Tensor:
    # Newton method for inverting A
    A_sz, A_dim, A_norm, A_dev = _matrices_info(A, norm=norm)
    A_batch = A_sz[0]

    Z = A / A_norm
    I = torch.eye(*A_sz[-2:], device=A_dev)
    I = _batcher(A_batch, I)
    X = A_p
    for it_ in range(iters):
        Y = Z.bmm(X)
        X = 2 * X - X.bmm(Y)
        if torch.linalg.matrix_norm(Y - I, ord=norm).max() / A_sz[-1] < tol:
            break
    return X / A_norm

def matrix_even_root_N_warm(p: int, A: torch.Tensor, A_p: torch.Tensor, tol: float=1e-4, iters: int=20, norm: str='fro', inner_tol: float=1e-4, inner_iters: int=5) -> torch.Tensor:
    # Coupled Newton iterations to compute A^(1/p) for even p
    # X_p: initial guess of A^(1/p)
    # Uses either inner Newton iterations to compute inverse or Cholesky solver to apply
    if p % 2 != 0:
        raise ValueError("matrix_even_root_N_warm only supports even roots! p: {}".format(p))
    A_sz, A_dim, A_norm, A_dev = _matrices_info(A, norm=norm)
    A_batch = A_sz[0]

    Z = A / A_norm
    I = torch.eye(*A_sz[-2:], device=A_dev)
    I = _batcher(A_batch, I)

    p_2 = p // 2
    A_norm_p = A_norm ** (1 / p)
    X = A_p / A_norm_p
    Xp2 = mat_pow(X, p_2)
    #if inner_iters > 0:
    Xp2_inv = matrix_inv_warm(Xp2, I, inner_tol, inner_iters, norm)
    M = Xp2_inv.bmm(Z.bmm(Xp2_inv))
    IM_pp2_inv = I
    #else:
    #    Xp2_LD, Xp2_pivot, _ = torch.linalg.ldl_factor_ex(Xp2)
    #    M = torch.linalg.ldl_solve(
    #        Xp2_LD, Xp2_pivot,
    #        torch.linalg.ldl_solve(Xp2_LD, Xp2_pivot, Z).transpose(-2, -1)
    #    )
    for i_ in range(iters):
        IM_p = (I * (p - 1) + M) / p
        if i_ % 2 == 0:
            X = X.bmm(IM_p)
        else:
            X = IM_p.bmm(X)
        X = (X + X.transpose(-2, -1)) / 2
        if torch.linalg.matrix_norm(IM_p - I, ord=norm).max() / A_sz[-1] < tol:
            break
        if i_ < iters:
            IM_pp2 = mat_pow(IM_p , p_2)
            #if inner_iters > 0:
            IM_pp2_inv = matrix_inv_warm(IM_pp2, IM_pp2_inv, inner_tol, inner_iters, norm)
            M_ = IM_pp2_inv.bmm(M.bmm(IM_pp2_inv))
            #else:
            #    IM_pp2_LD, IM_pp2_pivot, _ = torch.linalg.ldl_factor_ex(IM_pp2)
            #    M = torch.linalg.ldl_solve(
            #        IM_pp2_LD, IM_pp2_pivot,
            #        torch.linalg.ldl_solve(IM_pp2_LD, IM_pp2_pivot, M).transpose(-2, -1)
            #    )
            M = M_
    return X * A_norm_p

def matrix_sqrt_NS(A: torch.Tensor, iters: int=25, batched: bool=False, norm: str='fro') -> torch.Tensor:
    # 3D batch of matrices only
    # Newton Schulz iterations; converge quadratically, but no warm start
    A_sz, A_dim, A_norm, A_dev = _matrices_info(A, norm=norm)
    A_batch = A_sz[0]

    Y = A / A_norm
    Z = torch.eye(*A_sz[-2:], device=A_dev)
    eye3 = 3 * torch.eye(*A_sz[-2:], device=A_dev)

    # deal with batch dimension:
    Z = _batcher(A_batch, Z)
    eye3 = _batcher(A_batch, eye3)

    for it_ in range(iters):
        X = 0.5 * (eye3 - Z.bmm(Y))
        if it_ < iters - 1:
            if batched:  # if we have extra memory
                U = torch.cat([Y, X])
                V = torch.cat([X, Z])
                W = U.bmm(V)
                Y = W[:A_batch]
                Z = W[A_batch:]
            else:
                Y = Y.bmm(X)
                Z = X.bmm(Z)
        else:
            Y = Y.bmm(X)
    Y = Y * (A_norm ** (0.5))
    return Y

def matrix_sqrt_warm(L: torch.Tensor, L_sqrt_init: torch.Tensor, iters: int=100, norm: str='fro', accel: str='line') -> torch.Tensor:
    # 3D batch of matrices only
    # Binomial iterations. Linear convergence, debatable whether it's comparable even to NS without warm start
    # nb. naive applications of aitken delta acceleration don't work
    L_sz, L_dim, L_norm, L_dev = _matrices_info(L, norm=norm)
    L_batch = L_sz[0]

    eyes = _batcher(L_batch, torch.eye(*L_sz[-2:], device=L_dev))
    line_search = accel == 'line'

    A = eyes - L / L_norm
    L_norm_sqrt = L_norm ** 0.5
    X = eyes - L_sqrt_init.to(L_dev) / L_norm_sqrt
    for it_ in range(iters):
        if line_search:

            # solves:
            #  t = argmin_t || [I - (X + t*D)]^2 - (I - A) ||_F^2
            #    = argmin_t || C + L*t + Q*t^2 ||_F^2

            X1 = (A + X.bmm(X)) / 2
            Del = X1 - X
            XmI = X - eyes
            Del_norm = torch.linalg.matrix_norm(Del, ord=2)
            
            # this one arises from cancellation, not from t
            # so assign this before normalizing:
            con = 2 * Del  # (C)onstant

            # for numerical stability of the cubic solver
            Del = Del / Del_norm[..., None, None]

            qua =  Del.bmm(Del)  # (Q)uadratic
            lin = Del.bmm(XmI)  # (L)inear
            lin = lin + lin.transpose(-2, -1)

            # compute cubic coeffs
            a = 2 * (qua * qua).sum([-2, -1])
            b = 3 * (qua * lin).sum([-2, -1])
            c = 2 * (qua * con).sum([-2, -1]) + (lin * lin).sum([-2, -1])
            d = (lin * con).sum([-2, -1])
            try:
                t = cubic.solve_smallest(a, b, c, d, thr=Del_norm)

                # we know the update can be at least Del_norm (unaccelerated)
                low = (t < Del_norm)
                t[low] = Del_norm[low]
                # trust region, keep || X ||_2 < 1
                max_norm = torch.fmax(1 - torch.linalg.matrix_norm(X, ord=2), Del_norm)
                t[t > max_norm] = max_norm[t > max_norm]

                # Finally, apply update
                X = X + t[..., None, None] * Del
            except:
                X = X1
        else:
            X = (A + X.bmm(X)) / 2
    return (eyes - X) * L_norm_sqrt

def matrix_power_svd(matrix: torch.Tensor, power: float) -> torch.Tensor:
    # use CPU for svd for speed up
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).to(device)

"""
def make_pade_rooter(n: int, m: int, l: int, norm: str='fro'):
    # Uses pade approximation of order[m, l] to create a function
    # that computes the nth root on a batch of matrices
    # Nb. Not great compared to Newton or even Binomial w/ line search
    f = lambda x: mpmath.root(1 - x, n)
    a = mpmath.taylor(f, 0, m+l+4)
    p, q = mpmath.pade(a, m, l)
    pade_p, pade_q = [torch.Tensor(np.array(x).astype(float)) for x in [p, q]]

    def matrix_rt_pade(A: torch.Tensor, pades) -> torch.Tensor:
        # Heavily inspired by: https://github.com/KingJamesSong/FastDifferentiableMatSqrt
        A_sz, A_dim, A_norm, A_dev = _matrices_info(A, norm=norm)
        A_batch = A_sz[0]

        Y = A / A_norm
        pade_p, pade_q = [x.to(A_dev) for x in pades]
        I = torch.eye(*A_sz[-2:], device=A_dev)
        I = _batcher(A_batch, I)

        # # we might get underflow summing from large to small...
        # # try multiplicative updates instead?
        # p_rt = pade_p[0] * I
        # q_rt = pade_q[0] * I
        # X = I - Y
        # X_pow = X
        # for i_ in range(max(m, l)):
        #     if i_ < m:
        #         p_rt += pade_p[i_ + 1] * X_pow
        #     if i_ < l:
        #         q_rt += pade_q[i_ + 1] * X_pow
        #     X_pow = X_pow.bmm(X)
        #     X_pow = (X_pow + X_pow.transpose(-2, -1)) / 2
        X = I - Y
        if len(pade_p) > 0:
            p_rt = pade_p[-1] * X
        else:
            p_rt = torch.zeros_like(X)
        if len(pade_q) > 0:
            q_rt = pade_q[-1] * X
        else:
            q_rt = torch.zeros_like(X)
        for i_ in range(max(m, l)-1, 0, -1):
            if i_ < m:
                if i_ % 2 == 0:
                    p_rt = pade_p[i_] * X + X.bmm(p_rt)
                else:
                    p_rt = pade_p[i_] * X + p_rt.bmm(X)
            if i_ < l:
                if i_ % 2 == 0:
                    q_rt = pade_q[i_] * X + X.bmm(q_rt)
                else:
                    q_rt = pade_q[i_] * X + q_rt.bmm(X)
        p_rt += pade_p[0] * I
        q_rt += pade_q[0] * I

        # Nb: experimental API! Pin version of pytorch==1.13
        # cholesky: save 1/2 flops, numerically stable
        q_LD, q_pivots, _ = torch.linalg.ldl_factor_ex(q_rt)
        Y_rt = torch.linalg.ldl_solve(q_LD, q_pivots, p_rt)

        A_rt = Y_rt * (A_norm ** (1./n))
        return A_rt

    return lambda x: matrix_rt_pade(x, (pade_p, pade_q))
"""
