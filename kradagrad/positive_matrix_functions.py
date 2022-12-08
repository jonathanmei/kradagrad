import sys

#import mpmath
import numpy as np
import torch

import cubic

### Support matrix even and odd roots separately
# Note: most of these are batched, intended for Tensors sized (B, N, N)
def _mat_fro_norm(A):  # torch's version is incorrect
    return torch.sqrt((A ** 2).sum(dim=-1).sum(dim=-1))

def _mat_inf_norm(A):
    return torch.abs(A).sum(dim=-1).max(dim=-1)[0]

def matrices_norm(A, norm: str='inf'):
    if norm == 'inf':  # decently tight bound on largest eig for diag dom
        A_norm = _mat_inf_norm(A)
    elif norm == 'fro':  # looser bound on largest eig
        A_norm = _mat_fro_norm(A)
    else:  # trace, even looser bound on largest eig
        A_norm = torch.stack([torch.trace(a) for a in A])
    return A_norm

def symmetrize(A):
    return (A + A.transpose(-2, -1)) / 2

def _matrices_info(A: torch.Tensor, norm: str='inf'):
    # check if A is 3D, return size and norm
    A_sz = A.size()
    A_dim = A.dim()
    if A_dim < 3:
        caller = sys._getframe(1).f_code.co_name
        raise ValueError("{} supports batches of matrices only! Input dim: {}".format(caller, A_dim))
    A_norm = matrices_norm(A, norm=norm)
    A_norm = A_norm[..., None, None]
    A_dev = A.device

    return A_sz, A_dim, A_norm, A_dev

def _batcher(batch_size: int, M: torch.Tensor):
    return M.unsqueeze(0).expand([batch_size, -1, -1])

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
                Xpow = Xpow.bmm(X_prev)
        if i < len(pb) - 1:
            # stays more symmetric:
            X_prev = X_prev.bmm(X_prev.transpose(-2, -1))
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
    debug: print debugging info
    """
    if p == 2:
        # ignore the warm, NS is great
        X = matrix_sqrt_NS(A, **kwargs)
    elif p % 2 == 0:
        X = matrix_even_root_N_warm(p, A, **kwargs)
    else:
        X = matrix_even_root_N_warm(2 * p, A.bmm(A.transpose(-2, -1)), **kwargs)
    return X


def matrix_inv_warm(A: torch.Tensor, A_p: torch.Tensor=None, tol: float=1e-6, iters: int=10, norm='inf', debug: bool=False) -> torch.Tensor:
    # Newton method for inverting A
    orig_type = A.dtype
    A = A.type(torch.float32)
    A_sz, A_dim, A_norm, A_dev = _matrices_info(A, norm=norm)
    A_batch = A_sz[0]

    Z = A
    I = torch.eye(*A_sz[-2:], device=A_dev).type_as(A)
    I = _batcher(A_batch, I)
    I2 = 2 * I
    if A_p is None:
        A_p = A.transpose(-2, -1)
    X = A_p / A_norm
    for it_ in range(iters):
        Y = Z.bmm(X)
        if Y.isnan().any():
            if debug:
                print('warning: inv iterations unstable')
            break
        X = X.bmm(I2 - Y)
        #X = torch.baddbmm(X, X, Y, beta=2, alpha=-1)
        if matrices_norm(Y - I, norm=norm).max() < tol:
            if debug:
                print('inv quit after {} iter'.format(it_ + 1))
            break
    return X.type(orig_type)

def matrix_even_root_N_warm(p: int, A: torch.Tensor, A_p: torch.Tensor=None, tol: float=1e-6, iters: int=20, norm: str='inf', inner_tol: float=1e-6, inner_iters: int=10, debug: bool=False) -> torch.Tensor:
    # Coupled Newton iterations to compute A^(1/p) for even p
    # X_p: initial guess of A^(1/p)
    # Uses either inner Newton iterations to compute inverse or Cholesky solver to apply
    if p % 2 != 0:
        raise ValueError("matrix_even_root_N_warm only supports even roots! p: {}".format(p))
    orig_type = A.dtype
    A = A.type(torch.float32)
    A_sz, A_dim, A_norm, A_dev = _matrices_info(A, norm=norm)
    A_batch = A_sz[0]

    Z = A / A_norm
    I = torch.eye(*A_sz[-2:], device=A_dev).type_as(A)
    I = _batcher(A_batch, I)
    if A_p is None:
        A_p = I

    p_2 = p // 2
    A_norm_p = A_norm ** (1 / p)
    X = A_p / A_norm_p
    Xp2 = mat_pow(X, p_2)
    #if inner_iters > 0:
    Xp2_inv = matrix_inv_warm(Xp2, I, inner_tol, inner_iters, norm='inf', debug=debug)
    M = Xp2_inv.bmm(Z.bmm(Xp2_inv))
    IM_pp2_inv = I
    #IM_p_inv = I
    #else:
    #    Xp2_LD, Xp2_pivot, _ = torch.linalg.ldl_factor_ex(Xp2)
    #    M = torch.linalg.ldl_solve(
    #        Xp2_LD, Xp2_pivot,
    #        torch.linalg.ldl_solve(Xp2_LD, Xp2_pivot, Z).transpose(-2, -1)
    #    )
    Ip_1 = I * (p - 1)
    for i_ in range(iters):
        IM_p = (Ip_1 + M) / p
        if i_ % 2 == 0:
            X = X.bmm(IM_p)
        else:
            X = IM_p.bmm(X)
        X = symmetrize(X)
        if matrices_norm(IM_p - I, norm=norm).max() < tol:
            if debug:
                print('root exit early after {} iter'.format(i_ + 1))
            break
        if i_ < iters:
            IM_pp2 = mat_pow(IM_p, p_2)
            ##if inner_iters > 0:
            IM_pp2_inv = matrix_inv_warm(IM_pp2, IM_pp2_inv, inner_tol, inner_iters, norm='inf', debug=debug)
            M_ = IM_pp2_inv.bmm(M.bmm(IM_pp2_inv))
            
            #IM_p_inv = matrix_inv_warm(IM_p, IM_p_inv, inner_tol, inner_iters, norm='inf')
            #IM_pp2_inv = mat_pow(IM_p_inv, p_2)
            #M_ = IM_pp2_inv.bmm(M.bmm(IM_pp2_inv))

            #else:
            #    IM_pp2_LD, IM_pp2_pivot, _ = torch.linalg.ldl_factor_ex(IM_pp2)
            #    M = torch.linalg.ldl_solve(
            #        IM_pp2_LD, IM_pp2_pivot,
            #        torch.linalg.ldl_solve(IM_pp2_LD, IM_pp2_pivot, M).transpose(-2, -1)
            #    )
            M = M_
    return (X * A_norm_p).type(orig_type)

def matrix_sqrt_NS(A: torch.Tensor, iters: int=25, tol: float=1e-5, batched: bool=False, norm: str='inf', debug: bool=False, verbose: bool=False,**kwargs) -> torch.Tensor:
    # 3D batch of matrices only
    # Newton Schulz iterations; converge quadratically, but no warm start
    orig_type = A.dtype
    A = A.type(torch.float32)
    A_sz, A_dim, A_norm, A_dev = _matrices_info(A, norm=norm)
    A_batch = A_sz[0]

    Y = A / A_norm
    Z = torch.eye(*A_sz[-2:], device=A_dev).type_as(A)
    I = torch.eye(*A_sz[-2:], device=A_dev).type_as(A)
    eye3 = 3 * I

    # deal with batch dimension:
    Z = _batcher(A_batch, Z)
    I = _batcher(A_batch, I)
    eye3 = _batcher(A_batch, eye3)

    for it_ in range(iters):
        #X = 0.5 * (eye3 - Z.bmm(Y))
        X = torch.baddbmm(eye3, Z, Y, beta=0.5, alpha=-0.5)
        if it_ < iters - 1:
            Z = X.bmm(Z)
            Z = symmetrize(Z)
        Y = Y.bmm(X)
        Y = symmetrize(Y)
        if debug and (not Y.isfinite().all() or not Z.isfinite().all()):
            print(it_, '||A||', A_norm, '\nX', X, '\nZ', Z, '\nY', Y)
            raise ValueError('matrix_sqrt_NS broke')
        if matrices_norm(X - I, norm).max() < tol:
            if verbose:
                print('mat sqrt NS exit early after {} iters'.format(it_ + 1))
            break
    Y = Y * (A_norm ** (0.5))
    return Y.type(orig_type)

def matrix_sqrt_warm(L: torch.Tensor, L_sqrt_init: torch.Tensor, iters: int=100, norm: str='inf', accel: str='line') -> torch.Tensor:
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
            #X1 = torch.baddbmm(A, X, X, beta=0.5, alpha=0.5)
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
            #X = torch.baddbmm(A, X, X, beta=0.5, alpha=0.5)
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