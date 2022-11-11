import sys

import torch

import cubic

def _matrices_info(A: torch.Tensor, norm: str='fro'):
    # check if A is 3D, return size and norm
    # `trace` is True, use trace instead of spectral norm
    A_sz = A.size()
    A_dim = A.dim()
    if A_dim != 3:
        caller = sys._getframe(1).f_code.co_name
        raise ValueError("{} supports 3D Tensors only! Input dim: {}".format(caller, A_dim))

    if norm == 'tr':
        A_norm = torch.stack([torch.trace(a) for a in A])
    elif norm == 'fro':
        A_norm = torch.stack([torch.linalg.matrix_norm(a) for a in A])
    else:  # singular value
        A_norm = (1 + 1e-6) * torch.stack([torch.linalg.matrix_norm(a, ord=2) for a in A])
    A_norm = A_norm[..., None, None]
    A_dev = A.device

    return A_sz, A_dim, A_norm, A_dev

def _batcher(batch_size: int, M: torch.Tensor):
    return M.unsqueeze(0).repeat([batch_size, 1, 1])

def matrix_sqrt_NS(A: torch.Tensor, iters: int=25, batched: bool=False, norm: str='fro') -> torch.Tensor:
    # 3D batch of matrices only
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
            #  t = argmin_t || ()^2 - (I - A) ||_F^2
            #    = argmin_t || C + L*t + Q*t^2 ||_F^2

            X1 = (A + X.bmm(X)) / 2
            Del = X1 - X
            XmI = X - eyes
            Del_norm = torch.linalg.matrix_norm(Del, ord=2)
            
            # this one arises from cancellation, not from t
            # so assign this before normalizing:
            con = 2 * Del  # C

            # for numerical stability of the cubic solver
            Del = Del / Del_norm[..., None, None]

            qua =  Del.bmm(Del)  # Q
            lin = Del.bmm(XmI)  # L
            lin = lin + lin.transpose(-2, -1)

            # TODO: compute coeffs
            a = 2 * (qua * qua).sum([-2, -1])
            b = 3 * (qua * lin).sum([-2, -1])
            c = 2 * (qua * con).sum([-2, -1]) + (lin * lin).sum([-2, -1])
            d = (lin * con).sum([-2, -1])
            t = cubic.solve_smallest(a, b, c, d, thr=Del_norm)
            low = (t < Del_norm)
            t[low] = Del_norm[low]
            max_norm = torch.fmax(1 - torch.linalg.matrix_norm(X, ord=2), Del_norm)
            t[t > max_norm] = max_norm[t > max_norm]  # trust region
            X = X + t[..., None, None] * Del
        else:
            X = (A + X.bmm(X)) / 2
    return  (eyes - X) * L_norm_sqrt
