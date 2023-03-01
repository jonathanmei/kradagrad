from functools import  partial
import warnings
import numpy as np
import torch

from . import positive_matrix_functions as mf

class Kron:
    """
    A simple Kronecker factored square matrix
    """
    DEVICES_DTYPES = ('double', 'float', 'cuda', 'cpu')
    def __init__(self, mats):
        self._mats = tuple(mats)
        self._order = len(mats)
        self._tensor_shape = tuple(mat.shape[0] for mat in mats)

        # full matrix side length
        self._mat_size = np.prod(self._tensor_shape)
        for d_ in self.DEVICES_DTYPES:
            self.__setattr__(d_, partial(self._as_device_or_dtype, d_))

    def __matmul__(self, other):
        """
        Perform multiplication on matrices in batches
        (vectors must have trailing singleton dimension [..., 1])
        """
        assert (other.ndim >=2) and (other.shape[-2] == self._mat_size), "size mismatch"
        # allow mult two Kron objects
        if isinstance(other, KronDiag):
            res = Kron([
                mat_this * diag_other for (mat_this, diag_other)
                in zip(self._mats, other._mats)
            ])
        elif isinstance(other, Kron):
            res = Kron([
                mat_this @ mat_other for (mat_this, mat_other)
                in zip(self._mats, other._mats)
            ])
        elif isinstance(other, torch.Tensor):
            other = other.transpose(-1,-2)
            batch_shape = other.shape[:-1]
            batch_ix_end = len(batch_shape)
            res_ = other.reshape(batch_shape + self._tensor_shape)
            for i in range(self._order):
                res_ = torch.tensordot(res_, self._mats[i].type(res_.dtype), dims=[[batch_ix_end], [1]])
            res_ = res_.reshape(batch_shape + (self._mat_size,))
            res = res_.transpose(-1,-2)
        else:
            raise TypeError(f"Kron can't multiply {type(other)}")
        return res
    
    def __rmatmul__(self, other):
        if isinstance(other, torch.Tensor):
            return self.__matmul__(other.transpose(-2, -1)).transpose(-2, -1)
        else:
            return NotImplemented

    def __neg__(self):
        return Kron(tuple(-mat for mat in self._mats))

    @property
    def shape(self):
        return (self._mat_size, ) * 2

    @property
    def T(self):
        return Kron(tuple(mat.T for mat in self._mats))

    @property
    def ndim(self):
        return 2

    def pow(self, power):
        return Kron(tuple(mf.mat_pow(mat, power) for mat in self._mats))
    
    def eig(self):
        return self._eig()

    def eigh(self):
        return self._eig(symm=True)

    def _eig(self, symm=False):
        """
        Kron objects representing the eigvals and eigvectors
        """
        eiger = torch.linalg.eigh if symm else torch.linalg.eig
        vals_vecs = [eiger(mat) for mat in self._mats]
        return (
            KronDiag(tuple(vv_[0] for vv_ in vals_vecs)),
            Kron(tuple(vv_[1] for vv_ in vals_vecs))
        )

    def as_mat(self):
        """
        mostly for debugging
        """
        res = torch.Tensor([1]).to(self._mats[0].device).type(self._mats[0].dtype)
        for mat in self._mats:
            res = torch.kron(res, mat)
        return res

    def _as_device_or_dtype(self, fstr):
        if fstr not in self.DEVICES_DTYPES:
            raise ValueError(f"bad argument of `_as_device_or_dtype`: {fstr}")
        return Kron(tuple(mat.__getattribute__(fstr)() for mat in self._mats))

class KronDiag(Kron):
    def __matmul__(self, other):
        assert (other.ndim >=2) and (other.shape[-2] == self._mat_size), "size mismatch"
        # allow mult two Kron objects
        if isinstance(other, KronDiag):
            res = KronDiag([
                diag_this * diag_other for (diag_this, diag_other)
                in zip(self._mats, other._mats)
            ])
        elif isinstance(other, Kron):
            res = Kron([
                diag_this[:, None] * mat_other for (diag_this, mat_other)
                in zip(self._mats, other._mats)
            ])
        elif isinstance(other, torch.Tensor):
            other = other.transpose(-1,-2)
            batch_shape = other.shape[:-1]
            res_ = other.reshape(batch_shape + self._tensor_shape)
            for i in range(self._order):
                slc_ = (slice(None),) + (None, ) * (self._order - i - 1)
                res_ = self._mats[i][slc_] * res_
            res_ = res_.reshape(batch_shape + (-1,))
            res = res_.transpose(-1,-2)
        else:
            raise TypeError(f"Kron can't multiply {type(other)}")
        return res

    def __pow__(self, power):
        return KronDiag(tuple(diag ** power for diag in self._mats))
    
    def as_vec(self):
        """
        Return diagonal as a column vector (with singleton last dimension [..., 1])
        """
        res = super().as_mat()
        return res[:, None]

def lyapunov_reg(A, C, lam=1e0, X0=None, eps=1e-8):
    """
    Solves dense regularized symmetric Lyapunov problem
        1/2 |A X + X A^T - C|_F^2 + lam |X - X0|_F^2
     => (A^2 + lam/2 I) X + X (A^2 + lam/2 I) + 2 A X A = A C + C A + lam X0
    A bit more robust than `lyapunov`, allowing a wider range of `lam`
    """
    epsI = eps * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    S, U = torch.linalg.eigh(A + epsI)
    Uinv = U.transpose(-2, -1)
    S = torch.maximum(S, eps * torch.ones_like(S))
    AC = A @ C
    F = Uinv @ (_symm2(AC) + lam * X0).type_as(U) @ U
    Slam = S ** 2 + lam/2
    W = Slam[..., None] + Slam[..., None, :] + 2 * S[..., None] * S[..., None, :]
    Y = F / W
    X = U @ Y @ Uinv
    return X

def lyapunov(A, C, asymm=False, lam=1e0, X0=None, eps=1e-8):
    """
    Solves dense regularized Lyapunov problem
        A X + X A^T + lam X = C + lam X0
     => (A + lam/2 I) X + X (A^T + lam/2 I) = C + lam X0
    where A is assumed symmetric if `asymm` is False.
    """
    lamI = (eps + lam/2)*torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    if asymm:
        S, U = torch.linalg.eig(A + lamI)
        Uinv = U.inverse()
    else:
        S, U = torch.linalg.eigh(A + lamI)
        Uinv = U.transpose(-2, -1)
        S = torch.maximum(S, eps * torch.ones_like(S))
    F = Uinv @ (C + lam * X0).type_as(U) @ U
    W = S[..., None] + S[..., None, :]
    Y = F / W
    X = U @ Y @ Uinv
    if asymm:
        X = X.real
    return X

def _skew(A):
    return A/2 - A.T/2

def _symm2(A):
    return A + A.T

## low rank greedy riccati solver
# ref 1) https://www-user.tu-chemnitz.de/~benner/pub/blp-nlaa.pdf
# ref 2) https://sma.epfl.ch/~anchpcommon/publications/truncatedlr.pdf

# Alg 1 from [2]
# Testing to see if it works. Unused
# Note: >4 iterations seems to diverge, <4 is not accurate
@torch.no_grad()
def kron_lyap_greedy(A: Kron, C: torch.Tensor, rank=None, U: Kron=None, S: torch.Tensor=None, max_it=4):
    """
    Given positive definite A (optionally its eigendecomp U S U^T), solves
        A X + X A = C C^T
    for low rank X ~ Z Z^T using energy norm from [2]
    """
    C_neg0 = torch.zeros_like(C)[..., :0]
    C_neg1 = torch.zeros_like(C_neg0)
    Q =  torch.zeros_like(C_neg0)
    Z = torch.zeros_like(C_neg0)
    if rank is None:
        rank = C.shape[-1]
    if U is None or S is None:
        S, U = A.eigh()
        S = S.as_vec()
    for r_ in range(rank):
        z = torch.randn(C.shape[-2], 1, device=C.device, dtype=C.dtype)
        z = z - Q @ (Q.T @ z)
        for i_ in range(max_it):
            z_norm = torch.sqrt(z.T @ z)
            z = z / (z_norm + 1e-8)
            S_hat_inv = 1 / (S + z.T @ (A @ z))
            z_ = U @ (S_hat_inv * (U.T @ (
                C @ (C.T @ z) - C_neg0 @ (C_neg1.T @ z)
            )))
            z_ = z_ / (torch.sqrt(mf.matrices_norm(z_, 'fro')) + 1e-8)
            z = z_
        Q, _ = torch.linalg.qr(torch.cat([Q, z], -1))
        Z = torch.cat([Z, z], -1)
        C_neg0 = torch.cat([C_neg0, z, A @ z], -1)
        C_neg1 = torch.cat([C_neg1, A @ z, z], -1)
    return Z

# Adding B B^T: this is what we use in Newton's method
# Note: >4 iterations seems to diverge, <4 is not accurate
@torch.no_grad()
def kron_lyap_lr_greedy_energy(A: Kron, B: torch.Tensor, C: torch.Tensor, Z0: torch.Tensor=None, rank=None, U: Kron=None, S: torch.Tensor=None, max_it=2):
    """
    Given positive definite A (optionally its eigendecomp U S U^T), solves
        (A + B B^T) X + X (A + B B^T) = C C^T
    for low rank X ~ Z Z^T using [2], which treats
        f: X |--> (A+BB^T)X + X(A+BB^T)
    as a linear operator that induces a norm to define a loss function
    """
    Z = torch.zeros_like(C)[..., :0] if Z0 is None else Z0
    if rank is None:
        rank = C.shape[-1]
    if U is None or S is None:
        S, U = A.eigh()
        S = S.as_vec()
    def mul_AB(z):
        return A @ z + B @ (B.T @ z)
    def mul_C(z):
        return C @ (C.T @ z)
    def cost_fro(z, Az=None, mul_C=mul_C):
        Cz = mul_C(z)
        Az = Az if Az is not None else mul_AB(z)
        return (z.T @ z) * (Az.T @ Az) + (z.T @ Az) ** 2 - 2 * (Az.T @ Cz)

    # First, make sure initialization is helpful
    # Yes, this is useful (vs starting from scratch)
    cost_nothing = (C ** 2).sum()
    AZ = mul_AB(Z)
    keep_ix = tuple(cost_fro(Z[..., j:j+1], Az=AZ[..., j:j+1]) < cost_nothing for j in range(Z.shape[-1]))
    Z = Z[..., keep_ix]
    Q, R = torch.linalg.qr(Z)
    if Z.shape[-1]:
        R = kron_lyap_lr_galerkin(A, B, C, Q, R @ R.T)
        Z = Q @ R.type_as(Q)
    AZ = mul_AB(Z)
    C_neg0 = torch.cat([Z, AZ], -1)
    C_neg1 = torch.cat([AZ, Z], -1)

    # next, find remaining vecs to fill rank
    for r_ in range(Z.shape[-1], rank):
        z = torch.randn([C.shape[-2], 1], device=C.device, dtype=C.dtype)
        z = z - Q @ (Q.T @ z)
        def mul_Cj(z):
            return C @ (C.T @ z) - C_neg0 @ (C_neg1.T @ z)
        for i_ in range(max_it):
            z_norm = mf.matrices_norm(z, 'fro') 
            z = z * (1 / (z_norm + 1e-8))
            S_hat_inv = (S + (z.T @ mul_AB(z))[0, 0] + 1e-8) ** -1
            Cjz = mul_Cj(z)
            UtB = U.T @ B
            #ApIinvCCtz = U @ (S_hat_inv * (U.T @ Cjz))
            #BtApIinvB = UtB.T @ (S_hat_inv * UtB)
            #woodbury_mid = torch.eye(B.shape[-1], device=B.device, dtype=B.dtype) + BtApIinvB
            #woodbury_mid = UtB @ woodbury_mid.inverse()
            #woodbury =  U @ (S_hat_inv * woodbury_mid)
            #z = ApIinvCCtz - woodbury @ (B.T @ ApIinvCCtz)

            ## factor out the U (faster, no less accurate):
            UtApIinvCjz = S_hat_inv * (U.T @ Cjz)
            BtApIinvB = UtB.T @ (S_hat_inv * UtB)
            woodbury_mid = torch.eye(B.shape[-1], device=B.device, dtype=B.dtype) + BtApIinvB
            woodbury_leftmid = torch.linalg.solve(woodbury_mid, UtB, left=False)
            woodbury =  S_hat_inv * woodbury_leftmid
            z = U @ (UtApIinvCjz - woodbury @ (UtB.T @ UtApIinvCjz))
            #z = z / (torch.sqrt(mf.matrices_norm(z, 'fro')) + 1e-8)
            z = z * (1 / (torch.sqrt(torch.sqrt(z.T @ z)) + 1e-8))
        #Q, _ = torch.linalg.qr(torch.cat([Q, z], -1))
        # this is fine for accuracy since we will use full QR in Galerkin later anyway
        Q = torch.cat([Q, z - Q @ (Q.T @ z)], -1)
        Az = mul_AB(z)
        cost = cost_fro(z, Az, mul_C=mul_Cj)
        if cost > 0:
            #backtrack(cost_grad_fro, z, grad)
            #print(f'not helpful! quit at rank {r_}, additional cost {cost}, z_norm {z.T @ z}')
            return Z
        Z = torch.cat([Z, z], -1)
        C_neg0 = torch.cat([C_neg0, z, Az], -1)
        C_neg1 = torch.cat([C_neg1, Az, z], -1)
    return Z

# Standard, but also referenced in [2]
def kron_lyap_lr_galerkin(A: Kron, B: torch.Tensor, C: torch.Tensor, Q: torch.Tensor, R0: torch.Tensor=None, lam=1e-1):
    """
    Approx solves 
        (A + B B^T) X + X (A + B B^T) = C C^T
    in a reduced subspace X = Q R Q^T using dense small-scale regularized lyapunov.
    Returns R^{1/2}
    """
    BtQ = B.T @ Q
    CtQ = C.T @ Q
    Ap = Q.T @ (A @ Q) + (BtQ.T @ BtQ)
    Cp = CtQ.T @ CtQ
    R = lyapunov_reg(Ap, Cp, X0=R0, lam=lam)
    return mf.matrix_power_svd(R, 0.5, double=True).type_as(B)

# Alg 5 from [1]
@torch.no_grad()
def kron_riccati_lr_greedy_nm(A: Kron, C: torch.Tensor, rank: int=10, Z0: torch.Tensor=None, max_it: int=4, lam=1e-1, rank_inc=1):
    """
    solves riccati equation with symmetric A
        C C^T - A^T X - X A - X^2 = 0
    outputting low rank Z s.t. X = Z Z^T using newton's method,
    which solves a low-rank lyapunov equation in each iteration:
        F Y + Y F = G G^T
    where F = A + Z Z^T and G = [C, Z L]
        and L = cholesky(Z^T Z) (or any other factorization/sqrt)

    tweaks in notation from [1]:
        paper <= ours
        A <= -A
        B <= I
        C <= C^T
        Q <= I
        R <= I
    """
    if Z0 is None:
        Z0 = torch.zeros_like(C)[..., :0]
    Z = Z0
    # precompute
    S, U = A.eigh()
    S = S.as_vec()
    rank_inc = rank_inc if rank_inc is not None else C.shape[-1]
    init_rank = Z0.shape[-1]
    excess_rank = rank - init_rank
    loops = excess_rank // rank_inc + int(excess_rank % rank_inc > 0)
    if Z.shape[-1] > 0:
        # stable cholesky of Z^T Z:
        _, R = torch.linalg.qr(Z)
    for ii in range(loops):
        if Z.shape[-1] > 0:
            G = torch.cat([C, Z @ R.T], -1)
        else:
            G = C
        # note: don't overwrite Z yet! Used in galerkin as `B`
        Z0 = kron_lyap_lr_greedy_energy(A, Z, G, Z0=Z, rank=init_rank + (ii+1) * rank_inc if ii < loops-1 else rank, S=S, U=U, max_it=max_it)
        # recompute the whole QR for accuracy:
        Q, R = torch.linalg.qr(Z0)
        R = kron_lyap_lr_galerkin(A, Z, G, Q, R @ R.T, lam=lam).type_as(Q)
        Z = Q @ R
    return Z

# Refine existing solution using Riemannian 2nd order optimization
#  Rather slow if initialized randomly but converges very quickly near optimum
import pymanopt
def kron_riccati_riemann_refinement(A: Kron, C: torch.Tensor, Z0=None, max_it=10, outer_iter=2, inner_iter=20):
    n, k = Z0.shape[-2:]
    device = C.device
    mani = pymanopt.manifolds.PSDFixedRank(n, k)
    @pymanopt.function.pytorch(mani)
    def cost(Y):
        if isinstance(Y, np.ndarray):
            Y = torch.Tensor(Y)
        YtY = Y.T @ Y
        AY = A.cpu() @ Y
        s, k = [x.size()[-1] for x in [C, Y]]
        L = torch.cat([C.cpu(), AY, Y], -1)
        _, Lr = torch.linalg.qr(L, 'reduced')
        Lc, LAy, Ly = torch.split(Lr, [s, k, k], dim=-1)
        #Mat1 = -Lc @ Lc.T + _symm2(LAy @ Ly.T) + Ly @ YtY @ Ly.T
        Mat1 = -Lc @ Lc.T + _symm2((LAy  + Ly @ YtY / 2) @ Ly.T)
        return (Mat1 ** 2).sum() / 4
    prob = pymanopt.Problem(mani, cost)
    #truster = pymanopt.optimizers.trust_regions.TrustRegions(max_iterations=outer_iter, verbosity=0, log_verbosity=0)
    #opt_result = truster.run(prob, initial_point=Z0.cpu().numpy(), maxinner=inner_iter)
    line_searcher = pymanopt.optimizers.line_search.BackTrackingLineSearcher(max_iterations=max_it)
    truster = pymanopt.optimizers.steepest_descent.SteepestDescent(line_searcher=line_searcher, verbosity=0, log_verbosity=0)
    opt_result = truster.run(prob, initial_point=Z0.cpu().numpy())
    return torch.Tensor(opt_result.point).to(device)

# Combine everything! Fast greedy to initialize and Riemannian Newton to refine
def riccati_kron_solve(A: Kron, C: torch.Tensor, rank: int=10, greedy_iter: int=4, greedy_rank_inc: int=1, refine_max_it=10, refine_outer_iter: int=2, refine_inner_iter: int=20):
    Z = kron_riccati_lr_greedy_nm(A, C, rank=rank, max_it=greedy_iter, rank_inc=greedy_rank_inc)
    Z = kron_riccati_riemann_refinement(A, C, Z0=Z, max_it=refine_max_it, outer_iter=refine_outer_iter, inner_iter=refine_inner_iter)
    return Z

### Try diagonal matrix instead of Kron because diag mult is much faster. But it's bad for optimization... :(
@torch.no_grad()
def diag_lyap_lr_greedy(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, rank=None, max_it=6):
    """
    Given positive definite diagonal A (as column vector [n, 1]), solves
        (A + B B^T) X + X (A + B B^T) = C C^T
    for low rank X ~ Z Z^T using [2], which treats
        f: X |--> (A+BB^T)X + X(A+BB^T)
    as a linear operator that induces a norm to define a loss function
    """
    C_neg0 = torch.zeros_like(C)[..., :0]
    C_neg1 = torch.zeros_like(C_neg0)
    Q =  torch.zeros_like(C_neg0)
    Z = torch.zeros_like(C_neg0)
    if rank is None:
        rank = C.shape[-1]
    def mul_AB(z):
        return A * z + B @ (B.T @ z)
    for r_ in range(rank):
        z = torch.randn([C.shape[-2], 1], device=C.device, dtype=C.dtype)
        z = z - Q @ (Q.T @ z)
        def mul_CCt(z):
            return C @ (C.T @ z) - C_neg0 @ (C_neg1.T @ z)
        for i_ in range(max_it):
            z_norm = mf.matrices_norm(z, 'fro') 
            z = z / (z_norm + 1e-8)
            A_hat_inv = 1 / (A + z.T @ (mul_AB(z)) + 1e-8)
            CCtz = mul_CCt(z)
            ApIinvCCtz = A_hat_inv * CCtz
            BtApIinvB = B.T @ (A_hat_inv * B)
            woodbury_mid = torch.eye(B.shape[-1], device=B.device, dtype=B.dtype) + BtApIinvB
            woodbury_mid = B @ woodbury_mid.inverse()
            woodbury = A_hat_inv * woodbury_mid
            z_ = ApIinvCCtz - woodbury @ (B.T @ ApIinvCCtz)
            z_ = z_ / (torch.sqrt(mf.matrices_norm(z_, 'fro')) + 1e-8)
            z = z_
        Q, _ = torch.linalg.qr(torch.cat([Q, z], -1))

        Z = torch.cat([Z, z], -1)
        C_neg0 = torch.cat([C_neg0, z, mul_AB(z)], -1)
        C_neg1 = torch.cat([C_neg1, mul_AB(z), z], -1)
    return Z

@torch.no_grad()
def diag_riccati_lr_greedy_nm(A: torch.Tensor, C: torch.Tensor, rank: int=10, Z0: torch.Tensor=None, max_it: int=6):
    """
    solves riccati equation with diagonal A
        C C^T - A^T X - X A - X^2 = 0
    outputting low rank Z s.t. X = Z Z^T using newton's method,
    which solves a low-rank lyapunov equation in each iteration:
        F Y + Y F = G G^T
    where F = A + Z Z^T and G = [C, Z L]
        and L = cholesky(Z^T Z) (or any other factorization/sqrt)

    tweaks in notation from [1]:
        paper <= ours
        A <= -A
        B <= I
        C <= C^T
        Q <= I
        R <= I
    """
    if Z0 is None:
        Z0 = torch.zeros_like(C)[..., :0]
    Z = Z0
    # precompute
    loops = rank // C.shape[-1] + int(rank % C.shape[-1] > 0)
    for ii in range(loops):
        if Z.shape[-1] > 0:
            # stable cholesky of Z^T Z:
            _, LZ = torch.linalg.qr(Z)
            G = torch.cat([C, Z @ LZ.T], -1)
        else:
            G = C
        Z = diag_lyap_lr_greedy(A, Z, G, rank=None if ii < loops-1 else rank, max_it=max_it)
    return Z

def diag_riccati_riemann_refinement(A: torch.Tensor, C: torch.Tensor, Z0=None, outer_iter=2, inner_iter=10):
    n, k = Z0.shape[-2:]
    device = C.device
    mani = pymanopt.manifolds.PSDFixedRank(n, k)
    @pymanopt.function.pytorch(mani)
    def cost(Y):
        if isinstance(Y, np.ndarray):
            Y = torch.Tensor(Y)
        YtY = Y.T @ Y
        AY = A.cpu() * Y
        s, k = [x.size()[-1] for x in [C, Y]]
        L = torch.cat([C.cpu(), AY, Y], -1)
        _, Lr = torch.linalg.qr(L, 'reduced')    
        Lc, LAy, Ly = torch.split(Lr, [s, k, k], dim=-1)
        Mat1 = -Lc @ Lc.T + _symm2((LAy  + Ly @ YtY / 2) @ Ly.T)
        return (Mat1 ** 2).sum() / 4
    prob = pymanopt.Problem(mani, cost)
    truster = pymanopt.optimizers.trust_regions.TrustRegions(max_iterations=outer_iter, verbosity=0, log_verbosity=0)
    opt_result = truster.run(prob, initial_point=Z0.cpu().numpy(), maxinner=inner_iter)
    return torch.Tensor(opt_result.point).to(device)

def riccati_diag_solve(A: torch.Tensor, C: torch.Tensor, rank: int=10, greedy_iter: int=6, refine_outer_iter: int=2, refine_inner_iter: int=10):
    Z = diag_riccati_lr_greedy_nm(A, C, rank=rank, max_it=greedy_iter)
    Z = diag_riccati_riemann_refinement(A, C, Z0=Z, outer_iter=refine_outer_iter, inner_iter=refine_inner_iter)
    return Z

## Riemannian (torch): slow; use base FixedRankPSD instead. Still slow, but manageable.
##      Find fast initialization?
# reference: https://arxiv.org/pdf/1312.4883.pdf
if False:
    import pymanopt

    class RiccatiManifold(pymanopt.manifolds.manifold.Manifold):
        """
        Quotient Manifold on which to solve
            A X + X A + X^2 = C^T C
        as X = Y Y^T
        for A: [n, n]
            Y: [n, k]
            C: [s, n]  (C is not used to determine geometry)

        From the paper, B B^T = I, simplifying a lot

        """
        def __init__(self, A: Kron, k: int):
            self._A = A
            self._Asq = A @ A.T
            self._R, self._U = self._Asq.eigh()
            self._R = self._R.as_vec()
            self._n = A.shape[0]
            self._k = k
            name = f"Quotient manifold of {self._n}x{self._n} psd matrices of rank {k}, \
                with metric tuned to solving for X in the Riccati Equation A X + X A + X^2 = C^T C"
            dimension = self._n * k - k * (k - 1) // 2
            # aux vars
            self._point = None
            self._YtY = None
            self._M1 = None
            self._M2 = None
            self._M1_chol = None
            self._YtY_inv = None
            self._M1inv_M2 = None

            # Gradients
            self._Ge = None  # recompute riemannian only if euclidean changes, so we need to track euclidean
            self._Gr = None
            super().__init__(name, dimension)
        
        @property
        def typical_dist(self):
            return 10 * self._k
        
        @torch.no_grad()
        def _preprocess(self, Y):
            # cache values that can be reused at the current manifold point Y
            Y = torch.Tensor(Y)
            if self._point is None or not torch.allclose(self._point, Y):
                # recompute aux to prepare for other computations
                self._point = Y
                self._YtY = Y.T @ Y
                self._YtYsq = self._YtY @ self._YtY
                self._AsqY = self._Asq @ Y
                self._A1Y = self._AsqY + Y @ self._YtYsq
                self._M1 = self._YtY  # less descriptive name, but match the paper
                self._M2 = Y.T @ self._AsqY + self._YtY @ self._YtYsq
                self._M1_chol = torch.linalg.cholesky(self._M1)
                self._M1inv_M2 = torch.cholesky_solve(self._M2, self._M1_chol)
                self._YtY_inv = torch.cholesky_solve(torch.eye(self._k, device=Y.device, dtype=Y.dtype), self._M1_chol)


                # reset old Gradient
                self._Gr = None
                self._Ge = None

        @torch.no_grad()
        def _save_G(self, Ge, Gr):
            if self._Ge is None or not torch.allclose(self._Ge, Ge):
                self._Ge = Ge
            if self._Gr is None or not torch.allclose(self._Gr, Gr):
                self._Gr = Gr

        @torch.no_grad()
        def inner_product(self, point, vectora, vectorb):
            """
            aka the Riemannian metric
                a distance metric defined at `point` on manifold for two `vector` in tangent space
            """
            self._preprocess(point)
            Y = self._point  # alias
            # Break A1 in two to optimize matmul order (staying low rank):
            va_A1_vb_1 = vectora.T @ (self._Asq @ vectorb)
            Ytva = Y.T @ vectora
            Ytvb = Y.T @ vectorb
            va_A1_vb_2 = Ytva.T @ self._YtY @ Ytvb
            va_A1_vb = va_A1_vb_1 + va_A1_vb_2
            trace1 = (va_A1_vb * self._M2).sum()
            trace2 = ((vectora.T @ vectorb) * self._M2).sum()
            return trace1 + trace2

        @torch.no_grad()
        def norm(self, point, vector):
            return self.inner_product(point, vector, vector)

        # unused for trust region
        @torch.no_grad()
        def dist(self, pointa, pointb):
            """
            Note, this doesn't seem to be needed for Trust Region, and
            `log` is not implemented either
            """
            return self.norm(pointa, self.log(pointa, pointb))

        @torch.no_grad()
        def embedding(self, point, vector):
            return vector.numpy()  # needs to be numpy for pymanopt

        @torch.no_grad()
        def projection(self, point, vector):
            # According to the matlab reference, need to solve Lyapunov for Omega:
            #      M1 Omega M2 + M2 Omega M1 = RHS
            #   => Omega (M2/M1) + (M1\M2) Omega = M1\RHS/M1
            #      where M1, M2 are PD
            point = torch.Tensor(point)
            vector = torch.Tensor(vector)
            self._preprocess(point)

            RHS = _skew((self._A1Y.T @ vector) @ self._M1 + (point.T @ vector) @ self._M2)
            M1inv_RHS = torch.cholesky_solve(RHS, self._M1_chol)
            F = torch.cholesky_solve(M1inv_RHS.T, self._M1_chol).T
            Omega = lyapunov(self._M1inv_M2, F)
            return vector - point @ Omega

        to_tangent_space = projection

        @torch.no_grad()
        def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
            self._preprocess(point)
            # access cache if available:
            if self._Ge is not None and torch.allclose(self._Ge, euclidean_gradient):
                Gr = self._Gr
            else:
                Gr = self._lin_solv(euclidean_gradient)
                self._save_G(euclidean_gradient, Gr)
            return Gr

        @torch.no_grad()
        def euclidean_to_riemannian_hessian(
            self, point, euclidean_gradient, euclidean_hessian, tangent_vector
        ):
            self._preprocess(point)
            Y = self._point  # alias
            Eta = tangent_vector
            # Gr == riemannian gradient; uses cache if available
            Gr = self.euclidean_to_riemannian_gradient(point, euclidean_gradient)
            YtGr = Y.T @ Gr
            YtEta = Y.T @ Eta
            Y3Gr = self._YtY @ YtGr
            Y3Eta = self._YtY @ YtEta
            A1Gr = self._Asq @ Gr + Y @ Y3Gr
            GrtEta = Gr.T @ Eta
            A1dotGr = Eta @ Y3Gr + Y @ (_symm2(YtEta @ YtGr) + self._YtY @ GrtEta.T)
            M1dot = _symm2(YtEta)
            YtGrSym = _symm2(YtGr)
            GrtEtaSym = _symm2(GrtEta)
            YtEtaSymmYtYsq = M1dot @ self._YtYsq
            M2dot = _symm2(Eta.T @ self._AsqY) \
                + _symm2(YtEtaSymmYtYsq) \
                + self._YtYsq @ YtEtaSymmYtYsq
            
            T1 = A1dotGr @ self._M1 + A1Gr @ M1dot + Gr @ M2dot
            A1Eta = self._Asq @ Eta + Y @ Y3Eta
            YtGrSymYtY = YtGrSym @ self._YtY
            GrtEtaSymYtY = GrtEtaSym @ self._YtY 
            T2 = (Gr @ Y3Eta + Y @ (YtGrSym @ YtEta + self._YtY @ (Gr.T @ Eta))) @ self._M1 \
                + A1Eta @ YtGrSym + Eta @ (_symm2(Gr.T @ self._AsqY) + _symm2(YtGrSymYtY @ self._YtY) + self._YtY @ YtGrSymYtY)
            T3 = Y @ (_symm2(Gr.T @ A1Eta + YtGr @ self._M1 @ YtEta.T)) \
                + Gr @ (self._M1 @ YtEta.T @ self._YtY) + Eta @ (self._M1 @ YtGr.T @ self._YtY) \
                + self._AsqY @ GrtEtaSym + Y @ (
                    _symm2(GrtEtaSymYtY) + self._YtY @ GrtEtaSymYtY
                )
            RHS3 = torch.Tensor(euclidean_hessian) + (T2 - T1 - T3) / 2
            Hr = self._lin_solv(RHS3)  # Riemannian Hessian?
            return self.projection(point, Hr)

        @torch.no_grad()
        def retraction(self, point, vector, t=1):
            new_point = torch.Tensor(point) + t * vector
            self._preprocess(new_point)
            return new_point

        @torch.no_grad()
        def exp(self, *args, **kwargs):
            warnings.warn("Not a true exp; used retraction instead")
            return self.retraction(*args, **kwargs)

        # unused for trust region:
        @torch.no_grad()
        def log(self, pointa, pointb):
            """
            seems to be unnecessary for optimization
            """
            raise NotImplementedError("oops")

        @torch.no_grad()
        def random_point(self):
            return torch.randn(self._n, self._k)

        @torch.no_grad()
        def random_tangent_vector(self, point):
            eta = torch.randn(self._n, self._k)
            eta = self.projection(point, eta)
            nrm = self.norm(point, eta)
            return eta / (nrm + 1e-9)

        @torch.no_grad()
        def transport(self, point_a, point_b, vector_a):
            return self.projection(point_b, vector_a)

        @torch.no_grad()
        def zero_vector(self, point):
            return torch.zeros((self._n, self._k))

        @torch.no_grad()
        def _lin_solv(self, dir):
            """
            From paper, solution to 
                A1 eta M1 + eta M2 = dir
                A1 = A**2 + low_rank_stuff
                solve r independent linear equations
                    A1 + s_j I for j in range(k)
                Q_j = A**2 + s_j I
                Woodbury on Q_j + low rank

            Assumes self._preprocess has already been run!
            Nb. JMei doesn't yet understand _why_ this is necessary to define the
                Reimannian geometry, but _can_ explain how this solves the linear system
            """
            # R is a torch.Tensor of full diag elements
            # U is a Kron of the eigenvectors of A
            R, U = self._R, self._U
            S, V = torch.linalg.eig(self._M1inv_M2.T)
            dir = torch.Tensor(dir).type(V.dtype)
            Y = self._point.type(V.dtype)
            F = dir @ V
            UtF = U.T @ F
            w = [0] * self._k
            # Independent linear systems per column
            for j in range(self._k):
                UtF_j = UtF[..., j:j+1]  # keep the singleton dimension!!
                # compute Qj_inv
                eigs_inv = 1 / (R + S[j])
                # # can vectorize the last matmul:
                # QjinvF = U @ (eigs_inv * (U.T @ F_j))
                QjinvF = U @ (eigs_inv * UtF_j)
                YtQjinvF = Y.T @ QjinvF
                QjinvY = U @ (eigs_inv * (U.T @ Y))
                wood = self._YtY_inv + Y.T @ QjinvY
                w_j = QjinvF - QjinvY @ torch.linalg.solve(wood, YtQjinvF)
                w[j] = w_j
            W = torch.cat(w, -1)
            Y = F / W
            X = torch.linalg.solve(V, Y, left=False)
            return X.real

## Geoopt FixedRankPSD manifold (doesn't work with RiemannianAdam)
# Copied largely from PyManOpt
if False:
    import geoopt

    class PSDFixedRank(geoopt.manifolds.base.Manifold):
        """
        Represented as Y Y^T
        """
        def __init__(self, n, k, **kwargs):
            self._n = n
            self._k = k
            self.name = f"Quotient manifold of {n}x{n} psd matrices of rank {k}"
            self.ndim = 2
            super().__init__() 

        def dist(self, x, y, *unused, keepdim=False):
            return self.norm(x, self.logmap(x, y))
        
        def retr(self, x, u):
            return x + u
        
        def expmap(self, x, u):
            warnings.warn('using retr for exp')
            return self.retr(x, u)

        def transp(self, x, y, v):
            return self.proju(y, v)
        
        def logmap(self, x, y):
            u, _, vt = torch.linalg.svd(y.transpose(-2, -1) @ x)
            return y @ (u @ vt) - x

        def inner(self, x, u, v=None, keepdim=False, **unused):
            if v is None:
                v = u
            res = torch.tensordot(u, v, dims=[[-1,-2], [-1,-2]])
            if keepdim:
                res = res.unsqueeze(-1).unsqueeze(-1)
            return res

        def proju(self, x, u):
            xTx = x.transpose(-2,-1) @ x
            A = x.transpose(-2,-1) @ u
            Askew = A - A.transpose(-2, -1)
            Omega = lyapunov(xTx, Askew, asymm=False)
            return u - x @ Omega

        # TODO check pymanopt v geoopt
        def egrad2rgrad(self, x, u):
            #return u  # TODO pymanopt says this, but geoopt docs say:
            return self.proju(x, u)

        def projx(self, x):
            return x
        
        def _check_point_on_manifold(self, x, *unused, atol=1e-5, rtol=1e-5):
            return True, None
        
        def _check_vector_on_tangent(self, x, u, *unused, atol=1e-5, rtol=1e-5):
            return torch.allclose(u, self.proju(x, u), atol=atol, rtol=rtol), None

        def random(self, *size, dtype=None, device=None, **unused):
            return torch.randn(*size, dtype=dtype, device=device)    


## Low-rank Cholesky Newton Method (LRCF-NM)
# ref 1) https://www-user.tu-chemnitz.de/~benner/pub/blp-nlaa.pdf
if False:
    import pymor
    import pymor.bindings.pymess
    from pymor.algorithms import lradi, riccati
    from pymor.operators.interface import Operator, NumpyVectorSpace
  
    ## pymor (numpy) use pymess: crashes
    class Kron_pymor(Operator):
        """
        kronecker product of PD matrices
        """
        def __init__(self, mats):
            # own vars
            self.mats = mats
            self._inv_mats = None
            self._order = len(mats)
            self._tensor_shape = [mat.shape[0] for mat in mats]
            self._vec_size = np.prod(self._tensor_shape)
            # Operator abstract class vars
            self.linear = True
            self.source = NumpyVectorSpace(self._vec_size, id=0)
            self.range = NumpyVectorSpace(self._vec_size, id=0)
            super().__init__()
        
        def apply(self, vec, **unused):
            n_vec = len(vec)
            res = self.range.zeros(0)
            for k in range(n_vec):
                res_ = vec[k].to_numpy().reshape(self._tensor_shape)
                for i in range(self._order):
                    res_ = np.tensordot(res_, self.mats[i], axes=[[0], [1]])
                res.append(self.range.from_numpy(res_.ravel()))
            return res

        def apply_inverse(self, vec, **unused):
            if self._inv_mats is None:
                self._inv_mats = [np.linalg.inv(mat) for mat in self.mats]  # this can be optimized
            n_vec = len(vec)
            res = self.range.zeros(0)
            for k in range(n_vec):
                res_ = vec[k].to_numpy().reshape(self._tensor_shape)
                for i in range(self._order):
                    res_ = np.tensordot(res_, self._inv_mats[i], axes=[[0], [1]])
                res.append(self.range.from_numpy(res_.ravel()))
            return res
            

        def as_range_array(self, **unused):
            mat_ = np.array([1])
            for mat in self.mats:
                mat_ = np.kron(mat, mat_)
            return self.source.from_numpy(mat_)

    def kronecker_riccati_lr(mats, B):
        """
        Solves the low-rank riccati equation given by:
            A X + X A + X^2 = B B^T
        where A = \kron_prod mats_i

        Returns Z for which X = Z Z^T
        """
        A = Kron_pymor(mats)
        B = A.range.from_numpy(B.T)
        return _operator_riccati_lr(A, B).to_numpy().T

    def _operator_riccati_lr(A, B):
        """
        Solves the low-rank riccati equation given by:
            A X + X A + X^2 = B B^T
        Returns Z for which X = Z Z^T
        TODO: control tol
        TODO: control rank? 
        """
        C = A.source.from_numpy(np.eye(A.source.dim))
        #opts = pymor.bindings.pymess.ricc_lrcf_solver_options()['pymess_lrnm']
        opts = None
        return riccati.solve_ricc_lrcf(-A, None, B, C, options=opts)

    ## by hand
    def _shifts(A, C, tol=1e-3):
        E_ = pymor.operators.constructions.IdentityOperator(A.source)
        B_ = A.source.from_numpy(C.T)
        return lradi.wachspress_shifts_init(A, E_, B_, {'large_ritz_num': 5, 'small_ritz_num': 5, 'tol': tol})

    # doesn't work, it's hard
    # Alg 3 from [1]
    def _lrcf_lya(A, C, tol=1e-3, Z=None, S=None, U=None, A_pymor=None):
        """
        approximately solves
            (A + Z Z^T) X + X (A + Z Z^T) = -(C C^T + Z (Z^T Z) Z^T)
        using ADI with a single shift
        S, U = A.eigh()
        """
        if A_pymor is None:
            A_pymor = Kron_pymor([mat.cpu().numpy() for mat in A._mats])
        p_s = torch.Tensor(_shifts(A_pymor, Z.cpu().numpy(), tol=tol)).type(C.dtype)
        if U is None or S is None:
            S, U = A.eigh()
        ##init V
        Sp = S.as_vec() + p_s[0]  # has singleton dimension
        Spinv = 1/Sp
        ApIinv_CZ = U @ (Spinv * (U.T @ torch.cat([C, Z], -1)))
        wood_mid = (Z.T @ ApIinv_CZ[..., C.shape[-1]:] + torch.eye(Z.shape[1], device=Z.device, dtype=Z.dtype)).inverse()
        ZtZsqrt = mf.matrix_power_svd(Z.T @ Z, 1/2)
        V_tilde = torch.sqrt(torch.Tensor([-2*p_s[0]])) * torch.cat([
            ApIinv_CZ[..., :C.shape[-1]], ApIinv_CZ[..., C.shape[-1]:] @ ZtZsqrt
        ], -1)
        V = V_tilde - U @ (Spinv * (U.T @ (Z @ (wood_mid @ (Z.T @ V_tilde)))))
        Z1 = V
        for i in range(1, len(p_s)):
            Sp = S.as_vec() + p_s[i]  # has singleton dimension
            Spinv = 1/Sp
            ApIinv_V = U @ (Spinv * (U.T @ V))
            ApIinv_Z = U @ (Spinv * (U.T @ Z))
            wood_mid = (Z.T @ ApIinv_Z + torch.eye(Z.shape[1], device=Z.device, dtype=Z.dtype)).inverse()

            #(F + p I)\ V_{i-1} : 
            V1 = ApIinv_V - U @ (Spinv * (U.T @ (Z @ (wood_mid @ (Z.T @ ApIinv_V))))) 
            V = torch.sqrt(p_s[i]/p_s[i-1])*(V - (p_s[i] + p_s[i-1])*V1)
            Z1 = torch.cat([Z1, V], -1)
        return Z1

    # Alg 5 from [1]
    def _lrcf_nm(A, C, rank=10, Z0=None):
        """
        solves
            C^T C + A^T X + X A - X^2 = 0
        outputting low rank Z s.t. X = Z Z^T

        ours <= paper's
        B <= I
        Q <= I
        R <= I
        """
        #F = A - Z0 @ Z0.T
        if Z0 is None:
            Z0 = torch.randn(C.shape, device=C.device, dtype=C.dtype)
        Z = Z0
        S, U = A.eigh()
        A_pymor = Kron_pymor([mat.cpu().numpy() for mat in A._mats])
        Z = _lrcf_lya(A, C, Z=Z, S=S, U=U, A_pymor=A_pymor)
        # TODO postprocess for rank
        return Z

    def lrcf_nm(A: Kron, C: torch.Tensor, rank: int=10, Z0: torch.Tensor=None):
        """
        solves
            A X + X A + X^2 = C C^T
        for symm Kronecker factored A, outputting low rank Z s.t. X = Z Z^T
        """
        return _lrcf_nm(-A, C, rank=rank, Z0=Z0)

