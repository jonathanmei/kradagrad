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

def lyapunov(A, C, asymm=False):
    """
    Solves
        A X + X A^T = C
    where A is assumed symmetric if `asymm` is False.
    """
    if asymm:
        U, S, Vt = torch.linalg.svd(A)
    else:
        S, U = torch.linalg.eigh(A)
        Vt = U.transpose(-2, -1)

    F = U.transpose(-2, -1) @ C @ U
    W = S[..., None] + S[..., None, :]
    Y = F / W
    X = Vt.transpose(-2, -1) @ Y @ Vt
    return X

def _skew(A):
    return A/2 - A.T/2

def _symm2(A):
    return A + A.T

## lr- greedy
# ref 1) https://www-user.tu-chemnitz.de/~benner/pub/blp-nlaa.pdf
# ref 2) https://sma.epfl.ch/~anchpcommon/publications/truncatedlr.pdf

# Alg 1 from [2]
def lyap_kron_greedy(A: Kron, C: torch.Tensor, rank=None, U: Kron=None, S: torch.Tensor=None, max_it=4, tol=1e-1, skip_cost_check=True):
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
        err_old = torch.inf
        z = torch.randn(C.shape[-2], 1, device=C.device, dtype=C.dtype)
        z = z - Q @ (Q.T @ z)
        def cost(z):
            L = torch.cat([C, C_neg0, C_neg1, A @ z, z], -1)
            _, Lr = torch.linalg.qr(L)
            Lc, Lcn0, Lcn1, LAz, Lz = torch.split(Lr, [
                x.shape[-1] for x in [C, C_neg0, C_neg1, z, z]
            ], dim=-1)
            mat = -Lc @ Lc.T + _symm2(Lcn0 @ Lcn1.T) + _symm2(LAz @ Lz.T)
            return (mat ** 2).sum()
        for i_ in range(max_it):
            z_norm = mf.matrices_norm(z, 'fro') 
            z = z / (z_norm + 1e-8)
            S_hat_inv = 1 / (S + z.T @ (A @ z))
            z_ = U @ (S_hat_inv * (U.T @ (
                C @ (C.T @ z) - C_neg0 @ (C_neg1.T @ z)
            )))
            z_ = z_ / (torch.sqrt(mf.matrices_norm(z_, 'fro')) + 1e-8)
            if not skip_cost_check:
                err_new = cost(z_)
                if err_new > err_old * 2:
                    print(f"quit early after {i_} iters")
                    # reuse old z:
                    z = z * z_norm
                    break
                else:
                    err_old = err_new
            z = z_
        Q, _ = torch.linalg.qr(torch.cat([Q, z], -1))
        Z = torch.cat([Z, z], -1)
        C_neg0 = torch.cat([C_neg0, z, A @ z], -1)
        C_neg1 = torch.cat([C_neg1, A @ z, z], -1)
    return Z

# couldn't get Galerkin projection to work...

# Adding B B^T
def lyap_kron_lr_greedy(A: Kron, B: torch.Tensor, C: torch.Tensor, rank=None, U: Kron=None, S: torch.Tensor=None, max_it=4, tol=1e-1, skip_cost_check=True):
    """
    Given positive definite A (optionally its eigendecomp U S U^T), solves
        (A + B B^T) X + X (A + B B^T) = C C^T
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
    def mul_AB(z):
        return A @ z + B @ (B.T @ z)
    err_old_outer = torch.inf
    for r_ in range(rank):
        z = torch.randn([C.shape[-2], 1], device=C.device, dtype=C.dtype)
        z = z - Q @ (Q.T @ z)
        def cost(z):
            L = torch.cat([C, C_neg0, C_neg1, mul_AB(z), z], -1)
            _, Lr = torch.linalg.qr(L)
            Lc, Lcn0, Lcn1, LAz, Lz = torch.split(Lr, [
                x.shape[-1] for x in [C, C_neg0, C_neg1, z, z]
            ], dim=-1)
            mat = -Lc @ Lc.T + _symm2(Lcn0 @ Lcn1.T) + _symm2(LAz @ Lz.T)
            return (mat ** 2).sum()
        def mul_CCt(z):
            return C @ (C.T @ z) - C_neg0 @ (C_neg1.T @ z)
        err_old_inner = torch.inf
        for i_ in range(max_it):
            z_norm = mf.matrices_norm(z, 'fro') 
            z = z / (z_norm + 1e-8)
            S_hat_inv = 1 / (S + z.T @ (mul_AB(z)) + 1e-8)
            CCtz = mul_CCt(z)
            ApIinvCCtz = U @ (S_hat_inv * (U.T @ CCtz))
            BtApIinvB = B.T @ (U @ (S_hat_inv * (U.T @ B)))
            woodbury = U @ (S_hat_inv * (U.T @ B @ torch.linalg.solve(
                torch.eye(B.shape[-1], device=B.device, dtype=B.dtype) + BtApIinvB, B.T
            )))
            z_ = ApIinvCCtz - woodbury @ ApIinvCCtz
            z_ = z_ / (torch.sqrt(mf.matrices_norm(z_, 'fro')) + 1e-8)
            if not skip_cost_check:
                err_new_inner = cost(z_)
                if err_new_inner > err_old_inner * 2:
                    print(f"quit early after {i_} iters")
                    # reuse old z:
                    z = z * z_norm
                    break
                else:
                    err_old_inner = err_new_inner
            z = z_
        if not skip_cost_check:
            err_new_outer = cost(z)
            # check if this loss is decreasing
            if err_new_outer > err_old_outer:
                print(f'rank increase to {r_ + 1} hurts, quitting')
                return Z
            err_old_outer = err_new_outer
        #Q, _ = torch.linalg.qr(torch.cat([Q, z], -1))
        Q, _ = torch.linalg.qr(torch.cat([Q, z - Q @ (Q.T @ z)], -1))
        #Q = torch.cat([Q, z - Q @ (Q.T @ z)], -1) 

        Z = torch.cat([Z, z], -1)
        C_neg0 = torch.cat([C_neg0, z, mul_AB(z)], -1)
        C_neg1 = torch.cat([C_neg1, mul_AB(z), z], -1)
    return Z


# Alg 5 from [1]
def lr_greedy_nm(A: Kron, C: torch.Tensor, rank: int=10, Z0: torch.Tensor=None):
    """
    solves
        C C^T - A^T X - X A - X^2 = 0
    outputting low rank Z s.t. X = Z Z^T

    ours <= paper's
    A <= -A
    B <= I
    C <= C^T
    Q <= I
    R <= I
    """
    #F = A - Z0 @ Z0.T
    if Z0 is None:
        Z0 = torch.zeros_like(C)[..., :0]
    Z = Z0
    # precompute
    S, U = A.eigh()
    S = S.as_vec()
    loops = rank // C.shape[-1] + int(rank % C.shape[-1] > 0)
    for ii in range(loops):
        if Z.shape[-1] > 0:
            # stable cholesky of Z^T Z:
            _, LZ = torch.linalg.qr(Z)
            G = torch.cat([C, Z @ LZ.T], -1)
        else:
            G = C
        Z = lyap_kron_lr_greedy(A, Z, G, rank=None if ii < loops-1 else rank, S=S, U=U)
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
