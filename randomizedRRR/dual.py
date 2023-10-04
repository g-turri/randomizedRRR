from typing import Optional, Tuple
import logging
import numpy as np
from scipy.linalg import eig, eigh, LinAlgError, pinvh
from scipy.sparse.linalg import eigs
from randomizedRRR.utils import topk
from randomizedRRR.linalg import modified_QR
from utils import tonp, frnp
import torch
from torch import cholesky_solve
from torch.linalg import cholesky

def fit_rand_reduced_rank_regression_tikhonov(
        K_X: torch.Tensor,  # Kernel matrix of the input data
        K_Y: torch.Tensor,  # Kernel matrix of the output data
        tikhonov_reg: float,  # Tikhonov regularization parameter
        rank: int,  # Rank of the estimator
        n_oversamples: int,  # Number of oversamples
        optimal_sketching: bool,  # Whether to use optimal sketching (slower but more accurate) or not.
        iterated_power: int,  # Number of iterations of the power method
        rng_seed: Optional[int] = None,  # Seed for the random number generator (for reproducibility)
        _return_singular_values: bool = False
        # Whether to return the singular values of the projector. (Development purposes)
) -> Tuple[torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cuda = True if K_X.is_cuda else False
    dtype = K_X.dtype
    device = K_X.device
    dim = K_X.shape[0]
    inv_dim = dim ** -1
    l = rank + n_oversamples
    rng = np.random.default_rng(rng_seed)
    L = K_Y * inv_dim
    K = K_X * inv_dim
    K_reg = K + torch.eye(dim, dtype=K.dtype, device=device) * tikhonov_reg

    if optimal_sketching:
        Cov = L
        Om = torch.tensor(rng.multivariate_normal(np.zeros(dim), tonp(Cov, cuda), size=l).T, dtype=K_X.dtype, device=device)
    else:
        Om = torch.tensor(rng.standard_normal(size=(dim, l)), dtype=K_X.dtype, device = device)

    for _ in range(iterated_power):
        # Powered randomized rangefinder
        Omp = cholesky_solve(Om,cholesky(K_reg))
        Om = L @ (Om - tikhonov_reg * Omp)

    Omp = cholesky_solve(Om,cholesky(K_reg))
    KOmp = Omp.T @ K
    F_0 = KOmp @ Om
    Om = L @ (Om - tikhonov_reg * Omp)
    F_1 = KOmp @ Om

    # Generation of matrices U and V.
    try:
        sigma_sq, Q = eigh(tonp(F_1,cuda), tonp(F_0,cuda))
    except LinAlgError:
        sigma_sq, Q = eig(pinvh(tonp(F_0,cuda)) @ tonp(F_1,cuda))

    sigma_sq, Q = frnp(sigma_sq, device, dtype), frnp(Q, device, dtype)

    Q_norm = torch.sum(Q.conj() * (F_0 @ Q), axis=0)
    Q = Q @ torch.diag(Q_norm ** -0.5)
    _idxs = topk(tonp(sigma_sq.real, cuda), rank).indices
    sigma_sq = sigma_sq.real

    Q[:, frnp(_idxs,device,dtype).to(dtype=torch.int)]
    V = Omp @ Q
    U = K @ V
    if _return_singular_values:
        return U.real, V.real, sigma_sq
    else:
        return U.real, V.real

def fit_reduced_rank_regression_tikhonov(
        K_X: torch.Tensor,  # Kernel matrix of the input data
        K_Y: torch.Tensor,  # Kernel matrix of the output data
        tikhonov_reg: float,  # Tikhonov regularization parameter, can be 0
        rank: int,  # Rank of the estimator
        svd_solver: str = 'arnoldi',  # SVD solver to use. 'arnoldi' is faster but might be numerically unstable.
        _return_singular_values: bool = False
        # Whether to return the singular values of the projector. (Development purposes)
) -> Tuple[torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cuda = True if K_X.is_cuda else False
    dtype = K_X.dtype
    device = K_X.device

    dim = K_X.shape[0]
    rsqrt_dim = dim ** (-0.5)
    # Rescaled Kernel matrices
    K_Xn = K_X * rsqrt_dim
    K_Yn = K_Y * rsqrt_dim

    K = K_Yn @ K_Xn
    # Find U via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow.
    # Prefer svd_solver == 'randomized' in such a case.
    if svd_solver == 'arnoldi':
        # Adding a small buffer to the Arnoldi-computed eigenvalues.
        sigma_sq, U = eigs(tonp(K, cuda), rank + 3, tonp(regularize(K_X, tikhonov_reg), cuda))
    else:  # 'full'
        sigma_sq, U = eig(tonp(K, cuda), tonp(regularize(K_X, tikhonov_reg), cuda))

    max_imag_part = np.max(U.imag)
    if max_imag_part >= 2.2e-10:
        logging.warn(f"The computed projector is not real. The Kernel matrix is severely ill-conditioned.")
    U = np.real(U)

    # Post-process U. Promote numerical stability via additional QR decoposition if necessary.
    U = U[:, topk(sigma_sq.real, rank).indices]

    norm_inducing_op = (K_Xn @ (K_Xn.T)) + tikhonov_reg * K_X
    U, _, columns_permutation = modified_QR(U, M=tonp(norm_inducing_op, cuda), column_pivoting=True)
    U = U[:, np.argsort(columns_permutation)]
    if U.shape[1] < rank:
        logging.warn(
            f"The numerical rank of the projector is smaller than the selected rank ({rank}). {rank - U.shape[1]} "
            f"degrees of freedom will be ignored.")
        _zeroes = np.zeros((U.shape[0], rank - U.shape[1]))
        U = np.c_[U, _zeroes]
        assert U.shape[1] == rank
    U = frnp(U,device,dtype)
    V = K_X @ U
    if _return_singular_values:
        return U, V, sigma_sq
    else:
        return U, V

def regularize(M: torch.Tensor, reg: float):
    """Regularize a matrix by adding a multiple of the identity matrix to it.
    Args:
        M (ArrayLike): Matrix to regularize.
        reg (float): Regularization parameter.
    Returns:
        ArrayLike: Regularized matrix.
    """
    return M + reg * M.shape[0] * torch.eye(M.shape[0], dtype=M.dtype)