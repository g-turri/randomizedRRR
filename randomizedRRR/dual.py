from typing import Optional, Tuple
import logging
import numpy as np
from scipy.linalg import eig, eigh, LinAlgError, pinvh, qr
from scipy.sparse.linalg import eigs
from randomizedRRR.utils import topk, tonp, frnp
from randomizedRRR.linalg import modified_QR
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
    inv_dim = dim ** (-1.0)
    alpha = dim * tikhonov_reg
    K_reg = regularize(K_X, tikhonov_reg)
    K_reg_cholesky = cholesky(K_reg)
    l = rank + n_oversamples
    rng = np.random.default_rng(rng_seed)
    if optimal_sketching:
        Cov = inv_dim * K_Y
        Om = frnp(rng.multivariate_normal(np.zeros(dim), tonp(Cov, cuda), size=l).T, dtype=dtype, device=device)
    else:
        Om = frnp(rng.standard_normal(size=(dim, l)), dtype=dtype, device=device)

    for _ in range(iterated_power):
        # Powered randomized rangefinder
        Omp = cholesky_solve(Om, K_reg_cholesky)
        Om = (inv_dim * K_Y) @ (Om - alpha * Omp)
        Om, _ = qr(Om, mode="economic")

    Om = frnp(Om, dtype=dtype, device=device)
    KOm = cholesky_solve(Om, K_reg_cholesky)
    KOmp = Om - alpha * KOm

    F_0 = (Om.T @ KOmp)
    F_1 = (KOmp.T @ (inv_dim * (K_Y @ KOmp)))

    # Generation of matrices U and V.
    try:
        sigma_sq, Q = eigh(tonp(F_1, cuda), tonp(F_0, cuda))
    except LinAlgError:
        sigma_sq, Q = eig(pinvh(tonp(F_0, cuda)) @ tonp(F_1, cuda))

    Q = frnp(Q, device, dtype)

    Q_norm = torch.sum(Q.conj() * (F_0 @ Q), axis=0)
    Q = Q @ torch.diag(Q_norm ** -0.5)
    _idxs = topk(sigma_sq.real, rank).indices
    sigma_sq = sigma_sq.real

    Q = Q[:, _idxs.copy()]
    V = (dim ** 0.5) * (KOm @ Q)
    U = (dim ** 0.5) * (KOmp @ Q)
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
    # Find V via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow.
    # Prefer svd_solver == 'randomized' in such a case.
    if svd_solver == 'arnoldi':
        # Adding a small buffer to the Arnoldi-computed eigenvalues.
        sigma_sq, V = eigs(tonp(K, cuda), rank + 3, tonp(regularize(K_X, tikhonov_reg), cuda))
    else:  # 'full'
        sigma_sq, V = eig(tonp(K, cuda), tonp(regularize(K_X, tikhonov_reg), cuda))

    max_imag_part = np.max(V.imag)
    if max_imag_part >= 10.0 * V.shape[0] * np.finfo(V.dtype).eps:
        logging.warning(f"The computed projector is not real. The Kernel matrix is severely ill-conditioned.")
    V = np.real(V)

    # Post-process V. Promote numerical stability via additional QR decoposition if necessary.
    V = V[:, topk(sigma_sq.real, rank).indices]

    norm_inducing_op = (K_Xn @ (K_Xn.T)) + tikhonov_reg * K_X
    V, _, columns_permutation = modified_QR(V, M=tonp(norm_inducing_op, cuda), column_pivoting=True)
    V = V[:, np.argsort(columns_permutation)]
    if V.shape[1] < rank:
        logging.warning(
            f"The numerical rank of the projector is smaller than the selected rank ({rank}). {rank - V.shape[1]} "
            f"degrees of freedom will be ignored.")
        _zeroes = np.zeros((V.shape[0], rank - V.shape[1]))
        V = np.c_[V, _zeroes]
        assert V.shape[1] == rank
    V = frnp(V, device, dtype)
    U = K_X @ V
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
