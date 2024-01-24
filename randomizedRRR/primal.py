from typing import Optional
import numpy as np
from scipy.linalg import eigh, qr
from scipy.sparse.linalg import eigsh
from randomizedRRR.utils import topk, tonp, frnp
from randomizedRRR.linalg import weighted_norm
import torch
from torch import cholesky_solve
from torch.linalg import cholesky

def fit_reduced_rank_regression_tikhonov(
        C_X: torch.Tensor,  # Input covariance matrix
        C_XY: torch.Tensor,  # Cross-covariance matrix
        tikhonov_reg: float,  # Tikhonov regularization parameter, can be 0.0
        rank: int,  # Rank of the estimator
        svd_solver: str = 'arnoldi',  # SVD solver to use. Arnoldi is faster for low ranks.
        _return_singular_values: bool = False # Whether to return the singular values of the projector. (Development purposes)
):
    cuda = True if C_X.is_cuda else False
    dtype = C_X.dtype
    device = C_X.device

    dim = C_X.shape[0]
    reg_input_covariance = tonp(C_X + tikhonov_reg * torch.eye(dim, dtype=C_X.dtype), cuda)
    _crcov = C_XY @ (C_XY.T)
    if svd_solver == 'arnoldi':
        # Adding a small buffer to the Arnoldi-computed eigenvalues.
        values, vectors = eigsh(tonp(_crcov, cuda), rank + 3, M=reg_input_covariance)
    else:
        values, vectors = eigh(tonp(_crcov, cuda), reg_input_covariance)

    top_eigs = topk(values, rank)
    vectors = vectors[:, top_eigs.indices]
    # values = top_eigs.values

    _norms = weighted_norm(vectors, reg_input_covariance)

    _norms, vectors = frnp(_norms, device, dtype), frnp(vectors, device, dtype)

    vectors = vectors @ torch.diag(_norms ** (-1.0))
    if _return_singular_values:
        return vectors, np.sort(values)[::-1]
    else:
        return vectors

def fit_rand_reduced_rank_regression_tikhonov(
        C_X: torch.Tensor,  # Input covariance matrix
        C_XY: torch.Tensor,  # Cross-covariance matrix
        tikhonov_reg: float,  # Tikhonov regularization parameter
        rank: int,  # Rank of the estimator
        n_oversamples: int,  # Number of oversamples
        iterated_power: int,  # Number of power iterations
        rng_seed: Optional[int] = None,  # Random seed
        _return_singular_values: bool = False # Whether to return the singular values of the projector. (Development purposes)
):
    cuda = True if C_X.is_cuda else False
    dtype = C_X.dtype
    device = C_X.device

    dim = C_X.shape[0]
    reg_input_covariance = C_X + tikhonov_reg * torch.eye(dim, dtype=C_X.dtype)
    _crcov = C_XY @ (C_XY.T)
    rng = np.random.default_rng(rng_seed)
    sketch = frnp(rng.standard_normal(size=(reg_input_covariance.shape[0], rank + n_oversamples)), dtype=dtype, device=device)

    for _ in range(iterated_power):
        _tmp_sketch = cholesky_solve(sketch, cholesky(reg_input_covariance))
        sketch = _crcov @ _tmp_sketch
        sketch, _ = qr(sketch, mode="economic")

    sketch = frnp(sketch, dtype=dtype, device=device)
    sketch_p = cholesky_solve(sketch, cholesky(reg_input_covariance))

    F_0 = (sketch_p.T) @ sketch
    F_1 = (sketch_p.T) @ (_crcov @ sketch_p)

    values, vectors = eigh(tonp(F_1, cuda), tonp(F_0, cuda))
    _norms = weighted_norm(vectors, tonp(F_0, cuda))

    _norms, vectors = frnp(_norms, device, dtype), frnp(vectors, device, dtype)

    vectors = vectors @ torch.diag(_norms ** (-1.0))
    vectors = sketch_p @ vectors[:, frnp(topk(values, rank).indices, device, dtype).to(dtype=torch.int)]
    if _return_singular_values:
        return vectors, np.sort(values)[::-1]
    else:
        return vectors