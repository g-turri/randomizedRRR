from typing import Optional
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils import check_array


def weighted_norm(A: ArrayLike, M: Optional[ArrayLike] = None):
    r"""Weighted norm of the columns of A.

    Args:
        A (ndarray): 1D or 2D array. If 2D, the columns are treated as vectors.
        M (ndarray or LinearOperator, optional): Weigthing matrix. the norm of the vector :math:`a` is given by
        :math:`\langle a, Ma\rangle`. Defaults to None, corresponding to the Identity matrix. Warning: no checks are
        performed on M being a PSD operator.

    Returns:
        (ndarray or float): If ``A.ndim == 2`` returns 1D array of floats corresponding to the norms of
        the columns of A. Else return a float.
    """
    assert A.ndim <= 2, "'A' must be a vector or a 2D array"
    if M is None:
        norm = np.linalg.norm(A, axis=0)
    else:
        _A = np.dot(M, A)
        _A_T = np.dot(M.T, A)
        norm = np.real(np.sum(0.5 * (np.conj(A) * _A + np.conj(A) * _A_T), axis=0))
    return np.sqrt(norm)


def weighted_dot_product(A: ArrayLike, B: ArrayLike, M: Optional[ArrayLike] = None):
    """Weighted dot product between the columns of A and B. The output will be equivalent to :math:`A^{*} M B`
    if A and B are 2D arrays.

    Args:
        A, B (ndarray): 1D or 2D arrays.
        M (ndarray or LinearOperator, optional): Weigthing matrix. Defaults to None, corresponding to the
        Identity matrix. Warning: no checks are performed on M being a PSD operator.

    Returns:
        (ndarray or float): The result of :math:`A^{*} M B`.
    """
    assert A.ndim <= 2, "'A' must be a vector or a 2D array"
    assert B.ndim <= 2, "'B' must be a vector or a 2D array"
    A_adj = np.conj(A.T)
    if M is None:
        return np.dot(A_adj, B)
    else:
        _B = np.dot(M, B)
        return np.dot(A_adj, _B)


def _column_pivot(Q, R, k, squared_norms, columns_permutation):
    """
    Helper function to perform column pivoting on the QR decomposition at the k iteration. No checks are performed.
    For internal use only.
    """
    _arg_max = np.argmax(squared_norms[k:])
    j = k + _arg_max
    _in = [k, j]
    _swap = [j, k]
    # Column pivoting
    columns_permutation[_in] = columns_permutation[_swap]
    Q[:, _in] = Q[:, _swap]
    R[:k, _in] = R[:k, _swap]
    squared_norms[_in] = squared_norms[_swap]
    return Q, R, squared_norms, columns_permutation


def modified_QR(
    A: ArrayLike,
    M: Optional[ArrayLike] = None,
    column_pivoting: bool = False,
    rtol: float = 2.2e-16,
    verbose: bool = False,
):
    """Modified QR algorithm with column pivoting. Implementation follows the algorithm described in [1].

    Args:
        A (ndarray): 2D array whose columns are vectors to be orthogonalized.
        M (ndarray or LinearOperator, optional): PSD linear operator. If not None, the vectors are orthonormalized with
         respect to the scalar product induced by M. Defaults to None corresponding to Identity matrix.
        column_pivoting (bool, optional): Whether column pivoting is performed. Defaults to False.
        rtol (float, optional): relative tolerance in determining the numerical rank of A. Defaults to 2.2e-16.
        This parameter is used only when ``column_pivoting == True``.
        verbose (bool, optional): Whether to print informations and warnings about the progress of the algorithm.
        Defaults to False.

    Returns:
        tuple: A tuple of the form (Q, R), where Q and R satisfy A = QR. If ``column_pivoting == True``, the permutation
         of the columns of A is returned as well.

    [1] A. Dax: 'A modified Gram–Schmidt algorithm with iterative orthogonalization and column pivoting',
    https://doi.org/10.1016/S0024-3795(00)00022-7.
    """
    A = check_array(A)  # Ensure A is non-empty 2D array containing only finite values.
    num_vecs = A.shape[1]
    effective_rank = num_vecs
    dtype = A.dtype
    Q = np.copy(A)
    R = np.zeros((num_vecs, num_vecs), dtype=dtype)

    _roundoff = 1e-8  # From reference paper
    _tau = 1e-2  # From reference paper

    if (
        column_pivoting
    ):  # Initialize variables for fast pivoting, without re-evaluation of the norm at each step.
        squared_norms = weighted_norm(Q, M=M) ** 2
        max_norm = np.sqrt(np.max(squared_norms))
        columns_permutation = np.arange(num_vecs)

    for k in range(num_vecs):
        if column_pivoting:
            Q, R, squared_norms, columns_permutation = _column_pivot(
                Q, R, k, squared_norms, columns_permutation
            )
            norms_error_estimate = squared_norms * _roundoff
        if (
            k != 0
        ):  # Reorthogonalization of the column k+1 of A with respect to the previous orthonormal k vectors.
            alpha = weighted_dot_product(
                Q[:, :k], Q[:, k], M=M
            )  # alpha = Q[:,:k].T@M@Q[:,k]
            R[:k, k] += alpha
            Q[:, k] -= np.dot(Q[:, :k], alpha)

        # Numerical rank detection, performed only when column_pivoting == True
        norm_at_iter_k = weighted_norm(Q[:, k], M=M)
        if column_pivoting:
            if norm_at_iter_k < rtol * max_norm:
                effective_rank = k
                if verbose:
                    warn(
                        "Numerical rank of A has been reached with a relative tolerance rtol = {:.2e}. "
                        "Effective rank = {}. Stopping Orthogonalization procedure.".format(
                            rtol, effective_rank
                        )
                    )
                break
                # Normalization of the column k + 1
        R[k, k] = norm_at_iter_k
        Q[:, k] = Q[:, k] / R[k, k]
        # Orthogonalization of the remaining columns with respect to Q[:,k], i.e. the k+1 column of Q.
        if k < num_vecs - 1:
            R[k, k + 1 :] = weighted_dot_product(Q[:, k + 1 :], Q[:, k], M=M)
            Q[:, k + 1 :] -= np.outer(Q[:, k], R[k, k + 1 :])
            # Try fast update of the squared norms, recompute if numerical criteria are not attained.
            if column_pivoting:
                squared_norms[k + 1 :] -= (
                    R[k, k + 1 :] ** 2
                )  # Update norms using Phythagorean Theorem
                update_error_mask = (
                    _tau * squared_norms[k + 1 :] < norms_error_estimate[k + 1 :]
                )  # Check if the error estimate is too large
                if any(update_error_mask):
                    squared_norms[k + 1 :][update_error_mask] = weighted_norm(
                        Q[:, k + 1 :][:, update_error_mask], M=M
                    )  # Recompute the norms if necessary.
    if column_pivoting:
        return (
            Q[:, :effective_rank],
            R[:effective_rank],
            columns_permutation[:effective_rank],
        )
    else:
        return Q[:, :effective_rank], R[:effective_rank]
