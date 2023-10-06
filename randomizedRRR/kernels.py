import abc
import logging
from typing import Optional
import numpy as np
import torch
from randomizedRRR.utils import tonp

class BaseKernel(abc.ABC):
    @abc.abstractmethod
    def __call__(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None):
        """
        Evaluate the kernel.
        Returns a numpy array of shape (X.shape[0], Y.shape[0]).
        """
        pass

    @property
    @abc.abstractmethod
    def is_inf_dimensional(self):
        pass

    def _check_dims(self, X: Optional[torch.Tensor] = None):
        if X is None:
            return None
        else:
            if X.ndim == 0:
                X = X[None, None]  # One sample, one feature
            elif X.ndim == 1:
                logging.info("Detected 1-dimensional input in kernel evaluation. The first dimension is _always_ "
                             "assumed to be the number of samples. If you want to evaluate a kernel on a single "
                             "sample, input X[None, :] as argument.")
                X = X[:, None]  # The first dimension is always assumed to be the number of samples
            else:
                pass
            return X

    def __repr__(self):
        _r = "[" + self.__class__.__name__ + "] "
        for k, v in self.__dict__.items():
            if k[0] == "_":
                pass
            else:
                _r += f"{k}: {v} "
        return _r

class LinearKernel(BaseKernel):
    """
        Linear Kernel
        K(X, Y) = <X, Y>
    """

    def __init__(self):
        pass

    @property
    def is_inf_dimensional(self):
        return False

    def __call__(self, X, Y=None, normalize=False):
        X = self._check_dims(X)
        Y = self._check_dims(Y)
        if Y is None:
            K = X @ X.T
        else:
            K = X @ Y.T

        if normalize:
            D = torch.diag((K.diag() ** (1/2)) ** -1)
            K = D @ K @ D

        return K

class RBFKernel(BaseKernel):
    """
        RBF Kernel
    """

    def __init__(self, sigma=None):
        self.sigma = sigma

    @property
    def is_inf_dimensional(self):
        return True

    def __call__(self, X, Y=None):
        X = self._check_dims(X)
        Y = self._check_dims(Y)

        Xnorm = torch.sum(X ** 2, axis=-1)

        if Y is None:
            Y, Ynorm = X, Xnorm
        else:
            Ynorm = torch.sum(Y ** 2, axis=-1)

        if self.sigma == None:
            self.sigma = find_sigma_matern(X, Y)

        A = Xnorm[:, None]
        B = Ynorm[None, :]
        C = torch.mm(X, Y.T)

        return torch.exp(-(A + B - 2 * C)/(2 * self.sigma**2))

class MaternKernel(BaseKernel):
    """
        Matern Kernel
    """

    def __init__(self, nu=0.5, sigma=None, tol=1e-6):
        self.nu = nu
        self.sigma = sigma
        self.tol = tol

    @property
    def is_inf_dimensional(self):
        return True

    def __call__(self, X, Y=None):
        X = self._check_dims(X)
        Y = self._check_dims(Y)

        Xnorm = torch.sum(X ** 2, axis=-1)

        if Y is None:
            Y, Ynorm = X, Xnorm
        else:
            Ynorm = torch.sum(Y ** 2, axis=-1)

        if self.sigma == None:
            self.sigma = find_sigma(X, Y)

        A = Xnorm[:, None]
        B = Ynorm[None, :]
        C = torch.mm(X, Y.T)

        if self.nu == 0.5:
            return torch.exp(-torch.sqrt(A + B - 2 * C + self.tol)/(self.sigma))
        else:
            logging.info("nu = " + self.nu + "is not supported")
            return None

class LaplacianKernel(BaseKernel):
    """
        Laplacian Kernel
    """

    def __init__(self, L=None, rho=1.0):
        self.L = L
        self.rho = rho

    @property
    def is_inf_dimensional(self):
        return False

    def __call__(self, X, Y=None, normalize=False):
        X = self._check_dims(X)
        Y = self._check_dims(Y)

        if Y is None:
            Y = X

        if self.L == None:
            logging.info("Laplacian matrix is not provided, setting L=0")
            self.L = torch.zeros(X.shape[1], device=X.device).to_sparse()

        K = X @ (torch.eye(X.shape[1], device=X.device).to_sparse() + self.rho * self.L) @ Y.T

        if normalize:
            D = torch.diag((K.diag() ** (1 / 2)) ** -1)
            K = D @ K @ D

        return K

class CorrelationKernel(BaseKernel):
    """
        Correlation Kernel
    """

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    @property
    def is_inf_dimensional(self):
        return False

    def __call__(self, X, Y=None):
        X = self._check_dims(X)
        Y = self._check_dims(Y)

        # Rowwise mean of input arrays & subtract from input arrays themeselves
        Xm = X - X.mean(0)[None, :]
        if Y is None:
            Ym = Xm
        else:
            Ym = Y - Y.mean(0)[None, :]

        # Sum of squares across rows
        SSX = (Xm ** 2).sum(1)
        SSY = (Ym ** 2).sum(1)

        # Finally get corr coeff
        r = torch.mm(Xm, Ym.T) / torch.sqrt(torch.mm(SSX[:, None], SSY[None]))
        r = torch.maximum(torch.minimum(r, torch.tensor(1.0)), torch.tensor(-1.0))

        return torch.exp(-self.gamma*(1-r))

def find_sigma(x1, x2, n=1000):
    '''Implementation of the median heuristic. See Gretton 2012
       Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
       in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
       and y of all distances between points from both data sets X and Y.
    '''
    cuda = True if x1.is_cuda else False
    siz = torch.min((n, x1.shape[0], x2.shape[0]))
    x1, x2 = x1[0:siz], x2[0:siz]

    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    k1 = torch.sum((x1 * x1), 1)
    q = torch.tile(k1, (m, 1)).T
    del k1

    k2 = torch.sum((x2 * x2), 1)
    r = torch.tile(k2, (n, 1))
    del k2

    h = q + r
    del q, r

    # The norm
    h = h - 2 * torch.mm(x1, x2.T)
    h = torch.tensor(h, dtype=float)

    mdist = torch.median(h.flatten())

    sigma = (mdist / 2.0) ** (1/2)
    if not sigma: sigma = 1

    return tonp(sigma, cuda)

def find_sigma_matern(x1, x2, n=1000):
    '''Implementation of the median heuristic. See Gretton 2012
       Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
       in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
       and y of all distances between points from both data sets X and Y.
    '''
    cuda = True if x1.is_cuda else False
    siz = torch.min((n, x1.shape[0], x2.shape[0]))
    x1, x2 = x1[0:siz], x2[0:siz]

    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    k1 = torch.sum((x1 * x1), 1)
    q = torch.tile(k1, (m, 1)).T
    del k1

    k2 = torch.sum((x2 * x2), 1)
    r = torch.tile(k2, (n, 1))
    del k2

    h = q + r
    del q, r

    # The norm
    h = h - 2 * torch.mm(x1, x2.T)
    h = torch.tensor(h, dtype=float)

    mdist = torch.median(h.flatten()) ** (1/2)

    sigma = mdist
    if not sigma: sigma = 1

    return tonp(sigma, cuda)

def find_sigma(x1, x2, n=1000):
    '''Implementation of the median heuristic. See Gretton 2012
       Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
       in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
       and y of all distances between points from both data sets X and Y.
    '''
    cuda = True if x1.is_cuda else False
    siz = np.min((n, x1.shape[0], x2.shape[0]))
    x1, x2 = x1[0:siz], x2[0:siz]

    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    k1 = torch.sum((x1 * x1), 1)
    q = torch.tile(k1, (m, 1)).T
    del k1

    k2 = torch.sum((x2 * x2), 1)
    r = torch.tile(k2, (n, 1))
    del k2

    h = q + r
    del q, r

    # The norm
    h = h - 2 * torch.mm(x1, x2.T)
    h = torch.tensor(h, dtype=float)

    mdist = torch.median(h.flatten())

    sigma = (mdist / 2.0) ** (1/2)
    if not sigma: sigma = 1

    return tonp(sigma, cuda)

def find_sigma_matern(x1, x2, n=1000):
    '''Implementation of the median heuristic. See Gretton 2012
       Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
       in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
       and y of all distances between points from both data sets X and Y.
    '''
    cuda = True if x1.is_cuda else False
    siz = np.min((n, x1.shape[0], x2.shape[0]))
    x1, x2 = x1[0:siz], x2[0:siz]

    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    k1 = torch.sum((x1 * x1), 1)
    q = torch.tile(k1, (m, 1)).T
    del k1

    k2 = torch.sum((x2 * x2), 1)
    r = torch.tile(k2, (n, 1))
    del k2

    h = q + r
    del q, r

    # The norm
    h = h - 2 * torch.mm(x1, x2.T)
    h = torch.tensor(h, dtype=float)

    mdist = torch.median(h.flatten()) ** (1/2)

    sigma = mdist
    if not sigma: sigma = 1

    return tonp(sigma, cuda)