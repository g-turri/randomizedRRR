import abc
import logging
from typing import Optional
import torch

class BaseKernel(abc.ABC):
    @abc.abstractmethod
    def __call__(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None):
        """
        Evaluate the kernel.
        Returns a torch tensor of shape (X.shape[0], Y.shape[0]).
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

class PolynomialKernel(BaseKernel):
    """
        Polynomial Kernel
        K(x, y) = (a<x, y> + b)^p

        where:
        a = scale
        b = bias
        p = degree
    """

    def __init__(self, scale=1, bias=0, degree=2):
        self.scale = scale
        self.bias = bias
        self.degree = degree

    @property
    def is_inf_dimensional(self):
        return False

    def __call__(self, X, Y=None, normalize=False):
        X = self._check_dims(X)
        Y = self._check_dims(Y)
        if Y is None:
            K = (torch.mul(X@X.T, self.scale) + self.bias) ** self.degree
            if normalize:
                D = torch.diag((K.diag() ** (1 / 2)) ** -1)
                K = D @ K @ D
        else:
            K = (torch.mul(X@Y.T, self.scale) + self.bias) ** self.degree
            if normalize:
                K_X = X @ X.T
                D_row = torch.diag(1.0 / torch.sqrt(torch.diag(K_X)))

                K_Y = Y @ Y.T
                D_col = torch.diag(1.0 / torch.sqrt(torch.diag(K_Y)))

                K = D_row @ K @ D_col

        return K

class LinearKernel(PolynomialKernel):
    """
        Linear Kernel
        K(X, Y) = <X, Y>
    """
    def __init__(self, scale=1, bias=0):
        super().__init__(scale=scale, bias=bias, degree=1)

class RBFKernel(BaseKernel):
    """
        RBF Kernel
    """

    def __init__(self, sigma=None):
        self.sigma = sigma

    @property
    def is_inf_dimensional(self):
        return True

    def _compute_sigma(self, X, Y):
        return torch.median(torch.cdist(X, Y))

    def __call__(self, X, Y=None):
        X = self._check_dims(X)
        Y = self._check_dims(Y)

        Xnorm = torch.sum(X ** 2, axis=-1)

        if Y is None:
            Y, Ynorm = X, Xnorm
        else:
            Ynorm = torch.sum(Y ** 2, axis=-1)

        if self.sigma == None:
            self.sigma = self._compute_sigma(X, Y)

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

    def _compute_sigma(self, X, Y):
        return torch.median(torch.cdist(X, Y))*2

    def __call__(self, X, Y=None):
        X = self._check_dims(X)
        Y = self._check_dims(Y)

        Xnorm = torch.sum(X ** 2, axis=-1)

        if Y is None:
            Y, Ynorm = X, Xnorm
        else:
            Ynorm = torch.sum(Y ** 2, axis=-1)

        if self.sigma == None:
            self.sigma = self._compute_sigma(X, Y)

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

        if self.L == None:
            logging.info("Laplacian matrix is not provided, setting L=0")
            self.L = torch.zeros(X.shape[1], device=X.device).to_sparse()

        if Y is None:
            K = X @ (torch.eye(X.shape[1], device=X.device).to_sparse() + self.rho * self.L) @ X.T
            if normalize:
                D = torch.diag((K.diag() ** (1 / 2)) ** -1)
                K = D @ K @ D
        else:
            K = X @ Y.T
            if normalize:
                K_X = X @ (torch.eye(X.shape[1], device=X.device).to_sparse() + self.rho * self.L) @ X.T
                D_row = torch.diag(1.0 / torch.sqrt(torch.diag(K_X)))

                K_Y = Y @ (torch.eye(X.shape[1], device=X.device).to_sparse() + self.rho * self.L) @ Y.T
                D_col = torch.diag(1.0 / torch.sqrt(torch.diag(K_Y)))

                K = D_row @ K @ D_col

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

        # Rowwise mean of input arrays & subtract from input arrays themselves
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