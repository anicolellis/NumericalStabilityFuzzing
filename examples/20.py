"""
20.py: (INCOMPLETE) 
Patch: https://github.com/pytorch/pytorch/commit/470c496eb224bdd735eea1accf7269dfdd87d49f 
Problem: In multivariate normal distribution class, there is a function for computing the precision matrix that uses inverse, which is numerically unstable
"""
import torch
import math
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, lazy_property
from pyfuzz.fuzzers import *
from pyfuzz.byte_mutations import *
from pyfuzz.fuzz_data_interpreter import *


def _precision_to_scale_tril(P):
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    Lf = torch.cholesky(torch.flip(P, (-2, -1)))
    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    L = torch.triangular_solve(torch.eye(P.shape[-1], dtype=P.dtype, device=P.device),
                               L_inv, upper=False)[0]
    return L

class Unstable_MultivariateNormal(Distribution):
    """
    Creates a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.
    The multivariate normal distribution can be parameterized either
    in terms of a positive definite covariance matrix :math:`\mathbf{\Sigma}`
    or a positive definite precision matrix :math:`\mathbf{\Sigma}^{-1}`
    or a lower-triangular matrix :math:`\mathbf{L}` with positive-valued
    diagonal entries, such that
    :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`. This triangular matrix
    can be obtained via e.g. Cholesky decomposition of the covariance.
    Example:
        >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        tensor([-0.2102, -0.5429])
    Args:
        loc (Tensor): mean of the distribution
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal
    Note:
        Only one of :attr:`covariance_matrix` or :attr:`precision_matrix` or
        :attr:`scale_tril` can be specified.
        Using :attr:`scale_tril` will be more efficient: all computations internally
        are based on :attr:`scale_tril`. If :attr:`covariance_matrix` or
        :attr:`precision_matrix` is passed instead, it is only used to compute
        the corresponding lower triangular matrices using a Cholesky decomposition.
    """
    arg_constraints = {'loc': constraints.real_vector,
                       'covariance_matrix': constraints.positive_definite,
                       'precision_matrix': constraints.positive_definite,
                       'scale_tril': constraints.lower_cholesky}
    support = constraints.real
    has_rsample = True
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) + (scale_tril is not None) + (precision_matrix is not None) != 1:
            raise ValueError("Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified.")
        loc_ = loc.unsqueeze(-1)  # temporarily add dim on right
        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.scale_tril, loc_ = torch.broadcast_tensors(scale_tril, loc_)
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.covariance_matrix, loc_ = torch.broadcast_tensors(covariance_matrix, loc_)
        else:
            if precision_matrix.dim() < 2:
                raise ValueError("precision_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.precision_matrix, loc_ = torch.broadcast_tensors(precision_matrix, loc_)
        self.loc = loc_[..., 0]  # drop rightmost dim
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super(Unstable_MultivariateNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)
        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            self._unbroadcasted_scale_tril = torch.cholesky(covariance_matrix)
        else:  # precision_matrix is not None
            self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)
    
    @lazy_property
    def precision_matrix(self):
            # TODO: use `torch.potri` on `scale_tril` once a backwards pass is implemented.
            scale_tril_inv = torch.inverse(self._unbroadcasted_scale_tril)
            return torch.matmul(scale_tril_inv.transpose(-1, -2), scale_tril_inv).expand(
                self._batch_shape + self._event_shape + self._event_shape)

class Stable_MultivariateNormal(Distribution):
    """
    Creates a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.
    The multivariate normal distribution can be parameterized either
    in terms of a positive definite covariance matrix :math:`\mathbf{\Sigma}`
    or a positive definite precision matrix :math:`\mathbf{\Sigma}^{-1}`
    or a lower-triangular matrix :math:`\mathbf{L}` with positive-valued
    diagonal entries, such that
    :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`. This triangular matrix
    can be obtained via e.g. Cholesky decomposition of the covariance.
    Example:
        >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        tensor([-0.2102, -0.5429])
    Args:
        loc (Tensor): mean of the distribution
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal
    Note:
        Only one of :attr:`covariance_matrix` or :attr:`precision_matrix` or
        :attr:`scale_tril` can be specified.
        Using :attr:`scale_tril` will be more efficient: all computations internally
        are based on :attr:`scale_tril`. If :attr:`covariance_matrix` or
        :attr:`precision_matrix` is passed instead, it is only used to compute
        the corresponding lower triangular matrices using a Cholesky decomposition.
    """
    arg_constraints = {'loc': constraints.real_vector,
                       'covariance_matrix': constraints.positive_definite,
                       'precision_matrix': constraints.positive_definite,
                       'scale_tril': constraints.lower_cholesky}
    support = constraints.real
    has_rsample = True
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) + (scale_tril is not None) + (precision_matrix is not None) != 1:
            raise ValueError("Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified.")
        loc_ = loc.unsqueeze(-1)  # temporarily add dim on right
        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.scale_tril, loc_ = torch.broadcast_tensors(scale_tril, loc_)
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.covariance_matrix, loc_ = torch.broadcast_tensors(covariance_matrix, loc_)
        else:
            if precision_matrix.dim() < 2:
                raise ValueError("precision_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.precision_matrix, loc_ = torch.broadcast_tensors(precision_matrix, loc_)
        self.loc = loc_[..., 0]  # drop rightmost dim
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super(Stable_MultivariateNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)
        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            self._unbroadcasted_scale_tril = torch.cholesky(covariance_matrix)
        else:  # precision_matrix is not None
            self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)
    
    @lazy_property
    def precision_matrix(self):
        identity = torch.eye(self.loc.size(-1), device=self.loc.device, dtype=self.loc.dtype)
        # TODO: use cholesky_inverse when its batching is supported
        return torch.cholesky_solve(identity, self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape)

def deepstability20(data):
    fdi = FuzzedDataInterpreter(data)
    d = MultivariateNormal()

if __name__ == "__main__":
    runner = FunctionRunner(deepstability20)
    seed = [bytearray([0] * 12)]
    fuzzer = MutationFuzzer(seed, mutator=mutate_bytes)
    results = fuzzer.runs(runner, 1000)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    print("fuzzer.failure_cases:")
    print(fuzzer.failure_cases)