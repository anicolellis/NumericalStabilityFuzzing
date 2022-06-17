"""
21.py: 
Patch: https://github.com/pytorch/pytorch/commit/071971476d7431a24e527bdc181981678055a95d 
Problem: Binomial distribution class encounters overflow when logits are large. Note: the binomial distribution is parametrized by logits
"""
import torch
import pandas as pd
from numbers import Number
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, probs_to_logits, lazy_property, logits_to_probs
from pyfuzz.fuzzers import *
from pyfuzz.byte_mutations import *
from pyfuzz.fuzz_data_interpreter import *


def _clamp_by_zero(x):
    # works like clamp(x, min=0) but has grad at 0 is 0.5
    return (x.clamp(min=0) + x - x.clamp(max=0)) / 2

class Binomial(Distribution):
    r"""
    Creates a Binomial distribution parameterized by :attr:`total_count` and
    either :attr:`probs` or :attr:`logits` (but not both). :attr:`total_count` must be
    broadcastable with :attr:`probs`/:attr:`logits`.
    Example::
        >>> m = Binomial(100, torch.tensor([0 , .2, .8, 1]))
        >>> x = m.sample()
        tensor([   0.,   22.,   71.,  100.])
        >>> m = Binomial(torch.tensor([[5.], [10.]]), torch.tensor([0.5, 0.8]))
        >>> x = m.sample()
        tensor([[ 4.,  5.],
                [ 7.,  6.]])
    Args:
        total_count (int or Tensor): number of Bernoulli trials
        probs (Tensor): Event probabilities
        logits (Tensor): Event log-odds
    """
    arg_constraints = {'total_count': constraints.nonnegative_integer,
                       'probs': constraints.unit_interval,
                       'logits': constraints.real}
    has_enumerate_support = True
    def __init__(self, total_count=1, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            self.total_count, self.probs, = broadcast_all(total_count, probs)
            self.total_count = self.total_count.type_as(self.logits)
            is_scalar = isinstance(self.probs, Number)
        else:
            self.total_count, self.logits, = broadcast_all(total_count, logits)
            self.total_count = self.total_count.type_as(self.logits)
            is_scalar = isinstance(self.logits, Number)
        self._param = self.probs if probs is not None else self.logits
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        self.probs = self.logits
        super(Binomial, self).__init__(batch_shape, validate_args=validate_args)

    def unstable_log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_factorial_n = torch.lgamma(self.total_count + 1)
        log_factorial_k = torch.lgamma(value + 1)
        log_factorial_nmk = torch.lgamma(self.total_count - value + 1)
        # Note that: torch.log1p(-self.probs)) = - torch.log1p(self.logits.exp()))
        return (log_factorial_n - log_factorial_k - log_factorial_nmk +
                value * self.logits - self.total_count * torch.log1p(self.logits.exp()))

    def stable_log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_factorial_n = torch.lgamma(self.total_count + 1)
        log_factorial_k = torch.lgamma(value + 1)
        log_factorial_nmk = torch.lgamma(self.total_count - value + 1)
        # k * log(p) + (n - k) * log(1 - p) = k * (log(p) - log(1 - p)) + n * log(1 - p)
        #     (case logit < 0)              = k * logit - n * log1p(e^logit)
        #     (case logit > 0)              = k * logit - n * (log(p) - log(1 - p)) + n * log(p)
        #                                   = k * logit - n * logit - n * log1p(e^-logit)
        #     (merge two cases)             = k * logit - n * max(logit, 0) - n * log1p(e^-|logit|)
        normalize_term = (self.total_count * _clamp_by_zero(self.logits)
                            + self.total_count * torch.log1p(torch.exp(-torch.abs(self.logits)))
                            - log_factorial_n)
        return value * self.logits - log_factorial_k - log_factorial_nmk - normalize_term

def unstable_deepstability21(data):
    fdi = FuzzedDataInterpreter(data)

    total_count = 1.
    x = torch.FloatTensor([
      [fdi.claim_probability() for _ in range(4)]
    ])
    u_prob = Binomial(total_count, logits=x).unstable_log_prob(x)

    return u_prob

def stable_deepstability21(data):
    fdi = FuzzedDataInterpreter(data)

    total_count = 1.

    x = torch.FloatTensor([
      [fdi.claim_probability() for _ in range(4)]
    ])
    s_prob = Binomial(total_count, logits=x).stable_log_prob(x)
    
    return s_prob

if __name__ == "__main__":
    runner = FunctionRunner(unstable_deepstability21)
    seed = [bytearray([0] * 12)]
    fuzzer = MutationFuzzer(seed, mutator=mutate_bytes)
    results = fuzzer.runs(runner, 1000)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    print("fuzzer.failure_cases:")
    print(fuzzer.failure_cases)

    runner = FunctionRunner(stable_deepstability21)
    results = fuzzer.runs(runner, 1000)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    print("fuzzer.failure_cases:")
    print(fuzzer.failure_cases)