"""
2.py: 
Patch: https://github.com/pytorch/pytorch/commit/dfc7fa03e5d33f909b9d7853dd001086f5d782a0 
Problem: Matrix inverse is numerically unstable, as a result numerical and analytical gradients for LU decomposition are too different. 
gradients for the LU decomposition calculation is unstable, lu_backward is impelemented as autograd torch.det is using LU in forward, 
while det_backward is using svd_backward (singular value decomposition). 
The issue with svd_backward is that it is only stable for inputs with distinct singular values. 
As a result, TestGradientsCuda::test_fn_gradgrad_linalg_det_cuda_float64 fails on Windows with GPU, which compares the numerical and analytical gradient. 
SVD_backward is only stable for ranks n - 1 <= r <= n with singular values sufficiently far away from each other.
"""
import torch
import pandas as pd

from pyfuzz.fuzzers import *
from pyfuzz.byte_mutations import *
from pyfuzz.fuzz_data_interpreter import *

def unstable_backward(ctx, LU_grad):
    #LU, pivots = ctx.saved_tensors
    LU, pivots = torch.lu(ctx)
    P, L, U = torch.lu_unpack(LU, pivots)

    assert (L is not None) and (U is not None)

    I = LU_grad.new_zeros(LU_grad.shape)
    I.diagonal(dim1=-2, dim2=-1).fill_(1)
    
    Lt_inv = torch.triangular_solve(I, L, upper=False).solution.transpose(-1, -2)
    Ut_inv = torch.triangular_solve(I, U, upper=True).solution.transpose(-1, -2)

    phi_L = (L.transpose(-1, -2) @ LU_grad).tril_()
    phi_L.diagonal(dim1=-2, dim2=-1).fill_(0.0)
    phi_U = (LU_grad @ U.transpose(-1, -2)).triu_()

    self_grad_perturbed = Lt_inv @ (phi_L + phi_U) @ Ut_inv
    return P @ self_grad_perturbed, None, None

def stable_backward(ctx, LU_grad):
    #LU, pivots = ctx.saved_tensors
    LU, pivots = torch.lu(ctx)
    P, L, U = torch.lu_unpack(LU, pivots)

    assert (L is not None) and (U is not None) and (P is not None)

    phi_L = (L.transpose(-1, -2).conj() @ LU_grad).tril_()
    phi_L.diagonal(dim1=-2, dim2=-1).fill_(0.0)

    phi_U = (LU_grad @ U.transpose(-1, -2).conj()).triu_()
    phi = phi_L + phi_U

    X = torch.triangular_solve(phi, L.transpose(-1, -2).conj(), upper=True).solution
    A_grad = torch.triangular_solve(X.transpose(-1, -2).conj() @ P.transpose(-1, -2), U, upper=True) \
        .solution.transpose(-1, -2).conj()

    return A_grad, None, None

def deepstability2(data):
    fdi = FuzzedDataInterpreter(data)

    matrix_size = 3
    """
    test_input = torch.FloatTensor([
      [fdi.claim_float() for _ in range(matrix_size)] ,
      [fdi.claim_float() for _ in range(matrix_size)] ,
      [fdi.claim_float() for _ in range(matrix_size)]
    ])

    grad_input = torch.FloatTensor([
      [fdi.claim_float() for _ in range(matrix_size)] ,
      [fdi.claim_float() for _ in range(matrix_size)] ,
      [fdi.claim_float() for _ in range(matrix_size)]
    ])

    print(test_input)

    print(grad_input)
    """
    for _ in range(9):
      print('Number:', fdi.claim_float())
    return

    #pred_s, _, _ = stable_backward(test_input, grad_input)

    #return pred_s

def deepstability2_unstable(data):
    fdi = FuzzedDataInterpreter(data)

    matrix_size = 3
    """
    test_input = torch.FloatTensor([
      [fdi.claim_float() for _ in range(matrix_size)] ,
      [fdi.claim_float() for _ in range(matrix_size)] ,
      [fdi.claim_float() for _ in range(matrix_size)]
    ])

    grad_input = torch.FloatTensor([
      [fdi.claim_float(), fdi.claim_float(), fdi.claim_float()] ,
      [fdi.claim_float(), fdi.claim_float(), fdi.claim_float()] ,
      [fdi.claim_float(), fdi.claim_float(), fdi.claim_float()]
    ])

    print(test_input)

    print(grad_input)
    """
    for _ in range(2):
      print('Number:', fdi.claim_float())
    return

    #pred_u, _, _ = unstable_backward(test_input, grad_input)

    #return pred_u


if __name__ == "__main__":
    runner = FunctionRunner(deepstability2)
    seed = [bytearray([0] * 12)]
    fuzzer = MutationFuzzer(seed, mutator=mutate_bytes)
    results = fuzzer.runs(runner, 1000)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    print("fuzzer.failure_cases:")
    print(fuzzer.failure_cases)

    runner = FunctionRunner(deepstability2_unstable)
    seed = [bytearray([0] * 12)]
    fuzzer = MutationFuzzer(seed, mutator=mutate_bytes)
    results = fuzzer.runs(runner, 1000)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    print("fuzzer.failure_cases:")
    print(fuzzer.failure_cases)