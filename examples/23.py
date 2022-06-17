"""
23.py: (INCOMPLETE) 
Patch: https://github.com/pytorch/pytorch/commit/d16c8238e164c6499714de625eb73422382e5ec1 
Problem: Implementation of softmax for certain cases (when the dim argument of softmax and axis do not equal to ndim - 1, where ndim - 1 = the last dimension) is numerically unstable. 
Large inputs into the exponential function will produce infinity and output of softmax becomes NaN.
"""
import torch.onnx.symbolic_helper as sym_help
import pandas as pd
from pyfuzz.fuzzers import *
from pyfuzz.byte_mutations import *
from pyfuzz.fuzz_data_interpreter import *

def unstable_softmax(g, input, dim, dtype=None):
    input_dim = input.type().dim()
    if input_dim:
        if dim < 0:
            dim = input_dim + dim
        if input_dim == dim + 1:
            softmax = g.op('Softmax', input, axis_i=dim)
            if dtype and dtype.node().kind() != 'prim::Constant':
                parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
                softmax = g.op("Cast", softmax, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
            return softmax
    exp = g.op('Exp', input)
    sum = g.op('ReduceSum', exp, axes_i=[dim])
    softmax = g.op('Div', exp, sum)
    if dtype and dtype.node().kind() != 'prim::Constant':
        parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
        softmax = g.op("Cast", softmax, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return softmax

def stable_softmax(g, input, dim, dtype=None):
    input_dim = input.type().dim()
    if input_dim is not None:
        is_transpose_required = (input_dim != dim + 1)

        if is_transpose_required:
            axes = list(range(input_dim))
            axes[dim], axes[-1] = axes[-1], axes[dim]
            input = g.op("Transpose", input, perm_i=axes)
            dim = input_dim - 1

        softmax = g.op('Softmax', input, axis_i=dim)
        if dtype and dtype.node().kind() != 'prim::Constant':
            parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
            softmax = g.op("Cast", softmax, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])

        if is_transpose_required:
            softmax = g.op("Transpose", softmax, perm_i=axes)
        return softmax
    # Apply max normalization.
    input = g.op('Sub', input, g.op('ReduceMax', input, axes_i=[dim], keepdims_i=1))

    exp = g.op('Exp', input)
    sum = g.op('ReduceSum', exp, axes_i=[dim])
    softmax = g.op('Div', exp, sum)
    if dtype and dtype.node().kind() != 'prim::Constant':
        parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
        softmax = g.op("Cast", softmax, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return softmax

def deepstability23(data):
    fdi = FuzzedDataInterpreter(data)
    
def deepstability2_unstable(data):
    fdi = FuzzedDataInterpreter(data)

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
    results = fuzzer.runs(runner, 1000)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    print("fuzzer.failure_cases:")
    print(fuzzer.failure_cases)