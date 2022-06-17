import torch
import pandas as pd
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

def deepstability25(data):
    fdi = FuzzedDataInterpreter(data)
    # needs to be positive definite
    test_input = torch.FloatTensor([
      [fdi.claim_float() for _ in range(3)],
      [fdi.claim_float() for _ in range(3)],
      [fdi.claim_float() for _ in range(3)]
    ])
    print(type(test_input))
    prediction = _precision_to_scale_tril(test_input)
    #print(test_input, prediction)
    return prediction


if __name__ == "__main__":
    runner = FunctionRunner(deepstability25)
    seed = [bytearray([0] * 12)]
    fuzzer = MutationFuzzer(seed, mutator=mutate_bytes)
    results = fuzzer.runs(runner, 5)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    print("fuzzer.failure_cases:")
    print(fuzzer.failure_cases)