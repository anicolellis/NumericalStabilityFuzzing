# cosine similarity
# code derived from 
import numpy as np
import pandas as pd
import math
import torch

from pyfuzz.fuzzers import *
from pyfuzz.byte_mutations import *
from pyfuzz.fuzz_data_interpreter import *

# unstable
def unstable_cos_similarity(u, v, eps):
    x = np.sum(np.multiply(u, v))
    y = np.sum(np.multiply(u, u))
    z = np.sum(np.multiply(v, v))
    n = 1.0/(math.sqrt(y * z))
    if n > (1.0/eps):
        n = 1.0/eps
    result = x * n
    return result

# stable
def stable_cos_similarity(u, v, eps):
    x = np.sum(np.multiply(u, v))
    y = np.sum(np.multiply(u, u))
    z = np.sum(np.multiply(v, v))
    n = y * z
    if n < (eps * eps):
        n = eps * eps
    n = math.sqrt(n)
    result = x / n
    return result

def unstable_cosine(data):
    fdi = FuzzedDataInterpreter(data)

    test1 = np.matrix([
      [fdi.claim_float(), fdi.claim_float()] ,
      [fdi.claim_float(), fdi.claim_float()]
    ])
    test2 = np.matrix([
      [fdi.claim_float(), fdi.claim_float()] ,
      [fdi.claim_float(), fdi.claim_float()]
    ])

    return unstable_cos_similarity(test1, test2, 1e-8)

def stable_cosine(data):
    fdi = FuzzedDataInterpreter(data)

    test1 = np.matrix([
      [fdi.claim_float(), fdi.claim_float()] ,
      [fdi.claim_float(), fdi.claim_float()]
    ])
    test2 = np.matrix([
      [fdi.claim_float(), fdi.claim_float()] ,
      [fdi.claim_float(), fdi.claim_float()]
    ])

    print(test1)

    print(test2)

    return stable_cos_similarity(test1, test2, 1e-8)


if __name__ == "__main__":
    #runner = FunctionRunner(unstable_cosine)
    runner = FunctionRunner(stable_cosine)
    seed = [bytearray([0] * 12)]
    fuzzer = MutationFuzzer(seed, mutator=mutate_bytes)
    results = fuzzer.runs(runner, 1000)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    print("fuzzer.failure_cases:")
    print(fuzzer.failure_cases)