# cosine similarity
# code derived from 
import numpy as np
import pandas as pd
import math
import torch
import random

from pyfuzz.fuzzers import *
from pyfuzz.byte_mutations import *
from pyfuzz.fuzz_data_interpreter import *

float_max = 1000000.0
float_min = -1000000.0

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

def herbie_cos_similarity(u, v, eps):
    x = np.sum(np.multiply(u, v))
    y = np.sum(np.multiply(u, u))
    z = np.sum(np.multiply(v, v))
    n = y * z
    if n < (eps * eps):
        n = eps * eps
    n = math.pow(n, -0.5)
    result = x * n
    return result

"""
def unstable_cosine(data):
    fdi = FuzzedDataInterpreter(data)

    test1 = np.matrix([
      [float(i) for i in range(84)]
    ])
    test2 = np.matrix([
      [float(i) for i in range(84)]
    ])

    #print(test1, test2)

    result = unstable_cos_similarity(test1, test2, 1e-8)
    print(result)
    return result

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
    runner = FunctionRunner(unstable_cosine)
    #runner = FunctionRunner(stable_cosine)
    seed = [bytearray([0] * 12)]
    fuzzer = MutationFuzzer(seed, mutator=mutate_bytes)
    results = fuzzer.runs(runner, 500)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    print("fuzzer.failure_cases:")
    print(fuzzer.failure_cases)
"""

def create_1D(len, min, max):
  row = []
  for _ in range(len):
    row.append(random.uniform(min, max))
  return row

def create_2D(len1, len2, min, max):
  matrix = []
  for _ in range(len1):
    matrix.append(create_1D(len2, min, max))
  return matrix

if __name__ == "__main__":
  trials = 10
  length = 5
  ranges = [(-100.0, 100.0), (float_min, float_max)]

  for min, max in ranges:
    print('With min as ', min, ' and max as ', max)

    for _ in range(trials):
      #x = np.matrix(create_1D(length, min, max))
      x = [13.189142, 8.138781, -4.0982385, 5.143065]
      #y = np.matrix(create_1D(length, min, max))
      #y = x
      y = [13.188879, 8.138888, -4.0983186, 5.1430016]

      print(x, y)

      eps = 1e-12

      print('Unstable: ')
      print(unstable_cos_similarity(x, y, eps))

      print('Stable: ')
      print(stable_cos_similarity(x, y, eps))

      print('Herbie: ')
      print(herbie_cos_similarity(x, y, eps))

      print()