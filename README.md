The examples folder contains snippets of torch code, with versions that are numerically stable and unstable.

The name of the file corresponds to the row in the DeepStability database.

2, 21, and 25 run, but they do not function properly. The code for 20 and 23 is incomplete.

2.py: 
  Patch: https://github.com/pytorch/pytorch/commit/dfc7fa03e5d33f909b9d7853dd001086f5d782a0
  Problem: Matrix inverse is numerically unstable, as a result numerical and analytical gradients for LU decomposition are too different.
    gradients for the LU decomposition calculation  is unstable, lu_backward is impelemented as autograd
    torch.det is using LU in forward, while det_backward is using svd_backward (singular value decomposition).
    The issue with svd_backward is that it is only stable for inputs with distinct singular values. As a result, TestGradientsCuda::test_fn_gradgrad_linalg_det_cuda_float64 fails on Windows with GPU, which compares the numerical and analytical gradient. SVD_backward is only stable for ranks n - 1 <= r <= n with singular values sufficiently far away from each other.
    
21.py:
  Patch: https://github.com/pytorch/pytorch/commit/071971476d7431a24e527bdc181981678055a95d
  Problem: Binomial distribution class encounters overflow when logits are large. Note: the binomial distribution is parametrized by logits
  
25.py:
  Patch: https://github.com/pytorch/pytorch/commit/f8cab38578a99ad04d23256c2da877db4814f76f
  Problem: Matrix inverse triggers a cholesky error, because the matrix is not positive definite. Also, matrix inverse can cause numerical instability.
  
20.py: (INCOMPLETE)
  Patch: https://github.com/pytorch/pytorch/commit/470c496eb224bdd735eea1accf7269dfdd87d49f
  Problem: In multivariate normal distribution class, there is a function for computing the precision matrix that uses inverse, which is numerically unstable
  
23.py: (INCOMPLETE)
  Patch: https://github.com/pytorch/pytorch/commit/d16c8238e164c6499714de625eb73422382e5ec1
  Problem: Implementation of softmax  for certain cases (when the dim argument of softmax and axis do not equal to ndim - 1, where ndim - 1 = the last dimension) is numerically unstable. Large inputs into the exponential function will produce infinity and output of softmax becomes NaN.
