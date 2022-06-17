import torch
@staticmethod
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