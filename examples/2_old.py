import torch    
@staticmethod
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