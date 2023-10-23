from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch


def pca_U(k, A, model, device):
    # A == U_prob
    # Computes the low-rank SVD for U matrix
    l = k + 2
    n = A.shape[0]
    # total number of elements in parameters
    # Returns the total number of elements in the input tensor
    m = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert k > 0
    assert k <= min(m, n)
    assert l >= k
    # (l, m)
    R = 2 * (torch.rand(l, m).to(device) - 0.5)
    Q = []
    # A.shape
    tem_2 = torch.rand_like(A, requires_grad=True).to(device)
    # grad = torch.autograd.grad(f(x), x)
    grads = torch.autograd.grad((A * tem_2).sum(0),
                                model.parameters(),
                                create_graph=True)
    # return a copy of grads with different shape
    right_grad_torch = torch.cat([grad.reshape(-1)
                                  for grad in grads]).unsqueeze(-1)
    for row_vectors in R:
        # grad(outputs, inputs) grad_outputs = Usually gradients w.r.t. each output.
        grads = torch.autograd.grad(right_grad_torch,
                                    tem_2,
                                    grad_outputs=row_vectors.unsqueeze(-1),
                                    create_graph=True)
        Q.append(torch.cat([grad.reshape(-1) for grad in grads]).unsqueeze(0))
    Q = torch.cat(Q).transpose(1, 0)
    # Q is the orthogonal matrix, and R is the upper triangular matrix to solve linear equations and eigenvalue problems.
    (Q, _) = torch.linalg.qr(Q, 'reduced')
    final_prod = []
    for col_vectors in Q.transpose(1, 0):
        grads = torch.autograd.grad(
            (A * col_vectors.detach()).sum(0),
            model.parameters(),
            retain_graph=True)
        final_prod.append(torch.cat([grad.reshape(-1) for grad in grads]).unsqueeze(0))
    final_prod = torch.cat(final_prod).transpose(1, 0)
    (U, s, Ra) = torch.svd(final_prod)
    Ra = Ra.transpose(1, 0)
    Va = torch.matmul(Ra, Q.transpose(1, 0))
    return U[:, :k].detach(), s[:k].detach(), Va[:k, :].detach()
