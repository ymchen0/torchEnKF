import torch
import torch.nn as nn
from torchEnKF import misc
import math

class AddGaussian(nn.Module):
    """
        torch.nn.Module that adds a Gaussian perturbation to a given input.
        (softplus function is used to ensure positiveness.)

        The Gaussian perturbation is parameterized by q, which may take the following forms, depending on "param_type":
        1. "scalar": q has shape (1,). The perturbation is N(0, softplus(q)**2 * Id)
        2. "diag": q has shape (x_dim,). The perturbation is N(0, diag(softplus(q)**2))
        3. "tril": q has shape (x_dim, x_dim). The perturbation is N(0, LL^T), where l is lower-triangular.
            Diagonal entries of L are the diagonal entries of q transformed by a softplus.
            Lower triangular part of L is the same as that of q.
        4. "full": q has shape (x_dim, x_dim) and is positive definite. The perturbation is N(0, q).
            Use this if q does not need to be learned.
    """

    def __init__(self, x_dim, q_true, param_type, q_shape=None):
        # q_shape: Additional parameter for the linear Gaussian experiment appeared in paper
        super().__init__()
        self.x_dim = x_dim
        self.q = nn.Parameter(self.pre_process(q_true, param_type))
        self.param_type = param_type
        self.q_shape = q_shape
        if q_shape is not None:
            self.base = misc.construct_exp(x_dim)
            self.q_shape = nn.Parameter(q_shape)

    def pre_process(self, q_true, param_type):
        # We want to pass something to nn.Parameter that can take *every* real value, not just positives.
        if param_type == "scalar":
            return misc.softplus_inv(q_true)
        elif param_type == "diag":
            return misc.softplus_inv(q_true)
        elif param_type == "tril":
            return torch.tril(q_true, diagonal=-1) + torch.diag(misc.softplus_inv(q_true.diag()))
        elif param_type == "full":
            return q_true

    def post_process(self, q, param_type):
        if param_type == "scalar":
            return misc.softplus(q)
        elif param_type == "diag":
            return misc.softplus(q)
        elif param_type == "tril":
            return torch.tril(q, diagonal=-1) + torch.diag(misc.softplus(q.diag()))
        elif param_type == "full":
            return q

    def forward(self, X):
        if self.param_type == "scalar":
            if self.q_shape is None:
                X = X + self.post_process(self.q, self.param_type) * torch.randn_like(X)
            else:
                chol = self.post_process(self.q, self.param_type) * torch.linalg.cholesky(
                    torch.exp(self.q_shape * self.base))
                X = X + torch.randn_like(X) @ chol.t()
        elif self.param_type == "diag":
            X = X + self.post_process(self.q, self.param_type) * torch.randn_like(X)  # (x_dim) * (*bs, N_ensem, x_dim)
        elif self.param_type == "tril":
            # batch_shape = X.shape[:-1]
            chol = self.post_process(self.q, self.param_type)
            X = X + torch.randn_like(X) @ chol.t()  # (*bs, N_ensem, x_dim) @ (x_dim, x_dim)
            # X = X + torch.distributions.MultivariateNormal(torch.zeros(self.x_dim, device=self.q.device), scale_tril=chol).sample(batch_shape)
        elif self.param_type == "full":
            batch_shape = X.shape[:-1]
            chol = torch.linalg.cholesky(self.q)
            X = X + torch.distributions.MultivariateNormal(torch.zeros(self.x_dim, device=self.q.device),
                                                           scale_tril=chol).sample(batch_shape)  # (*bs, N_ensem, x_dim)
        return X

    def chol(self):
        if self.param_type == "scalar":
            if self.q_shape is None:
                return self.post_process(self.q, self.param_type) * torch.eye(self.x_dim, device=self.q.device)
            else:
                return self.post_process(self.q, self.param_type) * torch.linalg.cholesky(
                    torch.exp(self.q_shape * self.base))
        elif self.param_type == "diag":
            return self.post_process(self.q, self.param_type) * torch.eye(self.x_dim, device=self.q.device)
        elif self.param_type == "tril":
            return self.post_process(self.q, self.param_type)
        elif self.param_type == "full":
            return torch.linalg.cholesky(self.q)

    def inv(self):
        if self.param_type == "scalar":
            return 1 / (self.post_process(self.q, self.param_type) ** 2) * torch.eye(self.x_dim,device=self.q.device)
        elif self.param_type == "diag":
            return 1 / (self.post_process(self.q, self.param_type) ** 2) * torch.eye(self.x_dim,device=self.q.device)
        elif self.param_type == "tril":
            return torch.cholesky_inverse(self.post_process(self.q, self.param_type))
        elif self.param_type == "full":
            return torch.cholesky_inverse(torch.linalg.cholesky(self.q))

    def logdet(self):
        if self.param_type == "scalar":
            return 2 * self.x_dim * torch.log(self.post_process(self.q, self.param_type))
        elif self.param_type == "diag":
            return 2 * self.post_process(self.q, self.param_type).log().sum()
        elif self.param_type == "tril":
            return 2 * self.post_process(self.q, self.param_type).diagonal(dim1=-2,dim2=-1).log().sum(-1)
        elif self.param_type == "full":
            return 2 * torch.linalg.cholesky(self.q).diagonal(dim1=-2,dim2=-1).log().sum(-1)

    def full(self):
        chol = self.chol()
        return chol @ chol.t()

    def q_true(self):
        return self.post_process(self.q, self.param_type)

    def post_grad(self):
        # Some pytorch tricks to compute d(loss)/d(q_true) where q_true = post_process(self.q)
        leaf = self.post_process(self.q, self.param_type).detach().requires_grad_()
        q_sub = self.pre_process(leaf,self.param_type)  # ideally should recover self.q, but we can compute d(q_sub)/d(leaf)
        q_sub.backward(gradient=self.q.grad)
        return leaf.grad
