import torch
import torch.nn as nn
import utils

class AddGaussian(nn.Module):
  """

  """
  def __init__(self, x_dim, q_true, param_type, tril_radius=None, q_shape=None):
    super().__init__()
    self.x_dim = x_dim
    self.tril_radius = tril_radius
    self.q = nn.Parameter(self.pre_process(q_true, param_type))
    self.param_type = param_type
    self.q_shape = q_shape
    if q_shape is not None:
      self.base = utils.construct_exp(x_dim)
      self.q_shape = nn.Parameter(q_shape)
    
      

  def pre_process(self, q_true, param_type):
    # We want to pass something to nn.Parameter that can take *every* real value, not just positives.
    if param_type == "scalar":
      return utils.softplus_inv(q_true)
    elif param_type == "diag":
      return utils.softplus_inv(q_true)
    elif param_type == "tril":
      if self.tril_radius is None:
        return torch.tril(q_true, diagonal=-1) + torch.diag(utils.softplus_inv(q_true.diag()))
      else:
        return torch.tril(q_true, diagonal=-1) - torch.tril(q_true, diagonal=-1-self.tril_radius) + torch.tril(q_true, diagonal=self.tril_radius-self.x_dim) + torch.diag(utils.softplus_inv(q_true.diag()))
    elif param_type == "full":
      return q_true

  def post_process(self, q, param_type):
    if param_type == "scalar":
      return utils.softplus(q)
    elif param_type == "diag":
      return utils.softplus(q)
    elif param_type == "tril":
      return torch.tril(q, diagonal=-1) + torch.diag(utils.softplus(q.diag()))
    elif param_type == "full":
      return q

  def forward(self, X, X_prev=None):
    if self.param_type == "scalar":
      if self.q_shape is None:
        X = X + self.post_process(self.q, self.param_type) * torch.randn_like(X)
      else:
        chol = self.post_process(self.q, self.param_type) * torch.linalg.cholesky(torch.exp(self.q_shape * self.base))
        X = X + torch.randn_like(X) @ chol.t()
    elif self.param_type == "diag":
      X = X + self.post_process(self.q, self.param_type) * torch.randn_like(X) # (x_dim) * (*bs, N_ensem, x_dim)
    elif self.param_type == "tril":
      # batch_shape = X.shape[:-1]
      chol = self.post_process(self.q, self.param_type)
      X = X + torch.randn_like(X) @ chol.t() # (*bs, N_ensem, x_dim) @ (x_dim, x_dim)
      # X = X + torch.distributions.MultivariateNormal(torch.zeros(self.x_dim, device=self.q.device), scale_tril=chol).sample(batch_shape)
    elif self.param_type == "full":
      batch_shape = X.shape[:-1]
      chol = torch.linalg.cholesky(self.q)
      X = X + torch.distributions.MultivariateNormal(torch.zeros(self.x_dim, device=self.q.device), scale_tril=chol).sample(batch_shape) # (*bs, N_ensem, x_dim)
    return X

  def chol(self, X_prev=None):
    if self.param_type == "scalar":
      if self.q_shape is None:
        return self.post_process(self.q, self.param_type) * torch.eye(self.x_dim, device=self.q.device)
      else:
        return self.post_process(self.q, self.param_type) * torch.linalg.cholesky(torch.exp(self.q_shape * self.base))
    elif self.param_type == "diag":
      return self.post_process(self.q, self.param_type) * torch.eye(self.x_dim, device=self.q.device)
    elif self.param_type == "tril":
      return self.post_process(self.q, self.param_type)
    elif self.param_type == "full":
      return torch.linalg.cholesky(self.q)

  def full(self, X_prev=None):
    chol = self.chol()
    return chol @ chol.t()

  def q_true(self, X_prev=None):
    return self.post_process(self.q, self.param_type)

  def post_grad(self):
    # Some pytorch trick to compute d(loss)/d(q_true) where q_true = post_process(self.q)
    leaf = self.post_process(self.q, self.param_type).detach().requires_grad_()
    q_sub = self.pre_process(leaf, self.param_type) # ideally should recover self.q, but we can compute d(q_sub)/d(leaf)
    q_sub.backward(gradient=self.q.grad)
    return leaf.grad

class AddGaussianNet(nn.Module):
  """

  """
  def __init__(self, x_dim, hidden_layer_widths, param_type):
    super().__init__()
    self.x_dim = x_dim
    self.num_hidden_layers = len(hidden_layer_widths) - 2
    self.layers = nn.ModuleList()
    for i in range(self.num_hidden_layers+1):
      layer = nn.Linear(hidden_layer_widths[i], hidden_layer_widths[i+1])
      # if i == self.num_hidden_layers:
      #   layer.bias.data.fill_(2.0)
      self.layers.append(layer)
    self.param_type = param_type

  def pre_process(self, X_prev):
    # We want to pass something to nn.Parameter that can take *every* real value, not just positives.
    # X_prev: (*bs, N_ensem, x_dim)
    for layer in self.layers[:-1]:
      X_prev = torch.sigmoid(layer(X_prev))
    q = self.layers[-1](X_prev) # (*bs, N_ensem, q_dim)
    return q

  def post_process(self, q, param_type):
    if param_type == "scalar":
      return utils.softplus(q) # (*bs, N_ensem, 1)
    elif param_type == "diag":
      return utils.softplus(q) # (*bs, N_ensem, x_dim)
    elif param_type == "tril":
      batch_shape = q.shape[:-1]
      q = q.view(*batch_shape, self.x_dim, self.x_dim) # (*bs, N_ensem, x_dim, x_dim)
      return torch.tril(q, diagonal=-1) + torch.diag_embed(utils.softplus( torch.diagonal(q, dim1=-2, dim2=-1))) # (*bs, N_ensem, x_dim, x_dim)
    elif param_type == "full":
      batch_shape = q.shape[:-1]
      q = q.view(*batch_shape, self.x_dim, self.x_dim) # (*bs, N_ensem, x_dim, x_dim)
      return q

  def forward(self, X, X_prev):
    # X: (*bs, N_ensem, x_dim)
    q = self.pre_process(X_prev)
    if self.param_type == "scalar":
      X = X + self.post_process(q, self.param_type) * torch.randn_like(X) # (*bs, N_ensem, x_dim)
    elif self.param_type == "diag":
      X = X + self.post_process(q, self.param_type) * torch.randn_like(X) # (*bs, N_ensem, x_dim)
    elif self.param_type == "tril":
      # batch_shape = X.shape[:-1]
      chol = self.post_process(q, self.param_type)
      X = X + (torch.randn_like(X).unsqueeze(-2) @ chol.transpose(-2,-1)).squeeze(-2) # (*bs, N_ensem, x_dim) @ (*bs, N_ensem, x_dim, x_dim)
      # X = X + torch.distributions.MultivariateNormal(torch.zeros(self.x_dim, device=self.q.device), scale_tril=chol).sample(batch_shape)
    elif self.param_type == "full":
      batch_shape = X.shape[:-1]
      chol = torch.linalg.cholesky(q)
      X = X + torch.distributions.MultivariateNormal(torch.zeros(self.x_dim, device=q.device), scale_tril=chol).sample(batch_shape) # (*bs, N_ensem, x_dim)
    return X

  # TODO
  def chol(self, X_prev):
    q = self.pre_process(X_prev)
    if self.param_type == "scalar":
      return self.post_process(q, self.param_type).unsqueeze(-1) * torch.eye(self.x_dim, device=q.device) # (*bs, N_ensem, 1) * (x_dim, x_dim) -> (*bs, N_ensem, x_dim, x_dim)
    elif self.param_type == "diag":
      return self.post_process(q, self.param_type).unsqueeze(-1) * torch.eye(self.x_dim, device=q.device) # (*bs, N_ensem, x_dim) * (x_dim, x_dim) -> (*bs, N_ensem, x_dim, x_dim)
    elif self.param_type == "tril":
      return self.post_process(q, self.param_type) # (*bs, N_ensem, x_dim, x_dim)
    elif self.param_type == "full":
      return torch.linalg.cholesky(q) # (*bs, N_ensem, x_dim, x_dim)

  def full(self, X_prev):
    chol = self.chol(X_prev)
    return chol @ chol.transpose(-2,-1) # (*bs, N_ensem, x_dim, x_dim)

  def q_true(self, X_prev):
    q = self.pre_process(X_prev)
    return self.post_process(q, self.param_type)

  # def post_grad(self):
  #   # Some pytorch trick to compute d(loss)/d(q_true) where q_true = post_process(self.q)
  #   leaf = self.post_process(self.q, self.param_type).detach().requires_grad_()
  #   q_sub = self.pre_process(leaf, self.param_type) # ideally should recover self.q, but we can compute d(q_sub)/d(leaf)
  #   q_sub.backward(gradient=self.q.grad)
  #   return leaf.grad

class AddGaussianNet_from_basenet(nn.Module):
  """

  """
  def __init__(self, x_dim, base, hidden_layer_widths, param_type):
    super().__init__()
    self.x_dim = x_dim
    self.base = base
    self.num_hidden_layers = len(hidden_layer_widths) - 2
    self.layers = nn.ModuleList()
    for i in range(self.num_hidden_layers+1):
      layer = nn.Linear(hidden_layer_widths[i], hidden_layer_widths[i+1])
      self.layers.append(layer)
    self.param_type = param_type

  def pre_process(self, X_prev):
    # We want to pass something to nn.Parameter that can take *every* real value, not just positives.
    q = torch.sigmoid(self.base(X_prev))
    for layer in self.layers[:-1]:
      q = torch.sigmoid(layer(q))
    q = self.layers[-1](q)
    return q
  
  def post_process(self, q, param_type):
    if param_type == "scalar":
      return utils.softplus(q)
    elif param_type == "diag":
      return utils.softplus(q)
    elif param_type == "tril":
      batch_shape = q.shape[:-1]
      q = q.view(*batch_shape, self.x_dim, self.x_dim)
      return torch.tril(q, diagonal=-1) + torch.diag_embed(utils.softplus( torch.diagonal(q, dim1=-2, dim2=-1)))
    elif param_type == "full":
      batch_shape = q.shape[:-1]
      q = q.view(*batch_shape, self.x_dim, self.x_dim)
      return q

  def forward(self, X, X_prev):
    q = self.pre_process(X_prev)
    if self.param_type == "scalar":
      X = X + self.post_process(q, self.param_type) * torch.randn_like(X)
    elif self.param_type == "diag":
      X = X + self.post_process(q, self.param_type) * torch.randn_like(X) # (x_dim) * (*bs, N_ensem, x_dim)
    elif self.param_type == "tril":
      # batch_shape = X.shape[:-1]
      chol = self.post_process(q, self.param_type)
      X = X + (torch.randn_like(X).unsqueeze(-2) @ chol.transpose(-2,-1)).squeeze(-2) # (*bs, N_ensem, x_dim) @ (*bs, N_ensem, x_dim, x_dim)
      # X = X + torch.distributions.MultivariateNormal(torch.zeros(self.x_dim, device=self.q.device), scale_tril=chol).sample(batch_shape)
    elif self.param_type == "full":
      batch_shape = X.shape[:-1]
      chol = torch.linalg.cholesky(q)
      X = X + torch.distributions.MultivariateNormal(torch.zeros(self.x_dim, device=q.device), scale_tril=chol).sample(batch_shape) # (*bs, N_ensem, x_dim)
    return X

  # TODO
  def chol(self, X_prev):
    q = self.pre_process(X_prev)
    if self.param_type == "scalar":
      return self.post_process(q, self.param_type).unsqueeze(-1) * torch.eye(self.x_dim, device=q.device)
    elif self.param_type == "diag":
      return self.post_process(q, self.param_type).unsqueeze(-1) * torch.eye(self.x_dim, device=q.device)
    elif self.param_type == "tril":
      return self.post_process(q, self.param_type)
    elif self.param_type == "full":
      return torch.linalg.cholesky(q)

  def full(self, X_prev):
    chol = self.chol(X_prev)
    return chol @ chol.transpose(-2,-1)

  def q_true(self, X_prev):
    q = self.pre_process(X_prev)
    return self.post_process(q, self.param_type)

  # def post_grad(self):
  #   # Some pytorch trick to compute d(loss)/d(q_true) where q_true = post_process(self.q)
  #   leaf = self.post_process(self.q, self.param_type).detach().requires_grad_()
  #   q_sub = self.pre_process(leaf, self.param_type) # ideally should recover self.q, but we can compute d(q_sub)/d(leaf)
  #   q_sub.backward(gradient=self.q.grad)
  #   return leaf.grad

# class Gaussian():
#   def __init__(self, x_dim, q_true, param_type):
#     self.x_dim = x_dim
#     self.q = self.pre_process(q_true, param_type)
#     self.param_type = param_type

#   def pre_process(self, q_true, param_type):
#     if param_type == "scalar":
#       return utils.softplus_inv(q_true)
#     elif param_type == "diag":
#       return utils.softplus_inv(q_true)
#     elif param_type == "tril":
#       return torch.tril(q_true, diagonal=-1) + torch.diag(utils.softplus_inv(q_true.diag()))
#     elif param_type == "full":
#       return q_true

#   def post_process(self, q, param_type):
#     if param_type == "scalar":
#       return utils.softplus(q)
#     elif param_type == "diag":
#       return utils.softplus(q)
#     elif param_type == "tril":
#       return torch.tril(q, diagonal=-1) + torch.diag(utils.softplus(q.diag()))
#     elif param_type == "full":
#       return q
  
#   def sample(self, shape):
#     if self.param_type == "scalar":
#       return self.post_process(self.q, self.param_type) * torch.randn(shape, device=self.q.device)
#     elif self.param_type == "diag":
#       return self.post_process(self.q, self.param_type) * torch.randn(shape, device=self.q.device) # (x_dim) * (*bs, N_ensem, x_dim)
#     elif self.param_type == "tril":
#       chol = self.post_process(self.q, self.param_type)
#       return torch.randn(shape, device=self.q.device) @ chol.t() # (*bs, N_ensem, x_dim) @ (x_dim, x_dim)
#     elif self.param_type == "full":
#       batch_shape = shape[:-1]
#       chol = torch.cholesky(self.q)
#       return torch.distributions.MultivariateNormal(torch.zeros(self.x_dim, device=self.q.device), scale_tril=chol).sample(batch_shape) # (*bs, N_ensem, x_dim)
#     return X
  
#   def chol(self):
#     if self.param_type == "scalar":
#       return self.post_process(self.q, self.param_type) * torch.eye(self.x_dim, device=self.q.device)
#     elif self.param_type == "diag":
#       return self.post_process(self.q, self.param_type) * torch.eye(self.x_dim, device=self.q.device)
#     elif self.param_type == "tril":
#       return self.post_process(self.q, self.param_type)
#     elif self.param_type == "full":
#       return torch.cholesky(self.q)

#   def full(self):
#     chol = self.chol()
#     return chol @ chol.t()

#   def q_true(self):
#     return self.post_process(self.q, self.param_type)

#   def post_grad(self):
#     # Some pytorch trick to compute d(loss)/d(q_true) where q_true = post_process(self.q)
#     leaf = self.post_process(self.q, self.param_type).detach().requires_grad_()
#     q_sub = self.pre_process(leaf, self.param_type) # ideally should recover self.q, but we can compute d(q_sub)/d(leaf)
#     q_sub.backward(gradient=self.q.grad)
#     return leaf.grad
  
