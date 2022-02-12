import torch
import torch.nn as nn

class Linear_ODE(nn.Module):
  def __init__(self, x_dim, a, param=None):
    super().__init__()
    self.a = nn.Parameter(a)
    self.x_dim = x_dim
    self.param=param
    if param is None:
      self.A = self.a[:]
    else:
      self.A = param(self.a, x_dim)

  def forward(self, t, u):
    # du/dt = f(u, t), input: N * x_dim, output: N * x_dim 
    if self.param is None:
      A = self.a[:]
    else:
      A = self.param(self.a, self.x_dim)
    out = u @ (A - torch.eye(self.x_dim)).t()
    return out

class Linear_ODE_single_var(nn.Module):
  def __init__(self, x_dim, a):
    super().__init__()
    self.a = nn.Parameter(a)
    self.x_dim = x_dim
    self.exp = self.construct_exp(x_dim)
    # self.A = torch.pow(self.a, self.exp)

  def construct_exp(self, x_dim):
    exp = torch.zeros(x_dim, x_dim)
    for i in range(x_dim):
      for j in range(x_dim):
        exp[i, j] = abs(i-j)+1
    return exp

  def A(self):
    return torch.pow(self.a, self.exp)

  def forward(self, t, u):
    # du/dt = f(u, t), input: N * x_dim, output: N * x_dim 
    # A = torch.pow(self.a, self.exp)
    out = u @ (self.A() - torch.eye(self.x_dim)).t()
    return out

class Linear_ODE_diag(nn.Module):
  def __init__(self, x_dim, a):
    super().__init__()
    self.a = nn.Parameter(a)
    self.x_dim = x_dim
    self.num_a = a.shape[0]

  def A(self):
    A = torch.zeros(self.x_dim, self.x_dim)
    for i in range(self.num_a):
      diagonal = (i+1)//2 * (-1)**i # 0, -1, 1, -2, 2,... 
      len_one = self.x_dim - (i+1)//2 # (d, d-1, d-1, d-2, d-2...)
      A = A + torch.diag(self.a[i] * torch.ones(len_one), diagonal=diagonal)
    return A

  def forward(self, t, u):
    # du/dt = f(u, t), input: N * x_dim, output: N * x_dim 
    # A = torch.pow(self.a, self.exp)
    out = u @ (self.A() - torch.eye(self.x_dim)).t()
    return out

class Lorenz63(nn.Module):
  def __init__(self, coeff, x_dim=3):
    super().__init__()
    self.coeff = nn.Parameter(coeff)
    self.x_dim = x_dim
  
  def forward(self, t, u):
    # (*bs * x_dim) -> (*bs * x_dim)
    sigma, beta, rho = self.coeff
    out = torch.stack((sigma * (u[...,1] - u[...,0]), rho * u[...,0] - u[...,1] - u[...,0] * u[...,2], u[...,0] * u[...,1] - beta * u[...,2]), dim=-1)
    return out

class Linear(nn.Module):
  def __init__(self, x_dim, y_dim, H):
    super().__init__()
    self.H = nn.Parameter(H)
    self.x_dim = x_dim
    self.y_dim = y_dim

  def forward(self, u):
    # du/dt = f(u, t), input: N * x_dim, output: N * x_dim 
    out = u @ self.H.t()
    return out

class ODE_Net(nn.Module):
  def __init__(self, x_dim, hidden_layer_widths, scaled_layer=False):
    # e.g. hidden_layer_widths = [40, 128, 64, 40]
    super().__init__()
    self.hidden_layer_widths = hidden_layer_widths
    self.num_hidden_layers = len(hidden_layer_widths) - 2
    self.layers = nn.ModuleList()
    for i in range(self.num_hidden_layers+1):
      layer = nn.Linear(hidden_layer_widths[i], hidden_layer_widths[i+1])
      if scaled_layer:
        layer.weight.data.mul_(1/math.sqrt(layer_dims[i]))
      self.layers.append(layer)

  def forward(self, t, u):
    for layer in self.layers[:-1]:
      u = torch.relu(layer(u))
    out = self.layers[-1](u)
    return out

class FC_Net(nn.Module):
  def __init__(self, x_dim, hidden_layer_widths, scaled_layer=False):
    # e.g. hidden_layer_widths = [40, 128, 64, 40]
    super().__init__()
    self.num_hidden_layers = len(hidden_layer_widths) - 2
    self.layers = nn.ModuleList()
    for i in range(self.num_hidden_layers+1):
      layer = nn.Linear(hidden_layer_widths[i], hidden_layer_widths[i+1])
      if scaled_layer:
        layer.weight.data.mul_(1/math.sqrt(layer_dims[i]))
      self.layers.append(layer)

  def forward(self, u):
    for layer in self.layers[:-1]:
      u = torch.relu(layer(u))
    out = self.layers[-1](u)
    return out

class ODE_Net_from_basenet(nn.Module):
  # e.g. base_net = [3,40], this_hidden_layer_widths=[40,40,3]
  def __init__(self, base, hidden_layer_widths):
    super().__init__()
    self.base = base
    self.num_hidden_layers = len(hidden_layer_widths) - 2
    self.layers = nn.ModuleList()
    for i in range(self.num_hidden_layers+1):
      layer = nn.Linear(hidden_layer_widths[i], hidden_layer_widths[i+1])
      self.layers.append(layer)

  def forward(self, t, u):
    u = torch.relu(self.base(u))
    for layer in self.layers[:-1]:
      u = torch.relu(layer(u))
    out = self.layers[-1](u)
    return out



class L96_ODE_Net(nn.Module):
  def __init__(self, x_dim):
    super().__init__()
    self.x_dim = x_dim
    self.layer1 = nn.Conv1d(1, 6, 5, padding=2, padding_mode='circular')
    self.layer2 = nn.Conv1d(12, 1, 1)

    # self.layer1 = nn,Conv1d(1,6,5)

  def forward(self, t, u):
    bs = u.shape[:-1]
    out = torch.relu(self.layer1(u.view(-1, self.x_dim).unsqueeze(-2)))
    out = torch.cat((out**2, out), dim=-2)
    out = self.layer2(out).squeeze(-2).view(*bs, self.x_dim)
    return out

class L96_ODE_Net_2(nn.Module):
  def __init__(self, x_dim):
    super().__init__()
    self.x_dim = x_dim
    self.layer1 = nn.Conv1d(1, 72, 5, padding=2, padding_mode='circular')
    # self.layer1b = nn.Conv1d(1, 24, 5, padding=2, padding_mode='circular')
    # self.layer1c = nn.Conv1d(1, 24, 5, padding=2, padding_mode='circular')
    self.layer2 = nn.Conv1d(48, 37, 5, padding=2, padding_mode='circular')
    self.layer3 = nn.Conv1d(37, 1, 1)

    # self.layer1 = nn,Conv1d(1,6,5)

  def forward(self, t, u):
    bs = u.shape[:-1] # (*bs, x_dim)
    out = torch.relu(self.layer1(u.view(-1, self.x_dim).unsqueeze(-2))) # (bs, 1, x_dim) -> (bs, 72, x_dim)
    out = torch.cat((out[...,:24,:], out[...,24:48,:] * out[...,48:,:]), dim=-2) # (bs, 72, x_dim) -> (bs, 48, x_dim)
    out = torch.relu(self.layer2(out)) # (bs, 48, x_dim) -> (bs, 37, x_dim)
    out = self.layer3(out).squeeze(-2).view(*bs, self.x_dim) # (bs, 37, x_dim) -> (bs, 1, x_dim) -> (*bs, x_dim)
    return out

class Lorenz96(nn.Module):
  def __init__(self, F, x_dim, device):
    super().__init__()
    self.F = nn.Parameter(torch.tensor(F))
    self.x_dim = x_dim
    self.indices_p1 = torch.tensor([(i+1)%self.x_dim for i in range(x_dim)], dtype=torch.long, device=device)
    self.indices_m2 = torch.tensor([(i-2)%self.x_dim for i in range(x_dim)], dtype=torch.long, device=device)
    self.indices_m1 = torch.tensor([(i-1)%self.x_dim for i in range(x_dim)], dtype=torch.long, device=device)

  def forward(self, t, u):
    # du/dt = f(u, t), input: N * x_dim, output: N * x_dim 
    out = (u.index_select(-1, self.indices_p1) - u.index_select(-1, self.indices_m2)) * u.index_select(-1, self.indices_m1) - u + self.F
    return out

class Lorenz96_correction(nn.Module):
  def __init__(self, coeff, x_dim=40):
    super().__init__()
    device = coeff.device
    self.coeff = coeff
    self.x_dim = x_dim
    self.indices_p1 = torch.tensor([(i+1)%self.x_dim for i in range(x_dim)], dtype=torch.long, device=device)
    self.indices_p2 = torch.tensor([(i+2)%self.x_dim for i in range(x_dim)], dtype=torch.long, device=device)
    self.indices_m2 = torch.tensor([(i-2)%self.x_dim for i in range(x_dim)], dtype=torch.long, device=device)
    self.indices_m1 = torch.tensor([(i-1)%self.x_dim for i in range(x_dim)], dtype=torch.long, device=device)

    self.layer1 = nn.Conv1d(1, 72, 5, padding=2, padding_mode='circular')
    # self.layer1b = nn.Conv1d(1, 24, 5, padding=2, padding_mode='circular')
    # self.layer1c = nn.Conv1d(1, 24, 5, padding=2, padding_mode='circular')
    self.layer2 = nn.Conv1d(48, 37, 5, padding=2, padding_mode='circular')
    self.layer3 = nn.Conv1d(37, 1, 1)

  def forward(self, t, u):
    # (*bs, x_dim) -> (*bs, x_dim)
    u_m2 = u.index_select(-1, self.indices_m2)
    u_m1 = u.index_select(-1, self.indices_m1)
    u_p1 = u.index_select(-1, self.indices_p1)
    u_p2 = u.index_select(-1, self.indices_p2)
    to_cat = []
    to_cat.append(torch.ones_like(u))
    to_cat.extend([u_m2, u_m1, u, u_p1, u_p2])
    # to_cat.append(u)
    to_cat.extend([u_m2**2, u_m1**2, u**2, u_p1**2, u_p2**2])
    to_cat.extend([u_m2*u_m1, u_m1*u, u*u_p1, u_p1*u_p2])
    # to_cat.append(u_m2*u_m1)
    to_cat.extend([u_m2*u, u_m1*u_p1, u*u_p2])
    # to_cat.append(u_m1*u_p1)
    out1 = torch.stack(to_cat, dim=-1) @ self.coeff  # (*bs, x_dim, N_a) @ (N_a)  -> (*bs, x_dim)
    
    bs = u.shape[:-1] # (*bs, x_dim)
    out2 = torch.relu(self.layer1(u.view(-1, self.x_dim).unsqueeze(-2))) # (bs, 1, x_dim) -> (bs, 72, x_dim)
    out2 = torch.cat((out2[...,:24,:], out2[...,24:48,:] * out2[...,48:,:]), dim=-2) # (bs, 72, x_dim) -> (bs, 48, x_dim)
    out2 = torch.relu(self.layer2(out2)) # (bs, 48, x_dim) -> (bs, 37, x_dim)
    out2 = self.layer3(out2).squeeze(-2).view(*bs, self.x_dim) # (bs, 37, x_dim) -> (bs, 1, x_dim) -> (*bs, x_dim)
    return out1+out2

class Lorenz96_dict_param(nn.Module):
  def __init__(self, coeff, device, x_dim=40):
    super().__init__()
    self.coeff = nn.Parameter(coeff)
    self.x_dim = x_dim
    self.indices_p1 = torch.tensor([(i+1)%self.x_dim for i in range(x_dim)], dtype=torch.long, device=device)
    self.indices_p2 = torch.tensor([(i+2)%self.x_dim for i in range(x_dim)], dtype=torch.long, device=device)
    self.indices_m2 = torch.tensor([(i-2)%self.x_dim for i in range(x_dim)], dtype=torch.long, device=device)
    self.indices_m1 = torch.tensor([(i-1)%self.x_dim for i in range(x_dim)], dtype=torch.long, device=device)

  def forward(self, t, u):
    # (*bs, x_dim) -> (*bs, x_dim)
    u_m2 = u.index_select(-1, self.indices_m2)
    u_m1 = u.index_select(-1, self.indices_m1)
    u_p1 = u.index_select(-1, self.indices_p1)
    u_p2 = u.index_select(-1, self.indices_p2)
    to_cat = []
    to_cat.append(torch.ones_like(u))
    to_cat.extend([u_m2, u_m1, u, u_p1, u_p2])
    # to_cat.append(u)
    to_cat.extend([u_m2**2, u_m1**2, u**2, u_p1**2, u_p2**2])
    to_cat.extend([u_m2*u_m1, u_m1*u, u*u_p1, u_p1*u_p2])
    # to_cat.append(u_m2*u_m1)
    to_cat.extend([u_m2*u, u_m1*u_p1, u*u_p2])
    # to_cat.append(u_m1*u_p1)
    out = torch.stack(to_cat, dim=-1) @ self.coeff  # (*bs, x_dim, N_a) @ (N_a)  -> (*bs, x_dim)
    return out


class Lorenz96_FS(nn.Module):
  def __init__(self, param, device, xx_dim=36, xy_dim=10):
    super().__init__()
    self.param = nn.Parameter(param)

    self.xx_dim = xx_dim
    self.xy_dim = xy_dim
    self.x_dim = xx_dim * (xy_dim + 1)

    self.indices_x = torch.tensor([i for i in range(xx_dim)], dtype=torch.long)
    self.indices_x_p1 = torch.tensor([(i + 1) % self.xx_dim for i in range(xx_dim)], dtype=torch.long, device=device)
    self.indices_x_m2 = torch.tensor([(i - 2) % self.xx_dim for i in range(xx_dim)], dtype=torch.long, device=device)
    self.indices_x_m1 = torch.tensor([(i - 1) % self.xx_dim for i in range(xx_dim)], dtype=torch.long, device=device)
    self.indices_y_p1 = torch.tensor([(i + 1) % self.xy_dim for i in range(xy_dim)], dtype=torch.long, device=device)
    self.indices_y_p2 = torch.tensor([(i + 2) % self.xy_dim for i in range(xy_dim)], dtype=torch.long, device=device)
    self.indices_y_m1 = torch.tensor([(i - 1) % self.xy_dim for i in range(xy_dim)], dtype=torch.long, device=device)

  def forward(self, t, u):
    # (*bs * x_dim) -> (*bs * x_dim)
    bs = u.shape[:-1]
    F, h, c, b = self.param
    to_cat = []
    u_y = u[..., self.xx_dim:].reshape(*bs, self.xx_dim, self.xy_dim)  # (*bs, xx_dim, xy_dim)
    # print(u.index_select(-1, self.indices_x_p1).shape, u_y.mean(dim=-1).shape)
    to_cat.append((u.index_select(-1, self.indices_x_p1) - u.index_select(-1, self.indices_x_m2)) * u.index_select(-1, self.indices_x_m1) - u[...,:self.xx_dim] + F - h * c * u_y.mean(dim=-1))
    to_cat.append(c * (-b * u_y.index_select(-1, self.indices_y_p1) * (u_y.index_select(-1, self.indices_y_p2) - u_y.index_select(-1,self.indices_y_m1)) - u_y + h / self.xy_dim * u[...,:self.xx_dim].unsqueeze(-1)).view(*bs, self.xx_dim * self.xy_dim))
    out = torch.cat(to_cat, dim=-1)
    return out



class One_Layer_NN(nn.Module):
  def __init__(self, input_dim, output_dim, H=None, residual=False, bias=False):
    super().__init__()
    self.residual = residual
    self.layer1 = nn.Linear(input_dim, output_dim, bias=bias)
    if H is not None:
      self.layer1.weight.data = H
      self.H = H

  def forward(self, x):
    res = self.layer1(x)
    if self.residual:
      out = res + x
    else:
      out = res
    return out

class Two_Layer_NN(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim, residual=True, activation="relu", batchnorm=False):
    super().__init__()
    if activation == "relu":
      self.activation = torch.relu
    elif activation == "tanh":
      self.activation = torch.tanh
    self.residual = residual
    self.batchnorm = batchnorm
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.layer1 = nn.Linear(input_dim, hidden_dim)
    self.layer2 = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    # x = x[:,1].unsqueeze(1)
    res = x
    if self.batchnorm:
      res = nn.BatchNorm1d(self.input_dim)(res)
    res = self.activation(self.layer1(res))
    res = self.layer2(res)
    if self.residual:
      out = res + x
    else:
      out = res
    # out = res + 0.5*x[:,1].unsqueeze(1)
    return out

class Three_Layer_NN(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim, residual=True, activation="relu"):
    super().__init__()
    if activation == "relu":
      self.activation = torch.relu
    elif activation == "tanh":
      self.activation = torch.tanh
    self.residual = residual
    self.output_dim = output_dim
    self.layer1 = nn.Linear(input_dim, hidden_dim[0])
    self.layer2 = nn.Linear(hidden_dim[0], hidden_dim[1])
    self.layer3 = nn.Linear(hidden_dim[1], output_dim)

  def forward(self, x):
    res = self.activation(self.layer1(x))
    res = self.activation(self.layer2(res))
    res = self.layer3(res)
    if self.residual:
      out = res + x
    else:
      out = res
    return out

class Four_Layer_NN(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim, residual=True, activation="relu"):
    super().__init__()
    if activation == "relu":
      self.activation = torch.relu
    elif activation == "tanh":
      self.activation = torch.tanh
    self.residual = residual
    self.output_dim = output_dim
    self.layer1 = nn.Linear(input_dim, hidden_dim[0])
    self.layer2 = nn.Linear(hidden_dim[0], hidden_dim[1])
    self.layer3 = nn.Linear(hidden_dim[1], hidden_dim[2])
    self.layer4 = nn.Linear(hidden_dim[2], output_dim)

  def forward(self, x):
    res = self.activation(self.layer1(x))
    res = self.activation(self.layer2(res))
    res = self.activation(self.layer3(res))
    res = self.layer4(res)
    if self.residual:
      out = res + x
    else:
      out = res
    return out


