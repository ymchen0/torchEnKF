from experiments import DEModel
from torchdiffeq import odeint

from tqdm.auto import tqdm

from scipy import signal
import random
import torch
import torch.nn as nn
import numpy as np

def forecast_skill(true_ode_func, test_ode_func, t_obs, u0, device, ode_method, return_seq):
  '''
  Args:
    u0: group of initial conditions # (*n_test, x_dim)
    t_obs: equally spaced points, over which ode is integrated. Step_size is t_obs[0]
  Returns:
  '''
  step_size = t_obs[0]
  if len(u0.shape) == 1:
    u0 = u0.unsqueeze(0)
  out_true, _, _, _ = DEModel.generate_data(true_ode_func, None, t_obs, u0, None, None, device, ode_method, ode_options=dict(step_size=step_size)) # (n_obs, *n_test, x_dim)
  out_test, _, _, _ = DEModel.generate_data(test_ode_func, None, t_obs, u0, None, None, device, ode_method, ode_options=dict(step_size=step_size)) # (n_obs, *n_test, x_dim)
  rmse = torch.sqrt(((out_true - out_test)**2).mean(dim=-1).mean(dim=-1))
  if return_seq:
    rdn = random.randrange(u0.shape[0])
    return rmse, (out_true[:,rdn,:], out_test[:,rdn,:])
  else:
    return rmse # (n_obs)

def power_spectrum_density(ode_func, t_obs, u0, device, ode_method, nperseg, model_Q=None):
  if len(u0.shape) == 2:
    u0 = u0[0]
  x_dim = u0.shape[0]
  step_size = t_obs[0]
  fs = 1 / step_size.item()
  out, _, _, _ = DEModel.generate_data(ode_func, None, t_obs, u0, model_Q, None, device, ode_method, ode_options=dict(step_size=step_size)) # (n_obs, x_dim)
  out = out.detach().cpu().numpy()
  den_all = []
  for i in range(x_dim):
    freqs, den = signal.welch(out[:,i], fs, nperseg=nperseg)
    den_all.append(den)
  den_all = np.asarray(den_all)
  return freqs, den_all.mean(axis=0)

def lyapunov_exponent(ode_func, dt_obs, dt_int, n_obs, u0, device, ode_method):
  if len(u0.shape) == 2:
    u0 = u0[0]

  ode_func_single_arg = lambda u: ode_func(0,u)
  x_dim = u0.shape[0]
  U = torch.eye(x_dim, device=device)
  u_aug = torch.cat((u0, U.view(-1))) # (x_dim + x_dim**2)
  class Aug_ode(nn.Module):
    def __init__(self):
      super().__init__()
    
    def forward(self, t, u):
      A = torch.autograd.functional.jacobian(ode_func_single_arg, u[:x_dim], create_graph=False, strict=False, vectorize=True) # (x_dim, x_dim)
      A = A.detach()
      f1 = ode_func_single_arg(u[:x_dim])
      f2 = u[x_dim:].view(x_dim,x_dim) @ A.transpose(0,1)
      f2 = f2.view(-1)
      return torch.cat((f1,f2))

  aug_ode = Aug_ode()
  spec = torch.zeros(x_dim, device=device)
  for j in tqdm(range(n_obs), leave=False):
    _, u_aug = odeint(aug_ode, u_aug, torch.tensor([0., dt_obs], device=device), method=ode_method, options=dict(step_size=dt_int))
    Q, R = torch.linalg.qr(u_aug[x_dim:].view(x_dim,x_dim).transpose(0,1))
    Q = Q.detach()
    R = R.detach()
    spec += torch.log(torch.abs(torch.diag(R)))
    u_aug = torch.cat((u_aug[:x_dim], Q.transpose(0,1).view(-1)))
  spec = np.sort(spec.detach().cpu().numpy())[::-1]
  return spec / (dt_obs * n_obs)











