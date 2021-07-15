import math
import torch
from models import NNModel
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
from tqdm.auto import tqdm



def generate_data(ode_func, obs_func, t_obs, u0, model_Q_param, noise_R_param, device, ode_method='rk4', ode_options=None, adjoint=False, save_intermediate_step=False, t0=0., time_varying_obs=False, tqdm=False):
  """
  t_obs: 1D tensor. Time points where observations are available. # (n_obs)
  u0: Starting point of u at time t0. # (*bs, x_dim)
  obs_func: observation func. Make sure it's nn.Module if time_varying_obs=False
                    and list(nn.Module) if time_varying_obs=True. # (n_obs, )
  model_Q: (Additive) model error covariance. Can be regarded as additive covariance inflation # (x_dim, x_dim)
  """

  if save_intermediate_step and 'step_size' not in ode_options:
    raise ValueError('Specify step size first to save intermediate steps.')

  ode_integrator = odeint_adjoint if adjoint else odeint

  x_dim = u0.shape[-1]
  n_obs = t_obs.shape[0]
  bs = u0.shape[:-1]

  out = torch.empty(n_obs, *bs, x_dim, dtype=u0.dtype, device=device) # (n_obs, *bs, x_dim)
  out_intermediate = None
  t_intermediate = None
  X = u0

  y_obs = None
  if obs_func is not None:
    obs_func_0 = obs_func[0] if time_varying_obs else obs_func
    y_dim = obs_func_0(u0).shape[-1]
    y_obs = torch.empty(n_obs, *bs, y_dim, dtype=u0.dtype, device=device)

  if save_intermediate_step:
    step_size = ode_options['step_size']
    n_intermediate = round(((t_obs[-1] - t0) / step_size).item())
    t_intermediate = torch.linspace(t0, t_obs[-1], n_intermediate+1, device=device)
    out_intermediate = torch.empty(n_intermediate+1, *bs, x_dim, dtype=u0.dtype, device=device)
    out_intermediate[0] = u0.detach().clone()
    n_cur = 0

  t_cur = t0

  pbar = tqdm(range(n_obs), leave=False) if tqdm else range(n_obs)
  for j in pbar:
    X_prev = X
    if not save_intermediate_step:
      _, X = ode_integrator(ode_func, X, torch.tensor([t_cur, t_obs[j]], device=device), method=ode_method, options=ode_options)
    else:
      n_intermediate_j = round(((t_obs[j] - t_cur) / step_size).item())
      out_intermediate_j = ode_integrator(ode_func, X, torch.linspace(t_cur, t_obs[j], n_intermediate_j+1, device=device), method=ode_method, options=ode_options)
      out_intermediate[n_cur+1:n_cur+n_intermediate_j] = out_intermediate_j[1:-1].detach().clone()
      X = out_intermediate_j[-1]
      n_cur += n_intermediate_j
    t_cur = t_obs[j]

    if model_Q_param is not None:
      X = model_Q_param(X, X_prev)
    
    out[j] = X
    if save_intermediate_step:
      out_intermediate[n_cur] = X
    
    if obs_func is not None:
      obs_func_j = obs_func[j] if time_varying_obs else obs_func
      HX = obs_func_j(X)
      y_obs[j] = noise_R_param(HX)

  return out, y_obs, out_intermediate, t_intermediate


# class ODEModel:
#   def solve_dt(self, u, t, dt, solver, model_Q):
#     if solver == "Euler":
#       return u + self.f(t, u) * dt + model_Q * math.sqrt(dt) * torch.randn_like(u)
#     elif solver == "RK4":
#       k1 = self.f(t, u)
#       k2 = self.f(t+dt/2, u+dt*k1/2)
#       k3 = self.f(t+dt/2, u+dt*k2/2)
#       k4 = self.f(t+dt, u+dt*k3)
#       return u + dt*(k1+2*k2+2*k3+k4)/6

#   def forward(self, u_start, t_start, t_end, dt, solver, model_Q, device, error_internal_step=True, save_internal_step=False):
#     u_cur, t_cur = u_start, t_start
#     if save_internal_step:  # save first but not last
#       n_int = round((t_end - t_start) / dt)
#       u_int = torch.zeros((n_int, self.x_dim), device=device)
#       u_int[0] = u_cur

#     if error_internal_step:
#       for j in range(round((t_end - t_start) / dt)):
#         u_cur = self.solve_dt(u_cur, t_cur, dt, solver, model_Q)
#         t_cur += dt
#         if save_internal_step and j != (round((t_end - t_start) / dt) - 1):
#           u_int[j+1] = u_cur
#     else:
#       for j in range(round((t_end - t_start) / dt)):
#         u_cur = self.solve_dt(u_cur, t_cur, dt, solver, 0)
#         t_cur += dt
#         if save_internal_step and j != (round((t_end - t_start) / dt) - 1):
#           u_int[j+1] = u_cur
#       u_cur += model_Q * math.sqrt(t_end - t_start) * torch.randn_like(u_cur)

#     if save_internal_step:
#       return u_cur, u_int
#     else:
#       return u_cur

#   def run(self, u0, t0, t_eval, device, dt=0.01, solver="RK4", model_Q=0, error_internal_step=True, save_internal_step=False):
#     n_eval = len(t_eval)
#     out = torch.zeros((n_eval, self.x_dim), device=device)   # n_eval * 3
#     if save_internal_step:
#       n_all = round((t_eval[-1] - t0) / dt) + 1
#       t_all = [(t0 + i * dt) for i in range(n_all)]
#       out_all = torch.zeros((n_all, self.x_dim), device=device)
#     u_cur, t_cur = u0, t0
#     for i in range(n_eval):  
#       if save_internal_step:
#         u_cur, u_int = self.forward(u_cur, t_cur, t_eval[i], dt, solver, model_Q, device, error_internal_step, save_internal_step)
#         index_start = round((t_cur - t0) / dt)
#         index_end = round((t_eval[i] - t0) / dt)
#         out_all[index_start:index_end, :] = u_int
#         t_cur = t_eval[i]
#         out[i] = u_cur
#       else:
#         u_cur = self.forward(u_cur, t_cur, t_eval[i], dt, solver, model_Q, error_internal_step, save_internal_step)
#         t_cur = t_eval[i]
#         out[i] = u_cur
#     if save_internal_step:
#       out_all[-1] = u_cur
#       return out, out_all, t_all
#     else:
#       return out
    
#     # if save_internal_step:
#     #   n_all = round((t_eval[-1] - t0) / dt) + 1
#     #   t_all = [(t0 + i * dt) for i in range(n_all)]
#     #   out = torch.zeros((n_all, self.x_dim), device=device)
#     #   u_cur, t_cur = u0, t0
#     #   for i in range(n_eval - 1):  
#     #     u_cur, u_int = self.forward(u_cur, t_cur, t_eval[i], dt, solver, model_Q, error_internal_step, save_internal_step)
#     #     t_cur = t_eval[i]
#     #     index_start = (t_eval[i] - t0) / dt
#     #     index_end = (t_eval[i+1] - t0) / dt
#     #     out[index_start:index_end, :] = u_int
#     #   out[-1] = u_cur
#     #   return out
#     # else:
#     #   n_eval = len(t_eval)
#     #   out = torch.zeros((n_eval, self.x_dim), device=device)   # n_eval * 3
#     #   u_cur, t_cur = u0, t0
#     #   for i in range(n_eval):  
#     #     u_cur = self.forward(u_cur, t_cur, t_eval[i], dt, solver, model_Q, error_internal_step, save_internal_step)
#     #     t_cur = t_eval[i]
#     #     out[i] = u_cur
#     #   return out




  