import utils
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from models import DAModel, DEModel, NNModel


def loss_grad_vis_1d(ode_func, obs_func, t_obs, y_obs, init_m, init_C_param, model_Q_param, noise_R_param, device, n_runs_per_param, run_Kalman, N_ensem_list, obj, params_1d, viz_type, ode_func_param=None, **enkf_kwargs):
  x_dim = ode_func.x_dim
  num_ensem_size = len(N_ensem_list)
  nll_ls = torch.zeros((num_ensem_size, params_1d.shape[0], n_runs_per_param), device=device)
  grad_ls = torch.zeros_like(nll_ls)
  nll_kalman = None
  grad_kalman = None
  # recon_ls = torch.zeros_like(nll_ls)
  for ne in tqdm(range(len(N_ensem_list))):
    for i in tqdm(range(params_1d.shape[0])):
      for n in tqdm(range(n_runs_per_param), leave=False):
        if obj == "ode_func":
          if viz_type == 1:
            ode_func = NNModel.Linear_ODE(x_dim, params_1d[i], param=ode_func_param).to(device)
          elif viz_type == 2:
            ode_func = NNModel.Linear_ODE_single_var(x_dim, params_1d[i]).to(device)
          elif viz_type == 3:
            coeff = torch.tensor([10., 8/3, 28.])
            coeff[0] = params_1d[i]
            ode_func = NNModel.Lorenz63(coeff, x_dim).to(device)
        elif obj == "model_Q":
          model_Q_param = nn.Parameter(params_1d[i]).to(device)
        X_track, X_intermediate, neg_log_likelihood = DAModel.EnKF(ode_func, obs_func, t_obs, y_obs, N_ensem=N_ensem_list[ne], init_m=init_m, init_C_param=init_C_param, model_Q_param=model_Q_param, noise_R_param=noise_R_param, device=device, **enkf_kwargs)
        neg_log_likelihood.sum().backward()
        if obj == "ode_func":
          if viz_type == 1 or viz_type == 2:
            grad_ls[ne, i, n] = ode_func.a.grad
          elif viz_type == 3:
            grad_ls[ne, i, n] = ode_func.coeff.grad[0]
        elif obj == "model_Q":
          grad_ls[ne, i, n] = model_Q_param.grad      
        nll_ls[ne, i, n] = neg_log_likelihood.sum().item()
        # recon_ls[n, i] = utils.mse_loss(E.X_smooth.mean(axis=1)[1:], out).detach().clone()
  if run_Kalman:
    nll_kalman = torch.zeros(params_1d.shape[0], device=device)
    grad_kalman = torch.zeros_like(nll_kalman)
    for i in tqdm(range(params_1d.shape[0])):
      if obj == "ode_func":
        if viz_type == 1:
          ode_func = NNModel.Linear_ODE(x_dim, params_1d[i], param=ode_func_param).to(device)
        elif viz_type == 2:
          ode_func = NNModel.Linear_ODE_single_var(x_dim, params_1d[i]).to(device)
      elif obj == "model_Q":
        if viz_type == 1:
          ode_func = NNModel.Linear_ODE(x_dim, ode_func.a, param=ode_func_param).to(device)
        elif viz_type == 2:
          ode_func = NNModel.Linear_ODE_single_var(x_dim, ode_func.a).to(device)
        model_Q_param = nn.Parameter(params_1d[i]).to(device)
      m_track, C_track, neg_log_likelihood = DAModel.KF(ode_func, obs_func, t_obs, y_obs, init_m, init_C_param, model_Q_param, noise_R_param, device, **enkf_kwargs)
      neg_log_likelihood.sum().backward()
      if obj == "ode_func":
        grad_kalman[i] = ode_func.a.grad
      elif obj == "model_Q":
        grad_kalman[i] = model_Q_param.grad
      nll_kalman[i] = neg_log_likelihood.sum().item()
  # As = torch.exp(As)
  if obj == "model_Q" and (enkf_kwargs["model_Q_type"] == "scalar" or enkf_kwargs["model_Q_type"] == "diag"):
    grad_ls = grad_ls / utils.softplus_grad(params_1d[None, :, None])
    if run_Kalman:
      grad_kalman = grad_kalman / utils.softplus_grad(params_1d)
  return nll_ls, nll_kalman, grad_ls, grad_kalman

def loss_grad_vis_2d(ode_func, obs_func, t_obs, y_obs, init_m, init_C_param, model_Q_param, noise_R_param, device, n_runs_per_param, run_Kalman, N_ensem_list, params_x, params_y, viz_type, ode_func_param=None, **enkf_kwargs):
  x_dim = ode_func.x_dim
  num_ensem_size = len(N_ensem_list)
  nll_ls = torch.zeros((num_ensem_size, params_x.shape[0], params_y.shape[0], n_runs_per_param), device=device)
  grad_ls = torch.zeros((num_ensem_size, params_x.shape[0], params_y.shape[0], 2, n_runs_per_param), device=device)
  nll_kalman = None
  grad_kalman = None
  # recon_ls = torch.zeros_like(nll_ls)
  for ne in tqdm(range(len(N_ensem_list))):
    for i in tqdm(range(params_x.shape[0]), leave=False):
      for j in range(params_y.shape[0]):
        for n in range(n_runs_per_param):
          if viz_type == 1:
            ode_func = NNModel.Linear_ODE(x_dim, params_x[i], param=ode_func_param).to(device)
            model_Q_param = nn.Parameter(params_y[j]).to(device)
            # ode_func = NNModel.Linear_ODE(x_dim, torch.tensor([params_x[i], params_y[j]]), param=ode_func_param).to(device)
          elif viz_type == 2:
            ode_func = NNModel.Linear_ODE_single_var(x_dim, params_x[i]).to(device)
            model_Q_param = nn.Parameter(params_y[j]).to(device)
          elif viz_type == 3:
            coeff = torch.tensor([10., 8/3, 28.])
            coeff[0] = params_x[i]
            ode_func = NNModel.Lorenz63(coeff, x_dim).to(device)
            model_Q_param = nn.Parameter(params_y[j]).to(device)
          
          X_track, X_intermediate, neg_log_likelihood = DAModel.EnKF(ode_func, obs_func, t_obs, y_obs, N_ensem=N_ensem_list[ne], init_m=init_m, init_C_param=init_C_param, model_Q_param=model_Q_param, noise_R_param=noise_R_param, device=device, **enkf_kwargs)
          neg_log_likelihood.sum().backward()
          if viz_type == 1 or viz_type == 2:
            grad_ls[ne, i, j, 0, n] = ode_func.a.grad
            grad_ls[ne, i, j, 1, n] = model_Q_param.grad
            # grad_ls[ne, i, j, :, n] = ode_func.a.grad
          elif viz_type == 3:
            grad_ls[ne, i, j, 0, n] = ode_func.coeff.grad[0]
            grad_ls[ne, i, j, 1, n] = model_Q_param.grad
                
          nll_ls[ne, i, j, n] = neg_log_likelihood.sum().item()
          # recon_ls[n, i] = utils.mse_loss(E.X_smooth.mean(axis=1)[1:], out).detach().clone()
  if run_Kalman:
    nll_kalman = torch.zeros(params_x.shape[0], params_y.shape[0], device=device)
    grad_kalman = torch.zeros(params_x.shape[0], params_y.shape[0], 2, device=device)
    for i in tqdm(range(params_x.shape[0])):
      for j in tqdm(range(params_y.shape[0]), leave=False):
        if viz_type == 1:
          ode_func = NNModel.Linear_ODE(x_dim, params_x[i], param=ode_func_param).to(device)
          model_Q_param = nn.Parameter(params_y[j]).to(device)
          # ode_func = NNModel.Linear_ODE(x_dim, torch.tensor([params_x[i], params_y[j]]), param=ode_func_param).to(device)
        elif viz_type == 2:
          ode_func = NNModel.Linear_ODE_single_var(x_dim, params_x[i]).to(device)
          model_Q_param = nn.Parameter(params_y[j]).to(device)
        
        m_track, C_track, neg_log_likelihood = DAModel.KF(ode_func, obs_func, t_obs, y_obs, init_m, init_C_param, model_Q_param, noise_R_param, device, **enkf_kwargs)
        neg_log_likelihood.sum().backward()
        grad_kalman[i, j, 0] = ode_func.a.grad
        grad_kalman[i, j, 1] = model_Q_param.grad
        # grad_kalman[i,j,:] = ode_func.a.grad
        nll_kalman[i, j] = neg_log_likelihood.sum().item()
  # As = torch.exp(As)
  if (enkf_kwargs["model_Q_type"] == "scalar" or enkf_kwargs["model_Q_type"] == "diag"):
    grad_ls[:, :, :, 1, :] = grad_ls[:, :, :, 1, :] / utils.softplus_grad(params_y[None, None, :, None]) 
    if run_Kalman:
      grad_kalman[:, :, 1] = grad_kalman[:, :, 1] / utils.softplus_grad(params_y[None, :])

  return nll_ls, nll_kalman, grad_ls, grad_kalman

