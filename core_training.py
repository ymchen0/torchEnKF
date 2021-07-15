import utils
from models import DEModel, DAModel, NNModel, Noise

from utils import Timer
import copy
import math
import random
import torch
import torch.nn as nn
import numpy as np
import collections
import metrics
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
from contextlib import ExitStack

def train_test_split(y_obs, n_train):
  """
  Args:
    y_obs: (n_obs, bs, y_dim)
  """
  train = y_obs[:, 0:n_train, :]  # (n_obs, n_train, y_dim)
  test = y_obs[:, n_train:, :]   # (n_obs, n_test, y_dim)
  return train, test

def get_batch(t_obs, y_obs, out, batch_length, n_draws):
  """
  Args: 
    t_obs: (n_obs)
    y_obs: (n_obs, *bs, y_dim) 
    out: (n_obs, *bs, x_dim)
    batch_length: Starting at one randomly drawn time t_start in t_obs, select a subsequence (t_start, t_start+1, ..., t_start+bacth_length-1)
    n_draws: Repeat the above `n_draws' time and stack them. This creates a new batch dimension.
  """
  n_obs = t_obs.shape[0]
  start = np.random.choice(n_obs-batch_length+1, n_draws, replace=False)
  batch_y_obs = torch.stack([y_obs[i:i+batch_length] for i in start], dim=-2)  # (batch_length, *bs, n_draws, y_dim)
  batch_out = torch.stack([out[i:i+batch_length] for i in start], dim=-2)  # (batch_length, *bs, n_draws, x_dim)
  return t_obs[:batch_length], batch_y_obs, batch_out, utils.shrink_batch_dim(batch_y_obs)[:, 0, :, :], utils.shrink_batch_dim(batch_out)[:, 0, :, :]

def get_em_loss(X_track, ode_func, obs_func, t_obs, y_obs, init_m, init_C_param, model_Q_param, noise_R_param, device, track_Q=None,
                        ode_method='rk4', ode_options=None, adjoint=False, adjoint_method=None, adjoint_options=None, init_C_type="diag", model_Q_type="scalar", noise_R_type="scalar", t0=0., time_varying_obs=False, **unused_enkf_kwargs):
  """
  Args:
    X_track: (n_obs+1, *bs, N_ensem, x_dim)
    y_obs: (n_obs, *bs, y_dim)
  """

  ode_integrator = odeint_adjoint if adjoint else odeint
  
  n_obs = X_track.shape[0]-1
  if y_obs is not None:
    y_dim = y_obs.shape[-1]
  x_dim = X_track.shape[-1]
  N_ensem = X_track.shape[-2]
  


  t_cur = t0

  X = X_track[0] # (*bs, N_ensem, x_dim)
  nll_suro_em = torch.zeros(X.shape[:-1], device=device) # (*bs, N_ensem)

  if init_m is not None:
    init_C_chol = init_C_param.chol()
    init_d = torch.distributions.MultivariateNormal(init_m, scale_tril=init_C_chol)
    nll_suro_em += -init_d.log_prob(X) # (*bs, N_ensem)

  if track_Q is not None:
    if track_Q == "scalar":
      Q_track = torch.zeros(X.shape[:-2], device=device) # (*bs)
    elif track_Q == "diag":
      Q_track = torch.zeros(*X.shape[:-2], x_dim, device=device) # (*bs, x_dim)
    else:
      Q_track = torch.zeros(*X.shape[:-2], x_dim, x_dim, device=device) # (*bs, x_dim, x_dim)

  for j in range(n_obs):
    X_prev = X # (*bs, N_ensem, x_dim)
    if adjoint:
      _, X = ode_integrator(ode_func, X, torch.tensor([t_cur, t_obs[j]], device=device), method=ode_method, options=ode_options, adjoint_method=adjoint_method, adjoint_options=adjoint_options)
    else:
      _, X = ode_integrator(ode_func, X, torch.tensor([t_cur, t_obs[j]], device=device), method=ode_method, options=ode_options)
    t_cur = t_obs[j]

    if model_Q_param is not None:
      model_Q_chol = model_Q_param.chol(X_prev) # (x_dim, x_dim) or (*bs, N_ensem, x_dim, x_dim) 
    state_d = torch.distributions.MultivariateNormal(torch.zeros(x_dim, device=device), scale_tril=model_Q_chol)
    nll_suro_em += -state_d.log_prob(X_track[j+1] - X) # (*bs, N_ensem)

    if track_Q is not None:
      if track_Q == "scalar":
        Q_track += ((X_track[j+1] - X)**2).mean(dim=-1).mean(dim=-1) # (*bs)
      elif track_Q == "diag":
        Q_track += ((X_track[j+1] - X)**2).mean(dim=-2) #(*bs, x_dim)
      else:
        Q_track += ((X_track[j+1] - X).transpose(-1, -2)) @ (X_track[j+1] - X) / N_ensem #(*bs, x_dim, x_dim)


    X = X_track[j+1]

    if y_obs is not None:
      obs_func_j = obs_func[j] if time_varying_obs else obs_func
      y_obs_j = y_obs[j].unsqueeze(-2) # (*bs, 1, y_dim)
      if noise_R_param is not None:
        noise_R_chol = noise_R_param.chol()
      
      HX = obs_func_j(X) # (*bs, N_ensem, y_dim)
      obs_d = torch.distributions.MultivariateNormal(torch.zeros(y_dim, device=device), scale_tril=noise_R_chol)
      nll_suro_em += -obs_d.log_prob(y_obs_j - HX) # (*bs, N_ensem)

  if track_Q is not None:
    if track_Q == "scalar":
      next_Q = Noise.AddGaussian(x_dim, torch.sqrt(Q_track.mean()), 'scalar')
    elif track_Q == "diag":
      next_Q = Noise.AddGaussian(x_dim, torch.sqrt(utils.mean_over_all_but_last_k_dims(Q_track, 1)), 'diag')
    else:
      next_Q = Noise.AddGaussian(x_dim, utils.mean_over_all_but_last_k_dims(Q_track, 2), 'full')
    return nll_suro_em, next_Q
  else:
    return nll_suro_em, 0

def train_loop_em_new(ode_func, obs_func, t_obs, y_obs, out, out_intermediate, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, optimizer, n_epochs, batch_length, bs, 
                                  save_location=None, scheduler=None, print_every_n_epochs=None, true_system="L63",true_ode_func=None,  u0_test=None, draw_stats=False, test_every_n_epochs=None, test_filter=True,
                                  track_Q=False, lr_decay=None,  train_size=1, save_every_n_epochs=5, start_epoch=0, m_step_type="mean", m_steps=1, clip_norm=None, monitor=None, t0=0, timer=False, **enkf_kwargs):
  # time_varying_obs = enkf_kwargs.get('time_varying_obs', False)
  likelihood_warmup = enkf_kwargs.get('likelihood_warmup', 0)
  step_size = enkf_kwargs['ode_options']['step_size']
  n_obs = y_obs.shape[0]
  if len(y_obs.shape) == 2:
    y_obs, out = y_obs.unsqueeze(-2), out.unsqueeze(-2)
  
  if test_every_n_epochs is not None:
    y_obs_train, y_obs_test = train_test_split(y_obs, n_train=train_size) # (n_obs, n_train, y_dim) and (n_obs, n_test, y_dim)
    out_train, out_test = train_test_split(out, n_train=train_size) # (n_obs, n_train, x_dim) and (n_obs, n_test, x_dim)
  else:
    y_obs_train, out_train = y_obs, out
    test_neg_log_likelihood = torch.tensor(0.0, device=device)
    test_mse = one_step_fs = 0

  monitor_res = torch.zeros(0, device=device)
  batch_y_obs = y_obs_train.repeat(1, bs, 1) # (n_obs, bs*n_train, x_dim)

  if monitor is not None:
    if test_every_n_epochs is not None:
      test_neg_log_likelihood, test_mse, one_step_fs = test_epoch(ode_func, obs_func, t_obs, y_obs_test, out_test, init_m, init_C_param, model_Q_param, noise_R_param, device, None, None, None, test_filter,
                                                                u0_test, method_test='enkf', true_system=true_system, true_ode_func=true_ode_func, **enkf_kwargs)
    res = monitor(ode_func, model_Q_param, noise_R_param, torch.tensor(0., device=device), test_neg_log_likelihood, test_mse, one_step_fs, None, "diff", device)  # (1, n_monitors)
    monitor_res = torch.cat((monitor_res, res), dim=0) # (n_iter, n_monitors)

  for epoch in range(start_epoch, n_epochs):
    with ExitStack() if not timer else Timer("Filter"):
      with torch.no_grad():
        X, X_track, X_intermediate, train_neg_log_likelihood = DAModel.EnKF(ode_func, obs_func, t_obs, batch_y_obs, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, t0=t0, **enkf_kwargs) # (n_obs, bs*n_train, N_ensem, x_dim)
      if m_step_type == "mean":
        X_track_em = X_track.mean(dim=-2, keepdim=True) # (n_obs+1, bs*n_train, 1, x_dim)
      elif m_step_type == "all":
        X_track_em = X_track  # (n_obs+1, bs*n_train, N_ensem, x_dim)
      # optimizer.state=collections.defaultdict(dict)

      # if track_Q is not None:
      #   permutation = torch.arange(n_obs)
      #   indices = permutation[likelihood_warmup:]
      #   nn_input, nn_label = X_track_em[indices], X_track_em[indices+1]
      #   nn_input_label = torch.stack((nn_input, nn_label)) # (2, bs, 1, x_dim)
      #   _, model_Q_param = get_em_loss(nn_input_label, ode_func, obs_func, torch.tensor([t_obs[0]]), None, None, None, model_Q_param, noise_R_param, device, track_Q=track_Q, **enkf_kwargs)
      
      for step in range(m_steps):
        permutation = torch.randperm(n_obs-likelihood_warmup)+likelihood_warmup  # (likelihood_warmup to n_obs)
        # permutation = torch.arange(n_obs)
        for i in range(0, n_obs-likelihood_warmup, batch_length):
          optimizer.zero_grad() 
          indices = permutation[i:i+batch_length]
          nn_input, nn_label = X_track_em[indices], X_track_em[indices+1] # (batch_length_obs, bs*n_train, N_ensem, x_dim), (batch_length_obs+1, bs*n_train, N_ensem(1), x_dim)
          nn_input_label = torch.stack((nn_input, nn_label)) # (2, batch_length_obs, bs*n_train, N_ensem(1), x_dim)
          nll_suro_em, _ = get_em_loss(nn_input_label, ode_func, obs_func, torch.tensor([t_obs[0]]), None, None, None, model_Q_param, noise_R_param, device, track_Q=None, **enkf_kwargs)
          nll_suro_em.sum().backward()
          optimizer.step()
      
      if track_Q is not None:
        permutation = torch.arange(n_obs)
        indices = permutation[likelihood_warmup:]
        nn_input, nn_label = X_track_em[indices], X_track_em[indices+1]
        nn_input_label = torch.stack((nn_input, nn_label)) # (2, bs, 1, x_dim)
        _, model_Q_param = get_em_loss(nn_input_label, ode_func, obs_func, torch.tensor([t_obs[0]]), None, None, None, model_Q_param, noise_R_param, device, track_Q=track_Q, **enkf_kwargs)
      
      if lr_decay is not None and epoch >= 10:
        if epoch == 10:
          initial_lr = [g['lr'] for g in optimizer.param_groups]
        for i in range(len(optimizer.param_groups)):
          optimizer.param_groups[i]['lr'] = initial_lr[i] / math.pow(epoch+1-10, lr_decay)


    if test_every_n_epochs is not None and epoch % test_every_n_epochs == 0:
      test_neg_log_likelihood, test_mse, one_step_fs = test_epoch(ode_func, obs_func, t_obs, y_obs_test, out_test, init_m, init_C_param, model_Q_param, noise_R_param, device, None, None, None, test_filter,
                                                                u0_test, method_test='enkf', true_system=true_system, true_ode_func=true_ode_func, **enkf_kwargs)
      
    
    if n_epochs <= 300 or epoch % 10 == 0:
      print(f"Epoch {epoch}: nll = {train_neg_log_likelihood.mean().cpu().item()}, test_nll = {test_neg_log_likelihood.mean().cpu().item()}")


    if print_every_n_epochs is not None and epoch % print_every_n_epochs == 0:
      print_epoch(ode_func, obs_func, t_obs, y_obs_test, out_test, init_m, init_C_param, model_Q_param, noise_R_param, device, u0_test, method_test='enkf', true_system=true_system, true_ode_func=true_ode_func, draw_stats=draw_stats, **enkf_kwargs)


    if scheduler is not None:
      scheduler.step()

    if monitor is not None:
      res = monitor(ode_func, model_Q_param, noise_R_param, train_neg_log_likelihood, test_neg_log_likelihood, test_mse, one_step_fs, None, "diff", device)  # (1, n_monitors)
      monitor_res = torch.cat((monitor_res, res), dim=0) # (n_iter, n_monitors)
      if n_epochs <= 300 or epoch % 10 == 0:
        print(res)

    # if epoch % save_every_n_epochs == 0:
      # torch.save({'model_state_dict': net_state, 'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict() }, save_location)
  return monitor_res, model_Q_param



def train_loop_diff(ode_func, obs_func, t_obs, y_obs, out, out_intermediate, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, optimizer, n_epochs, bs, proposal='optimal', adaptive_resampling=False, detach_every=None,
                                    save_location=None, scheduler=None, print_every_n_epochs=None, true_system="L63", true_ode_func=None, u0_test=None, draw_stats=False, save_every_n_epochs=1, test_every_n_epochs=None, test_filter=True,
                                    tbptt=None, lr_decay=None, l2_reg=None, start_epoch=0, method="enkf", train_size=1, clip_norm=None, monitor=None, t0=0, timer=False, **enkf_kwargs):
  """
  Args: 
    y_obs: batch of observation data # (n_obs, y_dim) or (n_obs, n_train+n_test, y_dim)
  """
  time_varying_obs = enkf_kwargs.get('time_varying_obs', False)
  likelihood_warmup = enkf_kwargs.get('likelihood_warmup', 0)
  step_size = enkf_kwargs['ode_options']['step_size']
  n_obs = y_obs.shape[0]
  # model.eval() if optimizer is None else model.train()
  if len(y_obs.shape) == 2:
    y_obs, out = y_obs.unsqueeze(-2), out.unsqueeze(-2)
  
  if test_every_n_epochs is not None:
    y_obs_train, y_obs_test = train_test_split(y_obs, n_train=train_size) # (n_obs, n_train, y_dim) and (n_obs, n_test, y_dim)
    out_train, out_test = train_test_split(out, n_train=train_size) # (n_obs, n_train, x_dim) and (n_obs, n_test, x_dim)
  else:
    y_obs_train, out_train = y_obs, out
    test_neg_log_likelihood = torch.tensor(0.0, device=device)
    test_mse = one_step_fs = 0
  
  if tbptt is None:
    tbptt = n_obs

  monitor_res = torch.zeros(0, device=device)
  batch_y_obs = y_obs_train.repeat(1, bs, 1)
  
  if monitor is not None:
    if test_every_n_epochs is not None:
      test_neg_log_likelihood, test_mse, one_step_fs = test_epoch(ode_func, obs_func, t_obs, y_obs_test, out_test, init_m, init_C_param, model_Q_param, noise_R_param, device, proposal, adaptive_resampling, detach_every, test_filter,
                                                                u0_test, method_test=method, true_system=true_system, true_ode_func=true_ode_func, **enkf_kwargs)
    res = monitor(ode_func, model_Q_param, noise_R_param, torch.tensor([0.], device=device), test_neg_log_likelihood, test_mse, one_step_fs, None, "diff", device)  # (1, n_monitors)
    monitor_res = torch.cat((monitor_res, res), dim=0) # (n_iter, n_monitors)

  iter_count = 0
  for epoch in range(start_epoch, n_epochs):
    train_neg_log_likelihood = torch.zeros(bs*train_size, device=device)
    with ExitStack() if not timer else Timer("Filter"):
      # burn in, no grad
      with torch.no_grad():
        cur_obs_func = obs_func[:likelihood_warmup] if time_varying_obs else obs_func
        if method == 'enkf':
          X, X_track, X_intermediate, neg_log_likelihood = DAModel.EnKF(ode_func, cur_obs_func, t_obs[:likelihood_warmup], batch_y_obs[:likelihood_warmup], N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, detach_every=detach_every, t0=t0, **enkf_kwargs)
        elif method == 'kf':
          m_track, C_track, _, neg_log_likelihood = DAModel.KF(ode_func, cur_obs_func, t_obs[:likelihood_warmup], batch_y_obs[:likelihood_warmup], init_m, init_C_param, model_Q_param, noise_R_param, device, **enkf_kwargs)
        elif method == 'pf':
          X, w, X_track, X_intermediate, w_track, w_intermediate, neg_log_likelihood = DAModel.BootstrapPF(ode_func, cur_obs_func, t_obs[:likelihood_warmup], batch_y_obs[:likelihood_warmup], N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, proposal, adaptive_resampling, detach_every=detach_every, t0=t0, **enkf_kwargs)
        train_neg_log_likelihood += neg_log_likelihood
        if l2_reg is not None:
          train_neg_log_likelihood += -l2_reg(ode_func, model_Q_param)
        t_start = t_obs[likelihood_warmup-1] if likelihood_warmup >= 1 else t0

      # learning
      for start in range(likelihood_warmup, n_obs, tbptt):
        optimizer.zero_grad()
        end = min(start + tbptt, n_obs)
        cur_obs_func = obs_func[start:end] if time_varying_obs else obs_func
        if method == 'enkf':
          X, X_track, X_intermediate, neg_log_likelihood = DAModel.EnKF(ode_func, cur_obs_func, t_obs[start:end], batch_y_obs[start:end], N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, detach_every=detach_every, t0=t_start, init_X=X, **enkf_kwargs)
          X = X.detach()
        elif method == 'kf':
          m_track, C_track, _, neg_log_likelihood = DAModel.KF(ode_func, cur_obs_func, t_obs[start:end], batch_y_obs[start:end], init_m, init_C_param, model_Q_param, noise_R_param, device, **enkf_kwargs)
        elif method == 'pf':
          X, w, X_track, X_intermediate, w_track, w_intermediate, neg_log_likelihood = DAModel.BootstrapPF(ode_func, cur_obs_func, t_obs[start:end], batch_y_obs[start:end], N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, proposal, adaptive_resampling, detach_every=detach_every, t0=t_start, init_w=w, init_X=X,  **enkf_kwargs)
          X = X.detach()
          w = w.detach()
        t_start = t_obs[end-1]
        loss = (-neg_log_likelihood)
        if l2_reg is not None:
          loss += l2_reg(ode_func, model_Q_param)
        loss.mean().backward()
        train_neg_log_likelihood += (-loss).detach().clone()
        if clip_norm is not None:
          torch.nn.utils.clip_grad_norm_(ode_func.parameters(), clip_norm)
          torch.nn.utils.clip_grad_norm_(model_Q_param.parameters(), clip_norm)

        # iter_count += 1
        # if lr_decay is not None:
        #   if iter_count == 1:
        #     initial_lr = [g['lr'] for g in optimizer.param_groups]
        #   for i in range(len(optimizer.param_groups)):
        #     optimizer.param_groups[i]['lr'] = initial_lr[i] / math.pow(iter_count, lr_decay)
        
        optimizer.step()
        # if scheduler is not None:
        #   scheduler.step()
        
        
      if lr_decay is not None and epoch >= 10:
        if epoch == 10:
          initial_lr = [g['lr'] for g in optimizer.param_groups]
        for i in range(len(optimizer.param_groups)):
          optimizer.param_groups[i]['lr'] = initial_lr[i] / (math.pow(epoch+1-10, lr_decay))
    


  
    if test_every_n_epochs is not None and epoch % test_every_n_epochs == 0:
      test_neg_log_likelihood, test_mse, one_step_fs = test_epoch(ode_func, obs_func, t_obs, y_obs_test, out_test, init_m, init_C_param, model_Q_param, noise_R_param, device, proposal, adaptive_resampling, detach_every, test_filter,
                                                                u0_test, method_test=method, true_system=true_system, true_ode_func=true_ode_func, **enkf_kwargs)
      
    
    if n_epochs <= 300 or epoch % 10 == 0:
      print(f"Epoch {epoch}: nll = {train_neg_log_likelihood.mean().cpu().item()}, test_nll = {test_neg_log_likelihood.mean().cpu().item()}")


    if print_every_n_epochs is not None and epoch % print_every_n_epochs == 0:
      print_epoch(ode_func, obs_func, t_obs, y_obs_test, out_test, init_m, init_C_param, model_Q_param, noise_R_param, device, u0_test, method_test=method, true_system=true_system, true_ode_func=true_ode_func, draw_stats=draw_stats, **enkf_kwargs)


    if scheduler is not None:
      scheduler.step()

    if monitor is not None:
      res = monitor(ode_func, model_Q_param, noise_R_param, train_neg_log_likelihood, test_neg_log_likelihood, test_mse, one_step_fs, None, "diff", device)  # (1, n_monitors)
      monitor_res = torch.cat((monitor_res, res), dim=0) # (n_iter, n_monitors)
      if n_epochs <= 300 or epoch % 10 == 0:
        print(res)

    # if epoch % save_every_n_epochs == 0:
      # torch.save({'model_state_dict': net_state, 'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict() }, save_location)
  return monitor_res


def test_epoch(ode_func, obs_func, t_obs, y_obs_test, out_test, init_m, init_C_param, model_Q_param, noise_R_param, device, proposal, adaptive_resampling, detach_every, test_filter,
                                                                u0_test, method_test, true_system, true_ode_func=None, N_ensem_test=100, **enkf_kwargs):
  enkf_kwargs_copy = enkf_kwargs.copy() # avoid in-place modification
  enkf_kwargs_copy['save_filter_step'] = True
  enkf_kwargs_copy['save_first'] = False
  step_size = enkf_kwargs_copy['ode_options']['step_size']
  burn_in = t_obs.shape[0] // 5
  test_neg_log_likelihood = test_mse = one_step_fs = torch.tensor(0., device=device)

  k = 0
  with torch.no_grad():
    while True:
      # try:
      if test_filter:
        if method_test == 'enkf':
          _, X_track_test, _, test_neg_log_likelihood = DAModel.EnKF(ode_func, obs_func, t_obs, y_obs_test, N_ensem_test, init_m, init_C_param, model_Q_param, noise_R_param, device, detach_every=detach_every, **enkf_kwargs_copy) 
          if true_system == "L63": 
            test_mse1 = torch.sqrt(utils.mse_loss_last_dim(X_track_test.mean(dim=-2)[burn_in:], out_test[burn_in:]))
            test_mse2 = torch.sqrt(utils.mse_loss(X_track_test.mean(dim=-2)[burn_in:], out_test[burn_in:])).unsqueeze(0)
            test_mse = torch.cat((test_mse1, test_mse2))
          elif true_system == "L96":
            test_mse = torch.sqrt(utils.mse_loss(X_track_test.mean(dim=-2)[burn_in:], out_test[burn_in:]))

        elif method_test == 'pf':
          _, _, X_track_test, _, w_track_test, _, test_neg_log_likelihood = DAModel.BootstrapPF(ode_func, obs_func, t_obs, y_obs_test, N_ensem_test, init_m, init_C_param, model_Q_param, noise_R_param, device, proposal, adaptive_resampling, detach_every=detach_every, **enkf_kwargs_copy) 
          if true_system == "L63": 
            test_mse1 = torch.sqrt(utils.particle_mse_loss_last_dim(X_track_test[burn_in:], out_test[burn_in:], w_track_test[burn_in:]))
            test_mse2 = torch.sqrt(utils.particle_mse_loss(X_track_test[burn_in:], out_test[burn_in:], w_track_test[burn_in:])).unsqueeze(0)
            test_mse = torch.cat((test_mse1, test_mse2))
          elif true_system == "L96":
            test_mse = torch.sqrt(utils.particle_mse_loss(X_track_test[burn_in:], out_test[burn_in:], w_track_test[burn_in:]))
      
      if true_system == "L63": 
        one_step_fs = metrics.forecast_skill(true_ode_func, ode_func, torch.tensor([0.05, 0.1, 0.2, 0.4],device=device), u0_test, device, 'rk4', False)
      elif true_system == "L96":
        one_step_fs2 = metrics.forecast_skill(true_ode_func, ode_func, torch.tensor([0.05],device=device), u0_test, device, 'rk4', False)
        one_step_fs = metrics.forecast_skill(true_ode_func, ode_func, torch.tensor([0.01, 0.02, 0.05, 0.1],device=device), u0_test, device, 'rk4', False)
        one_step_fs = torch.cat((one_step_fs2, one_step_fs))
      break
      # except RuntimeError:
      #   print("Warning! Filter divergence while testing.")
      #   k += 1
      #   if k >= 10:
      #     raise RuntimeError('Filter divergence too often...')
  
  return test_neg_log_likelihood, test_mse, one_step_fs

def print_epoch(ode_func, obs_func, t_obs, y_obs_test, out_test, init_m, init_C_param, model_Q_param, noise_R_param, device, 
                                                                u0_test, method_test, true_system, draw_stats, true_ode_func=None, N_ensem_test=100, **enkf_kwargs):
  with torch.no_grad():
    if true_system == "L63":
      t_obs_test = torch.arange(0.01, 20, 0.01, device=device)
      rdn = random.randrange(u0_test.shape[0])
      out_test1, _, _, _ = DEModel.generate_data(ode_func, None, t_obs_test, u0_test[rdn], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.01), adjoint=False, save_intermediate_step=False)
      out_test_true1, _, _, _ = DEModel.generate_data(true_ode_func, None, t_obs_test, u0_test[rdn], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.01),adjoint=False, save_intermediate_step=False)
      if (isinstance(model_Q_param, Noise.AddGaussianNet) or isinstance(model_Q_param, Noise.AddGaussianNet_from_basenet)):
        if (model_Q_param.param_type in {'scalar',  'diag'}):
          error_test = torch.linalg.norm(model_Q_param.q_true(out_test1), dim=-1)
        elif (model_Q_param.param_type == 'tril'):
          error_test = torch.linalg.norm(model_Q_param.q_true(out_test1), dim=(-1,-2))
      else:
        error_test = None
      utils.plot_dynamic(t_obs_test, out_test_true1, t_obs=None, y_obs=out_test1,online_color=error_test, fig_num_limit=3, text="test", contour_plot=False)
    elif true_system == "L96":
      t_obs_test = torch.arange(0.05, 10, 0.05, device=device)
      rdn = random.randrange(u0_test.shape[0])
      out_test1, _, _, _ = DEModel.generate_data(ode_func, None, t_obs_test, u0_test[rdn], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.01),adjoint=False, save_intermediate_step=False)
      out_test_true1, _, _, _ = DEModel.generate_data(true_ode_func, None, t_obs_test, u0_test[rdn], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.01),adjoint=False, save_intermediate_step=False)
      utils.plot_dynamic(t_obs_test, out_test_true1, t_obs=None, y_obs=out_test1, fig_num_limit=5, text="test", contour_plot=True)
  if draw_stats:
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=False)
    with torch.no_grad():
      fs = metrics.forecast_skill(true_ode_func, ode_func, t_obs_test, u0_test, device, 'rk4', False)
    axes[0].plot(t_obs_test.cpu().numpy(), fs.cpu().numpy())
    axes[0].set_title('Forecast Skill')
    spec = metrics.lyapunov_exponent(ode_func, step_size, step_size, 1000, u0_test[3], device, 'rk4')
    spec_true = metrics.lyapunov_exponent(true_ode_func, step_size, step_size, 1000, u0_test[3], device, 'rk4')
    axes[1].plot(spec, label='learned')
    axes[1].plot(spec_true, label='Truth')
    axes[1].set_title("Lyapunov spectrum")
    axes[1].legend()
    t_test = torch.arange(0, 120, 0.05)[1:].to(device)
    with torch.no_grad():
      f, Pxx_den=metrics.power_spectrum_density(ode_func, t_test, u0_test[0], device,'rk4',512)#, model_Q=0.01*torch.eye(x_dim, device=device))
      f, Pxx_den_true=metrics.power_spectrum_density(true_ode_func, t_test, u0_test[0], device,'rk4',512)
    axes[2].semilogy(f, Pxx_den, label='learned')
    axes[2].semilogy(f, Pxx_den_true, label='Truth')
    axes[2].set_title("Power Spectrum density")
    axes[2].set_xlabel('frequency')
    axes[2].set_ylabel('Density')
    axes[2].legend()
    plt.show()
  return


















# def train_loop_em(ode_func, obs_func, t_obs, y_obs, out, out_intermediate, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, optimizer, n_epochs, batch_length, bs, 
#                                   save_location=None, scheduler=None, print_every_n_epochs=None, true_system="L63",true_ode_func=None,  u0_test=None, draw_stats=False, test_every_n_epochs=None, tbptt=None, track_Q=False, lr_decay=None,  train_size=1, save_every_n_epochs=5, start_epoch=0, m_step_type="mean", m_steps=1, clip_norm=None, monitor=None, t0=0, **enkf_kwargs):
#   """
#   Only for equally-spaced observations
#   Args: 
#     y_obs: batch of observation data # (*bs, y_dim)
#   """
#   likelihood_warmup = enkf_kwargs.get('likelihood_warmup', 0)
#   step_size = enkf_kwargs['ode_options']['step_size']
#   # model.eval() if optimizer is None else model.train()
#   if len(y_obs.shape) == 2:
#     y_obs = y_obs.unsqueeze(-2)
#     out = out.unsqueeze(-2)
  
#   if test_every_n_epochs is not None:
#     y_obs_train, y_obs_test = train_test_split(y_obs, n_train=train_size) # (n_obs, n_train, y_dim) and (n_obs, n_test, y_dim)
#     out_train, out_test = train_test_split(out, n_train=train_size) # (n_obs, n_train, x_dim) and (n_obs, n_test, x_dim)
#   else:
#     y_obs_train = y_obs
#     out_train = out
#     test_neg_log_likelihood = torch.tensor(0.0, device=device)
#     test_mse = 0
#     one_step_fs = 0
#   monitor_res = torch.zeros(0, device=device)
#   for epoch in range(start_epoch, n_epochs):
#     if tbptt is None:
#       batch_t_obs, batch_y_obs, batch_out, first_y_obs, first_out = get_batch(t_obs, y_obs_train, out_train, batch_length, bs) # (batch_length), (batch_length, *bs, n_draws, y_dim), (batch_length, *bs, n_draws, x_dim)
#       with torch.no_grad():
#         # (batch_length+1, *bs, n_draws, N_ensem, x_dim)
#         X, X_track, X_intermediate, neg_log_likelihood = DAModel.EnKF(ode_func, obs_func, batch_t_obs, batch_y_obs, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, **enkf_kwargs)

#       if m_step_type == "mean":
#         X_track_m = X_track.mean(dim=-2, keepdim=True)
#         for step in range(m_steps):
#           optimizer.zero_grad()
#           nll_suro_em = get_em_loss(X_track_m, ode_func, obs_func, batch_t_obs, batch_y_obs, init_m, init_C_param, model_Q_param, noise_R_param, device, **enkf_kwargs)
#           nll_suro_em.mean().backward()
#           if clip_norm is not None:
#             torch.nn.utils.clip_grad_norm_(ode_func.parameters(), clip_norm)
#             torch.nn.utils.clip_grad_norm_(model_Q_param, clip_norm)
#           optimizer.step()
#       elif m_step_type == "all":
#         for step in range(m_steps):
#           optimizer.zero_grad()
#           nll_suro_em = get_em_loss(X_track, ode_func, obs_func, batch_t_obs, batch_y_obs, init_m, init_C_param, model_Q_param, noise_R_param, device, **enkf_kwargs)
#           nll_suro_em.mean().backward()
#           if clip_norm is not None:
#             torch.nn.utils.clip_grad_norm_(ode_func.parameters(), clip_norm)
#             torch.nn.utils.clip_grad_norm_(model_Q_param, clip_norm)
#           optimizer.step()
#     elif tbptt == 1:
#       batch_t_obs, batch_y_obs, batch_out, first_y_obs, first_out = t_obs, y_obs_train.repeat(1,batch_length,1), out_train, y_obs_train, out_train
#       while True:
#         try:
#           with torch.no_grad():
#             with Timer("Filter"):
#               # (n_obs+1, 1, N_ensem, x_dim)
#               X, X_track, X_intermediate, neg_log_likelihood = DAModel.EnKF(ode_func, obs_func, batch_t_obs, batch_y_obs, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, **enkf_kwargs)
#           break
#         except RuntimeError:
#           print("Warning! Filter divergence while training.")
#       n_obs = X_track.shape[0] - 1
#       if m_step_type == "mean":
#         X_track_m = X_track.mean(dim=-2, keepdim=True)
#         # optimizer.state=collections.defaultdict(dict)
#         for step in range(m_steps):
#           permutation = torch.randperm(n_obs)
#           # permutation = torch.arange(n_obs)
#           for i in range(likelihood_warmup, n_obs, bs):
#             optimizer.zero_grad() 
#             indices = permutation[i:i+bs]
#             nn_input, nn_label = X_track_m[indices], X_track_m[indices+1] # (bs, 1, x_dim)
#             nn_input_label = torch.stack((nn_input, nn_label)) # (2, bs, 1, x_dim)
#             nll_suro_em = get_em_loss(nn_input_label, ode_func, obs_func, torch.tensor([t_obs[0]]), None, None, None, model_Q_param, noise_R_param, device, **enkf_kwargs)
#             nll_suro_em.sum().backward()
#             optimizer.step()
#         if track_Q:
#           permutation = torch.arange(n_obs)
#           indices = permutation[likelihood_warmup:]
#           nn_input, nn_label = X_track_m[indices], X_track_m[indices+1]
#           nn_input_label = torch.stack((nn_input, nn_label)) # (2, bs, 1, x_dim)
#           _, model_Q_param = get_em_loss(nn_input_label, ode_func, obs_func, torch.tensor([t_obs[0]]), None, None, None, model_Q_param, noise_R_param, device, track_Q=1, **enkf_kwargs)
#       elif m_step_type == "all":  
#         for step in range(m_steps):
#           permutation = torch.randperm(n_obs)
#           for i in range(likelihood_warmup, n_obs, bs):
#             optimizer.zero_grad() 
#             indices = permutation[i:i+bs]
#             nn_input, nn_label = X_track[indices], X_track[indices+1] # (bs, N_ensem, x_dim)
#             nn_input_label = torch.stack((nn_input, nn_label)) # (2, bs, N_ensem, x_dim)
#             nll_suro_em = get_em_loss(nn_input_label, ode_func, obs_func, torch.tensor([t_obs[0]]), None, None, None, model_Q_param, noise_R_param, device, **enkf_kwargs)
#             nll_suro_em.sum(dim=0).mean().backward()
#             optimizer.step()
#         if track_Q:
#           permutation = torch.arange(n_obs)
#           indices = permutation[likelihood_warmup:]
#           nn_input, nn_label = X_track[indices], X_track[indices+1]
#           nn_input_label = torch.stack((nn_input, nn_label)) # (2, bs, 1, x_dim)
#           _, model_Q_param = get_em_loss(nn_input_label, ode_func, obs_func, torch.tensor([t_obs[0]]), None, None, None, model_Q_param, noise_R_param, device, track_Q=1, **enkf_kwargs)
      
      



#       if lr_decay is not None:
#         if epoch == 0:
#           initial_lr = [g['lr'] for g in optimizer.param_groups]
#         for i in range(len(optimizer.param_groups)):
#           optimizer.param_groups[i]['lr'] = initial_lr[i] / math.pow(epoch+1, lr_decay)
#     # elif tbptt == 2:
#     #   batch_t_obs, batch_y_obs, batch_out, first_y_obs, first_out = t_obs, y_obs_train.repeat(1,batch_length,1), out_train, y_obs_train, out_train
#     #   while True:
#     #     try:
#     #       with torch.no_grad():
#     #         with Timer("Filter"):
#     #           # (n_obs+1, 1, N_ensem, x_dim)
#     #           X, X_track, X_intermediate, neg_log_likelihood = DAModel.EnKF(ode_func, obs_func, batch_t_obs, batch_y_obs, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, **enkf_kwargs)
#     #       break
#     #     except RuntimeError:
#     #       print("Warning! Filter divergence while training.")
#     #   X_track_m = X_track.mean(dim=-2, keepdim=True)
#     #   n_obs = X_track.shape[0] - 1
#     #   permutation = torch.arange(n_obs)
#     #   indices = permutation[likelihood_warmup:]
#     #   if m_step_type == "mean":
#     #     nn_input, nn_label = X_track_m[indices], X_track_m[indices+1]
#     #   elif m_step_type == "all":
#     #     nn_input, nn_label = X_track[indices], X_track[indices+1]
#     #   nn_input_label = torch.stack((nn_input, nn_label)) # (2, bs, 1, x_dim)
#     #   nll_suro_em, model_Q_param = get_em_loss(nn_input_label, ode_func, obs_func, torch.tensor([t_obs[0]]), None, None, None, model_Q_param, noise_R_param, device, track_Q=1, **enkf_kwargs)
#     #   if optimizer is not None:
#     #     nll_suro_em.sum().backward()
#     #     optimizer.step()
    
#     if test_every_n_epochs is not None and epoch % test_every_n_epochs == 0:
#       while True:
#         try:
#           with torch.no_grad():
#             _, X_track_test, _, test_neg_log_likelihood = DAModel.EnKF(ode_func, obs_func, t_obs, y_obs_test, 500, init_m, init_C_param, model_Q_param, noise_R_param, device, **enkf_kwargs)  
#             if true_system == "L63": 
#               test_mse = torch.sqrt(utils.mse_loss_last_dim(X_track_test.mean(dim=-2)[31:], out_test[30:]))
#             elif true_system == "L96":
#               test_mse = torch.sqrt(utils.mse_loss(X_track_test.mean(dim=-2)[31:], out_test[30:]))
#             one_step_fs = metrics.forecast_skill(true_ode_func, ode_func, torch.tensor([step_size],device=device), u0_test, device, 'rk4', False)
#           break
#         except RuntimeError:
#           print("Warning! Filter divergence while testing.")

#     if n_epochs <= 100 or epoch % 10 == 0:
#       print(f"Epoch {epoch}: nll = {neg_log_likelihood.mean().cpu().item()}")
    

#     if print_every_n_epochs is not None and epoch % print_every_n_epochs == 0:
#       utils.plot_filter(batch_t_obs, first_out[:, 0, :], utils.shrink_batch_dim(X_track)[1:, 0, :, :], fig_num_limit=5)
#       with torch.no_grad():
#         if true_system == "L63":
#           true_coeff = torch.tensor([10., 8/3, 28.])
#           true_ode_func = NNModel.Lorenz63(true_coeff, 3)
#           t_obs_test = torch.arange(0.05, 10, 0.05, device=device)
#           rdn = random.randrange(batch_t_obs.shape[0])
#           out_test1, _, _, _ = DEModel.generate_data(ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           out_test_true1, _, _, _ = DEModel.generate_data(true_ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           utils.plot_dynamic(t_obs_test, out_test_true1, t_obs=None, y_obs=out_test1, fig_num_limit=3, text="test", contour_plot=False)
#         elif true_system == "L96":
#           true_F = 8.
#           true_ode_func = NNModel.Lorenz96(F=true_F, x_dim=40, device=device).to(device)
#           t_obs_test = torch.arange(0.05, 10, 0.05, device=device)
#           rdn = random.randrange(batch_t_obs.shape[0])
#           out_test1, _, _, _ = DEModel.generate_data(ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           out_test_true1, _, _, _ = DEModel.generate_data(true_ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           utils.plot_dynamic(t_obs_test, out_test_true1, t_obs=None, y_obs=out_test1, fig_num_limit=5, text="test", contour_plot=True)
#       if draw_stats:
#         fig, axes = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=False)
#         with torch.no_grad():
#           fs = metrics.forecast_skill(true_ode_func, ode_func, t_obs_test, u0_test, device, 'rk4', False)
#         axes[0].plot(t_obs_test.cpu().numpy(), fs.cpu().numpy())
#         axes[0].set_title('Forecast Skill')
#         spec = metrics.lyapunov_exponent(ode_func, step_size, step_size, 1000, u0_test[3], device, 'rk4')
#         spec_true = metrics.lyapunov_exponent(true_ode_func, step_size, step_size, 1000, u0_test[3], device, 'rk4')
#         axes[1].plot(spec, label='learned')
#         axes[1].plot(spec_true, label='Truth')
#         axes[1].set_title("Lyapunov spectrum")
#         axes[1].legend()
#         t_test = torch.arange(0, 120, 0.05)[1:].to(device)
#         with torch.no_grad():
#           f, Pxx_den=metrics.power_spectrum_density(ode_func, t_test, u0_test[0], device,'rk4',512)#, model_Q=0.01*torch.eye(x_dim, device=device))
#           f, Pxx_den_true=metrics.power_spectrum_density(true_ode_func, t_test, u0_test[0], device,'rk4',512)
#         axes[2].semilogy(f, Pxx_den, label='learned')
#         axes[2].semilogy(f, Pxx_den_true, label='Truth')
#         axes[2].set_title("Power Spectrum density")
#         axes[2].set_xlabel('frequency')
#         axes[2].set_ylabel('Density')
#         axes[2].legend()
#         plt.show()

#     if scheduler is not None:
#       scheduler.step()

#     if monitor is not None:
#       res = monitor(ode_func, model_Q_param, neg_log_likelihood, test_neg_log_likelihood, test_mse, one_step_fs, nll_suro_em, "em", device)  # (1, n_monitors)
#       monitor_res = torch.cat((monitor_res, res), dim=0) # (n_iter, n_monitors)
#       if n_epochs <= 100 or epoch % 10 == 0:
#         print(res)
#   return monitor_res



# def train_loop(strategy, ode_func, obs_func, t_obs, y_obs, out, out_intermediate, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, optimizer, n_epochs, batch_length, bs, 
#                                     save_location=None, scheduler=None, print_every_n_epochs=None, true_system="L63", save_every_n_epochs=1, test_every_n_epochs=None, 
#                                     tbptt=None, lr_decay=None, start_epoch=0, method="enkf", train_size=1, clip_norm=None, monitor=None, **enkf_kwargs):
#   '''
#   Args:
#     y_obs: batch of observation data # (n_obs, y_dim) or (n_obs, n_train+n_test, y_dim)
#     out: batch of true state # (n_obs, x_dim) or (n_obs, n_train+n_test, x_dim)
#   '''
#   step_size = enkf_kwargs['ode_options']['step_size']
#   # model.eval() if optimizer is None else model.train()
#   if len(y_obs.shape) == 2:
#     y_obs = y_obs.unsqueeze(-2)
#     out = out.unsqueeze(-2)
  
#   if test_every_n_epochs is not None:
#     y_obs_train, y_obs_test = train_test_split(y_obs, n_train=train_size) # (n_obs, n_train, y_dim) and (n_obs, n_test, y_dim)
#     out_train, out_test = train_test_split(out, n_train=train_size) # (n_obs, n_train, x_dim) and (n_obs, n_test, x_dim)
#   else:
#     y_obs_train = y_obs
#     out_train = out
#     test_neg_log_likelihood = torch.tensor(0.0, device=device)
#     test_mse = 0
#     one_step_fs = 0

#   monitor_res = torch.zeros(0, device=device)
#   for epoch in range(start_epoch, n_epochs):
#     batch_y_obs = y_obs_train.repeat(1,bs,1)  # (n_obs), (n_obs, n_train * bs, y_dim), (n_obs, n_train * bs, x_dim)
      


#   return

# def train_loop_direct(ode_func, obs_func, t_obs, y_obs, out, out_intermediate, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, optimizer, n_epochs,
#                                         scheduler=None, print_every_n_epochs=None, bs=None, true_system="L63",true_ode_func=None,  u0_test=None, draw_stats=False, test_every_n_epochs=None, save_every_n_epochs=1,  start_epoch=0, train_size=1, monitor=None, **enkf_kwargs):
#   step_size = enkf_kwargs['ode_options']['step_size']
#   if test_every_n_epochs is not None:
#     y_obs_train, y_obs_test = train_test_split(y_obs, n_train=train_size)
#     out_train, out_test = train_test_split(out, n_train=train_size)
#   else:
#     y_obs_train = y_obs.unsqueeze(-2)
#     out_train = out.unsqueeze(-2)
#     test_neg_log_likelihood = torch.tensor(0.0, device=device)
#     test_mse = 0

#   neg_log_likelihood = torch.tensor(0.0, device=device)
#   test_neg_log_likelihood = torch.tensor(0.0, device=device)
#   test_mse = torch.tensor(0.0, device=device)

#   n_obs = y_obs.shape[0]
#   y_dim = y_obs.shape[-1]
#   # H = torch.eye(y_dim)
#   # obs_func = NNModel.Linear(y_dim, y_dim, H=H).to(device)

#   batch_t_obs, batch_y_obs, batch_out, first_y_obs, first_out = t_obs, y_obs_train.repeat(1,1,1), out_train, y_obs_train, out_train

#   monitor_res = torch.zeros(0, device=device)
#   # X_track_m = X_track.mean(dim=-2, keepdim=True)
#   # optimizer.state=collections.defaultdict(dict)
#   for epoch in range(start_epoch, n_epochs):
#     permutation = torch.randperm(n_obs-1)
#     # permutation = torch.arange(n_obs)
#     for i in range(0, n_obs-1, bs):
#       optimizer.zero_grad() 
#       indices = permutation[i:i+bs]
#       nn_input, nn_label = batch_y_obs[indices], batch_y_obs[indices+1] # (bs, 1, x_dim)
#       nn_input_label = torch.stack((nn_input, nn_label)) # (2, bs, 1, x_dim)
#       nll_suro_em = get_em_loss(nn_input_label, ode_func, obs_func, torch.tensor([t_obs[0]]), None, None, None, model_Q_param, None, device, **enkf_kwargs)
#       nll_suro_em.mean().backward()
#       optimizer.step()
  
#     if test_every_n_epochs is not None and epoch % test_every_n_epochs == 0:
#       while True:
#         try:
#           with torch.no_grad():
#             # _, X_track_test, _, test_neg_log_likelihood = DAModel.EnKF(ode_func, obs_func, t_obs, y_obs_test, 500, init_m, init_C_param, model_Q_param, noise_R_param, device, **enkf_kwargs)  
#             # if true_system == "L63": 
#             #   test_mse = torch.sqrt(utils.mse_loss_last_dim(X_track_test.mean(dim=-2)[31:], out_test[30:]))
#             # elif true_system == "L96":
#             #   test_mse = torch.sqrt(utils.mse_loss(X_track_test.mean(dim=-2)[31:], out_test[30:]))
#             one_step_fs = metrics.forecast_skill(true_ode_func, ode_func, torch.tensor([step_size],device=device), u0_test, device, 'rk4', False)
#           break
#         except RuntimeError:
#           print("Warning! Filter divergence while testing.")

#     print(f"Epoch {epoch}: loss = {nll_suro_em.mean().cpu().item()}, test_nll = {test_neg_log_likelihood.mean().cpu().item()}")
    

#     if print_every_n_epochs is not None and epoch % print_every_n_epochs == 0:
#       # utils.plot_filter(batch_t_obs, first_out[:, 0, :], utils.shrink_batch_dim(X_track)[1:, 0, :, :], fig_num_limit=5)
#       with torch.no_grad():
#         if true_system == "L63":
#           true_coeff = torch.tensor([10., 8/3, 28.])
#           true_ode_func = NNModel.Lorenz63(true_coeff, 3)
#           t_obs_test = torch.arange(0.05, 10, 0.05, device=device)
#           rdn = random.randrange(batch_t_obs.shape[0])
#           out_test1, _, _, _ = DEModel.generate_data(ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           out_test_true1, _, _, _ = DEModel.generate_data(true_ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           utils.plot_dynamic(t_obs_test, out_test_true1, t_obs=None, y_obs=out_test1, fig_num_limit=3, text="test", contour_plot=False)
#         elif true_system == "L96":
#           true_F = 8.
#           true_ode_func = NNModel.Lorenz96(F=true_F, x_dim=40, device=device).to(device)
#           t_obs_test = torch.arange(0.05, 10, 0.05, device=device)
#           rdn = random.randrange(batch_t_obs.shape[0])
#           out_test1, _, _, _ = DEModel.generate_data(ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           out_test_true1, _, _, _ = DEModel.generate_data(true_ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           utils.plot_dynamic(t_obs_test, out_test_true1, t_obs=None, y_obs=out_test1, fig_num_limit=5, text="test", contour_plot=True)
#       if draw_stats:
#         fig, axes = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=False)
#         with torch.no_grad():
#           fs = metrics.forecast_skill(true_ode_func, ode_func, t_obs_test, u0_test, device, 'rk4', False)
#         axes[0].plot(t_obs_test.cpu().numpy(), fs.cpu().numpy())
#         axes[0].set_title('Forecast Skill')
#         spec = metrics.lyapunov_exponent(ode_func, step_size, step_size, 1000, u0_test[3], device, 'rk4')
#         spec_true = metrics.lyapunov_exponent(true_ode_func, step_size, step_size, 1000, u0_test[3], device, 'rk4')
#         axes[1].plot(spec, label='learned')
#         axes[1].plot(spec_true, label='Truth')
#         axes[1].set_title("Lyapunov spectrum")
#         axes[1].legend()
#         t_test = torch.arange(0, 120, 0.05)[1:].to(device)
#         with torch.no_grad():
#           f, Pxx_den=metrics.power_spectrum_density(ode_func, t_test, u0_test[0], device,'rk4',512)#, model_Q=0.01*torch.eye(x_dim, device=device))
#           f, Pxx_den_true=metrics.power_spectrum_density(true_ode_func, t_test, u0_test[0], device,'rk4',512)
#         axes[2].semilogy(f, Pxx_den, label='learned')
#         axes[2].semilogy(f, Pxx_den_true, label='Truth')
#         axes[2].set_title("Power Spectrum density")
#         axes[2].set_xlabel('frequency')
#         axes[2].set_ylabel('Density')
#         axes[2].legend()
#         plt.show()

#     if scheduler is not None:
#       scheduler.step()

#     if monitor is not None:
#       res = monitor(ode_func, model_Q_param, neg_log_likelihood, test_neg_log_likelihood, test_mse, one_step_fs, nll_suro_em, "direct", device)  # (1, n_monitors)
#       monitor_res = torch.cat((monitor_res, res), dim=0) # (n_iter, n_monitors)
#       print(res)

# def train_loop_particle(ode_func, obs_func, t_obs, y_obs, out, out_intermediate, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, optimizer, n_epochs, batch_length, bs, proposal, adaptive_resampling=False,
#                                     save_location=None, scheduler=None, print_every_n_epochs=None, true_system="L63",true_ode_func=None,  u0_test=None, draw_stats=False, save_every_n_epochs=1, test_every_n_epochs=None, tbptt=None, lr_decay=None, start_epoch=0,  train_size=1, clip_norm=None, monitor=None, t0=0, **enkf_kwargs):
#   """
#   Args: 
#     y_obs: batch of observation data # (n_obs, *bs, y_dim) Most likely (n_obs, n_train+n_test, y_dim)
#   """
#   likelihood_warmup = enkf_kwargs.get('likelihood_warmup', 0)
#   step_size = enkf_kwargs['ode_options']['step_size']
#   # model.eval() if optimizer is None else model.train()
#   if len(y_obs.shape) == 2:
#     y_obs = y_obs.unsqueeze(-2)
#     out = out.unsqueeze(-2)
  
#   if test_every_n_epochs is not None:
#     y_obs_train, y_obs_test = train_test_split(y_obs, n_train=train_size) # (n_obs, n_train, y_dim) and (n_obs, n_test, y_dim)
#     out_train, out_test = train_test_split(out, n_train=train_size) # (n_obs, n_train, x_dim) and (n_obs, n_test, x_dim)
#   else:
#     y_obs_train = y_obs
#     out_train = out
#     test_neg_log_likelihood = torch.tensor(0.0, device=device)
#     test_mse = 0
#     one_step_fs = 0
  
#   monitor_res = torch.zeros(0, device=device)
#   for epoch in range(start_epoch, n_epochs):
#     if tbptt is None:
#       # batch_t_obs, batch_y_obs, batch_out, first_y_obs, first_out = get_batch(t_obs, y_obs_train, out_train, batch_length, bs)  # (batch_length), (batch_length, *bs, n_draws, y_dim), (batch_length, *bs, n_draws, x_dim)
#       batch_t_obs, batch_y_obs, batch_out, first_y_obs, first_out = t_obs, y_obs_train.repeat(1,bs,1), out_train, y_obs_train, out_train

#       optimizer.zero_grad()
#       X, w, X_track, X_intermediate, w_track, w_intermediate, neg_log_likelihood = DAModel.BootstrapPF(ode_func, obs_func, batch_t_obs, batch_y_obs, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, proposal, adaptive_resampling, **enkf_kwargs)
#       loss = -neg_log_likelihood.mean()# + 1* torch.norm(ode_func.param, 1)
#       loss.backward()
#       if clip_norm is not None:
#         torch.nn.utils.clip_grad_norm_(ode_func.parameters(), clip_norm)
#         torch.nn.utils.clip_grad_norm_(model_Q_param, clip_norm)
#       optimizer.step()
#       if lr_decay is not None:
#         if epoch == 0:
#           initial_lr = [g['lr'] for g in optimizer.param_groups]
#         for i in range(len(optimizer.param_groups)):
#           optimizer.param_groups[i]['lr'] = initial_lr[i] / math.pow(epoch+1, lr_decay)
#     else:
#       batch_t_obs, batch_y_obs, batch_out, first_y_obs, first_out = t_obs, y_obs_train.repeat(1,bs,1), out_train, y_obs_train, out_train
#       with Timer("Filter"):
#         X, w, X_track, X_intermediate, w_track, w_intermediate, neg_log_likelihood = DAModel.BootstrapPF(ode_func, obs_func, batch_t_obs, batch_y_obs, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, proposal, adaptive_resampling, optimizer, tbptt, **enkf_kwargs)
#       if lr_decay is not None:
#         if epoch == 0:
#           initial_lr = [g['lr'] for g in optimizer.param_groups]
#         for i in range(len(optimizer.param_groups)):
#           optimizer.param_groups[i]['lr'] = initial_lr[i] / math.pow(epoch+1, lr_decay)



#     if test_every_n_epochs is not None and epoch % test_every_n_epochs == 0:
#       with torch.no_grad():
#         _,_,X_track_test, _, w_track_test, _, test_neg_log_likelihood = DAModel.BootstrapPF(ode_func, obs_func, t_obs, y_obs_test, 500, init_m, init_C_param, model_Q_param, noise_R_param, device,proposal, adaptive_resampling, **enkf_kwargs)  
#         if true_system == "L63": 
#           test_mse = torch.sqrt(utils.particle_mse_loss_last_dim(X_track_test[31:], out_test[30:], w_track_test[31:]))
#         elif true_system == "L96":
#           test_mse = torch.sqrt(utils.particle_mse_loss(X_track_test[31:], out_test[30:], w_track_test[31:]))
#         one_step_fs = metrics.forecast_skill(true_ode_func, ode_func, torch.tensor([step_size],device=device), u0_test, device, 'rk4', False)
    
#     if n_epochs <= 100 or epoch % 10 == 0:
#       print(f"Epoch {epoch}: nll = {neg_log_likelihood.mean().cpu().item()}, test_nll = {test_neg_log_likelihood.mean().cpu().item()}")
    



#     if print_every_n_epochs is not None and epoch % print_every_n_epochs == 0:
#       utils.plot_filter(batch_t_obs, first_out[:, 0, :], (w_track[1:, 0, :], utils.shrink_batch_dim(X_track)[1:, 0, :, :]), name="Particle", fig_num_limit=3)
#       with torch.no_grad():
#         if true_system == "L63":
#           true_coeff = torch.tensor([10., 8/3, 28.])
#           true_ode_func = NNModel.Lorenz63(true_coeff, 3)
#           t_obs_test = torch.arange(0.05, 10, 0.05, device=device)
#           rdn = random.randrange(batch_t_obs.shape[0])
#           out_test1, _, _, _ = DEModel.generate_data(ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           out_test_true1, _, _, _ = DEModel.generate_data(true_ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           utils.plot_dynamic(t_obs_test, out_test_true1, t_obs=None, y_obs=out_test1, fig_num_limit=3, text="test", contour_plot=False)
#         elif true_system == "L96":
#           true_F = 8.
#           true_ode_func = NNModel.Lorenz96(F=true_F, x_dim=40, device=device).to(device)
#           t_obs_test = torch.arange(0.05, 16, 0.05, device=device)
#           rdn = random.randrange(batch_t_obs.shape[0])
#           out_test1, _, _, _ = DEModel.generate_data(ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           out_test_true1, _, _, _ = DEModel.generate_data(true_ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           utils.plot_dynamic(t_obs_test, out_test_true1, t_obs=None, y_obs=out_test1, fig_num_limit=5, text="test", contour_plot=True)
#       if draw_stats:
#         fig, axes = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=False)
#         with torch.no_grad():
#           fs = metrics.forecast_skill(true_ode_func, ode_func, t_obs_test, u0_test, device, 'rk4', False)
#         axes[0].plot(t_obs_test.cpu().numpy(), fs.cpu().numpy())
#         axes[0].set_title('Forecast Skill')
#         spec = metrics.lyapunov_exponent(ode_func, step_size, step_size, 1000, u0_test[3], device, 'rk4')
#         spec_true = metrics.lyapunov_exponent(true_ode_func, step_size, step_size, 1000, u0_test[3], device, 'rk4')
#         axes[1].plot(spec, label='learned')
#         axes[1].plot(spec_true, label='Truth')
#         axes[1].set_title("Lyapunov spectrum")
#         axes[1].legend()
#         t_test = torch.arange(0, 120, 0.05)[1:].to(device)
#         with torch.no_grad():
#           f, Pxx_den=metrics.power_spectrum_density(ode_func, t_test, u0_test[0], device,'rk4',512)#, model_Q=0.01*torch.eye(x_dim, device=device))
#           f, Pxx_den_true=metrics.power_spectrum_density(true_ode_func, t_test, u0_test[0], device,'rk4',512)
#         axes[2].semilogy(f, Pxx_den, label='learned')
#         axes[2].semilogy(f, Pxx_den_true, label='Truth')
#         axes[2].set_title("Power Spectrum density")
#         axes[2].set_xlabel('frequency')
#         axes[2].set_ylabel('Density')
#         axes[2].legend()
#         plt.show()

#     if scheduler is not None:
#       scheduler.step()

#     if monitor is not None:
#       res = monitor(ode_func, model_Q_param, noise_R_param, neg_log_likelihood, test_neg_log_likelihood, test_mse, one_step_fs, None, "particle", device)  # (1, n_monitors)
#       monitor_res = torch.cat((monitor_res, res), dim=0) # (n_iter, n_monitors)
#       if n_epochs <= 100 or epoch % 10 == 0:
#         print(res)

#     # if epoch % save_every_n_epochs == 0:
#       # torch.save({'model_state_dict': net_state, 'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict() }, save_location)
#   return monitor_res


# def train_loop_diff(ode_func, obs_func, t_obs, y_obs, out, out_intermediate, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, optimizer, n_epochs, batch_length, bs, 
#                                     save_location=None, scheduler=None, print_every_n_epochs=None, true_system="L63", true_ode_func=None, u0_test=None, draw_stats=False, save_every_n_epochs=1, test_every_n_epochs=None, 
#                                     tbptt=None, lr_decay=None, start_epoch=0, method="enkf", train_size=1, clip_norm=None, monitor=None, t0=0, **enkf_kwargs):
#   """
#   Args: 
#     y_obs: batch of observation data # (n_obs, *bs, y_dim) Most likely (n_obs, n_train+n_test, y_dim)
#   """
#   likelihood_warmup = enkf_kwargs.get('likelihood_warmup', 0)
#   step_size = enkf_kwargs['ode_options']['step_size']
#   # model.eval() if optimizer is None else model.train()
#   if len(y_obs.shape) == 2:
#     y_obs = y_obs.unsqueeze(-2)
#     out = out.unsqueeze(-2)
  
#   if test_every_n_epochs is not None:
#     y_obs_train, y_obs_test = train_test_split(y_obs, n_train=train_size) # (n_obs, n_train, y_dim) and (n_obs, n_test, y_dim)
#     out_train, out_test = train_test_split(out, n_train=train_size) # (n_obs, n_train, x_dim) and (n_obs, n_test, x_dim)
#   else:
#     y_obs_train = y_obs
#     out_train = out
#     test_neg_log_likelihood = torch.tensor(0.0, device=device)
#     test_mse = 0
#     one_step_fs = 0
  
#   monitor_res = torch.zeros(0, device=device)
#   for epoch in range(start_epoch, n_epochs):





#     if tbptt is None:
#       # batch_t_obs, batch_y_obs, batch_out, first_y_obs, first_out = get_batch(t_obs, y_obs_train, out_train, batch_length, bs)  # (batch_length), (batch_length, *bs, n_draws, y_dim), (batch_length, *bs, n_draws, x_dim)
#       batch_t_obs, batch_y_obs, batch_out, first_y_obs, first_out = t_obs, y_obs_train.repeat(1,bs,1), out_train, y_obs_train, out_train
#       optimizer.zero_grad()
#       if method == "enkf":
#         X, X_track, X_intermediate, neg_log_likelihood = DAModel.EnKF(ode_func, obs_func, batch_t_obs, batch_y_obs, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, **enkf_kwargs)
#       elif method == "kf":
#         m_track, C_track, neg_log_likelihood = DAModel.KF(ode_func, obs_func, batch_t_obs, batch_y_obs, init_m, init_C_param, model_Q_param, noise_R_param, device, **enkf_kwargs)
#       loss = -neg_log_likelihood.mean()# + 1* torch.norm(ode_func.param, 1)
#       loss.backward()
#       if clip_norm is not None:
#         torch.nn.utils.clip_grad_norm_(ode_func.parameters(), clip_norm)
#         torch.nn.utils.clip_grad_norm_(model_Q_param.parameters(), clip_norm)
#       optimizer.step()
#       if lr_decay is not None:
#         if epoch == 0:
#           initial_lr = [g['lr'] for g in optimizer.param_groups]
#         for i in range(len(optimizer.param_groups)):
#           optimizer.param_groups[i]['lr'] = initial_lr[i] / math.pow(epoch+1, lr_decay)
#     else:
#       batch_t_obs, batch_y_obs, batch_out, first_y_obs, first_out = t_obs, y_obs_train.repeat(1,bs,1), out_train, y_obs_train, out_train
#       if method == "enkf":
#         with Timer("Filter"):
#           X, X_track, X_intermediate, neg_log_likelihood = DAModel.EnKF(ode_func, obs_func, batch_t_obs, batch_y_obs, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, optimizer, tbptt, **enkf_kwargs)
#       if lr_decay is not None:
#         if epoch == 0:
#           initial_lr = [g['lr'] for g in optimizer.param_groups]
#         for i in range(len(optimizer.param_groups)):
#           optimizer.param_groups[i]['lr'] = initial_lr[i] / math.pow(epoch+1, lr_decay)
    
#     if test_every_n_epochs is not None and epoch % test_every_n_epochs == 0:
#       while True:
#         try:
#           with torch.no_grad():
#             _, X_track_test, _, test_neg_log_likelihood = DAModel.EnKF(ode_func, obs_func, t_obs, y_obs_test, 500, init_m, init_C_param, model_Q_param, noise_R_param, device, **enkf_kwargs) 
#             if true_system == "L63": 
#               test_mse = torch.sqrt(utils.mse_loss_last_dim(X_track_test.mean(dim=-2)[31:], out_test[30:]))
#             elif true_system == "L96":
#               test_mse = torch.sqrt(utils.mse_loss(X_track_test.mean(dim=-2)[31:], out_test[30:]))
#             one_step_fs = metrics.forecast_skill(true_ode_func, ode_func, torch.tensor([step_size],device=device), u0_test, device, 'rk4', False)
#           break
#         except RuntimeError:
#           print("Warning! Filter divergence while testing.")
      
    
#     if n_epochs <= 100 or epoch % 10 == 0:
#       print(f"Epoch {epoch}: nll = {neg_log_likelihood.mean().cpu().item()}, test_nll = {test_neg_log_likelihood.mean().cpu().item()}")
    



#     if print_every_n_epochs is not None and epoch % print_every_n_epochs == 0:
#       utils.plot_filter(batch_t_obs, first_out[:, 0, :], utils.shrink_batch_dim(X_track)[1:, 0, :, :], fig_num_limit=3)
#       with torch.no_grad():
#         if true_system == "L63":
#           true_coeff = torch.tensor([10., 8/3, 28.])
#           true_ode_func = NNModel.Lorenz63(true_coeff, 3)
#           t_obs_test = torch.arange(0.05, 10, 0.05, device=device)
#           rdn = random.randrange(batch_t_obs.shape[0])
#           out_test1, _, _, _ = DEModel.generate_data(ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           out_test_true1, _, _, _ = DEModel.generate_data(true_ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           utils.plot_dynamic(t_obs_test, out_test_true1, t_obs=None, y_obs=out_test1, fig_num_limit=3, text="test", contour_plot=False)
#         elif true_system == "L96":
#           true_F = 8.
#           true_ode_func = NNModel.Lorenz96(F=true_F, x_dim=40, device=device).to(device)
#           t_obs_test = torch.arange(0.05, 10, 0.05, device=device)
#           rdn = random.randrange(batch_t_obs.shape[0])
#           out_test1, _, _, _ = DEModel.generate_data(ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           out_test_true1, _, _, _ = DEModel.generate_data(true_ode_func, None, t_obs_test, first_out[rdn,0], None, None, device, ode_method='rk4', ode_options=dict(step_size=0.05),adjoint=False, save_intermediate_step=False)
#           utils.plot_dynamic(t_obs_test, out_test_true1, t_obs=None, y_obs=out_test1, fig_num_limit=5, text="test", contour_plot=True)
#       if draw_stats:
#         fig, axes = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=False)
#         with torch.no_grad():
#           fs = metrics.forecast_skill(true_ode_func, ode_func, t_obs_test, u0_test, device, 'rk4', False)
#         axes[0].plot(t_obs_test.cpu().numpy(), fs.cpu().numpy())
#         axes[0].set_title('Forecast Skill')
#         spec = metrics.lyapunov_exponent(ode_func, step_size, step_size, 1000, u0_test[3], device, 'rk4')
#         spec_true = metrics.lyapunov_exponent(true_ode_func, step_size, step_size, 1000, u0_test[3], device, 'rk4')
#         axes[1].plot(spec, label='learned')
#         axes[1].plot(spec_true, label='Truth')
#         axes[1].set_title("Lyapunov spectrum")
#         axes[1].legend()
#         t_test = torch.arange(0, 120, 0.05)[1:].to(device)
#         with torch.no_grad():
#           f, Pxx_den=metrics.power_spectrum_density(ode_func, t_test, u0_test[0], device,'rk4',512)#, model_Q=0.01*torch.eye(x_dim, device=device))
#           f, Pxx_den_true=metrics.power_spectrum_density(true_ode_func, t_test, u0_test[0], device,'rk4',512)
#         axes[2].semilogy(f, Pxx_den, label='learned')
#         axes[2].semilogy(f, Pxx_den_true, label='Truth')
#         axes[2].set_title("Power Spectrum density")
#         axes[2].set_xlabel('frequency')
#         axes[2].set_ylabel('Density')
#         axes[2].legend()
#         plt.show()

#     if scheduler is not None:
#       scheduler.step()

#     if monitor is not None:
#       res = monitor(ode_func, model_Q_param, noise_R_param, neg_log_likelihood, test_neg_log_likelihood, test_mse, one_step_fs, None, "diff", device)  # (1, n_monitors)
#       monitor_res = torch.cat((monitor_res, res), dim=0) # (n_iter, n_monitors)
#       if n_epochs <= 100 or epoch % 10 == 0:
#         print(res)

#     # if epoch % save_every_n_epochs == 0:
#       # torch.save({'model_state_dict': net_state, 'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict() }, save_location)
#   return monitor_res