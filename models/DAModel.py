import torch
import math
import utils
from models import NNModel
from models import Noise
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint

from tqdm.auto import tqdm
import time

class Timer(object):
  def __init__(self, name=None):
    if name is None:
      self.name = 'foo'
    else:
      self.name = name

  def __enter__(self):
    self.tstart = time.time()

  def __exit__(self, type, value, traceback):
    if self.name is None:
      print('[%s]' % self.name,)
    print(f"{self.name}, Elapsed: {time.time() - self.tstart}s")

def construct_Gaspari_Cohn(loc_radius, x_dim, device):
  def G(z):
    if z >= 0 and z < 1:
      return 1. - 5./3*z**2 + 5./8*z**3 + 1./2*z**4 - 1./4*z**5
    elif z >= 1 and z < 2:
      return 4. - 5.*z + 5./3*z**2 + 5./8*z**3 - 1./2*z**4 + 1./12*z**5 - 2./(3*z)
    else:
      return 0
  taper = torch.zeros(x_dim, x_dim, device=device)
  for i in range(x_dim):
    for j in range(x_dim):
      dist = min(abs(i-j), x_dim - abs(i-j))
      taper[i, j] = G(dist/loc_radius)
  return taper

def EnKF(ode_func, obs_func, t_obs, y_obs, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, init_X=None, Cf_track=None, detach_every=None,
                        ode_method='rk4', ode_options=None, adjoint=False, adjoint_method=None, adjoint_options=None, save_filter_step=True, save_intermediate_step=False, 
                        smooth_lag=0, t0=0., var_inflation=None, localization_radius=None, compute_likelihood=False, likelihood_warmup=0, linear_obs=True, time_varying_obs=False, save_first=False, simulation_type=1, tqdm=False, **ode_kwargs):
  """
  Args:
    t_obs: 1D tensor. Time points where observations are available. # (n_obs,)
    y_obs: observed values at t_eval. # (n_obs, *bs, y_dim)
    obs_func: nn.Module if time_varying_obs=False
          list(nn.Module) if time_varying_obs=True. # (n_obs, )
    N_ensem: Ensemble size.
    init_m, init_C: Initial mean and covariance. # (x_dim,)  (x_dim, x_dim)
    model_Q_param: (Additive) model error covariance. Can be regarded as additive covariance inflation # (x_dim, x_dim)
    model_Q_type: If "scalar", then \sigma = exp(model_Q_param)
            If "diag", then \sigma_1,...\sigma_n = exp(model_Q_param)
            If "tril", then cov = LL^T, where L is lower_triangular, and diag(L) is positive, e.g., exp(...). Here model_Q_param=L
    noise_R_param: Noise covariance. # (obs_dim, obs_dim)
    noise_R_type: Same as above
    ode_options: dict(step_size=...) for fixed step solvers. If None, use default time step (between observations)
    intermediate_step: Whether to return intermediate time-steps for filter (mainly for plotting, and when bs is None). If True, expand t_obs
    smooth_lag: How much time to 'look back' when smoothing

  Returns:
    X_track: Filtered states at time t_obs. # (n_obs, *bs, N_ensem, x_dim)
    X_intermediate: If save_intermediate_step is true, returns trajectory of filtered states from t0 to t_obs[-1], with step_size given by ode_options. # (n_intermediate+1, *bs, N_ensem, x_dim)
    neg_log_likelihood: Negative log likelihood # (*bs)
  """
  if save_intermediate_step and 'step_size' not in ode_options:
    raise ValueError('Specify step size first to save intermediate steps.')

  ode_integrator = odeint_adjoint if adjoint else odeint

  x_dim = init_m.shape[0]
  y_dim = y_obs.shape[-1]
  n_obs = y_obs.shape[0]
  bs = y_obs.shape[1:-1]  # torch.Size()

  neg_log_likelihood = torch.zeros(bs, device=device) if compute_likelihood else None  # (*bs),  tensor(0.) if no batch dimension

  if localization_radius is not None:
    taper = construct_Gaspari_Cohn(localization_radius, x_dim, device)

  if init_X is not None:
    X = init_X.detach()
  else:
    X = init_C_param(init_m.expand(*bs, N_ensem, x_dim))

  X_track = None
  X_intermediate = None
  if save_filter_step:
    X_track = torch.empty(n_obs+1, *bs, N_ensem, x_dim, dtype=init_m.dtype, device=device)
    X_track[0] = X.detach().clone()
  if save_intermediate_step:
    step_size = ode_options['step_size']
    n_intermediate = round(((t_obs[-1] - t0) / step_size).item()) if n_obs > 0 else 0
    X_intermediate = torch.empty(n_intermediate+1, *bs, N_ensem, x_dim, dtype=init_m.dtype, device=device)
    X_intermediate[0] = X.detach().clone()
    n_cur = 0

  t_cur = t0
  
  pbar = tqdm(range(n_obs), leave=False) if tqdm else range(n_obs)
  for j in pbar:
    ################ Forecast step ##################
    X_prev = X
    if not save_intermediate_step:
      if adjoint:
        _, X = ode_integrator(ode_func, X, torch.tensor([t_cur, t_obs[j]], device=device), method=ode_method, options=ode_options, adjoint_method=adjoint_method, adjoint_options=adjoint_options, **ode_kwargs)
      else:
        _, X = ode_integrator(ode_func, X, torch.tensor([t_cur, t_obs[j]], device=device), method=ode_method, options=ode_options, **ode_kwargs)
    else:
      n_intermediate_j = round(((t_obs[j] - t_cur) / step_size).item())
      if adjoint:
        X_intermediate_j = ode_integrator(ode_func, X, torch.linspace(t_cur, t_obs[j], n_intermediate_j+1, device=device), method=ode_method, options=ode_options, adjoint_method=adjoint_method, adjoint_options=adjoint_options)
      else:
        X_intermediate_j = ode_integrator(ode_func, X, torch.linspace(t_cur, t_obs[j], n_intermediate_j+1, device=device), method=ode_method, options=ode_options)
      X_intermediate[n_cur+1:n_cur+n_intermediate_j] = X_intermediate_j[1:-1].detach().clone()
      X = X_intermediate_j[-1]
      n_cur += n_intermediate_j
    t_cur = t_obs[j]

    # (Additive) covariance inflation
    if model_Q_param is not None:
      X = model_Q_param(X, X_prev)

    X_m = X.mean(dim=-2).unsqueeze(-2)  # (*bs, 1, x_dim)
    X_ct = X - X_m

    # (Multiplicative) covariance inflation
    if var_inflation is not None:
      X = var_inflation * (X - X_m) + X_m

    ################ Analysis step ##################
    obs_func_j = obs_func[j] if time_varying_obs else obs_func
    y_obs_j = y_obs[j].unsqueeze(-2)  # (*bs, 1, y_dim)
    # Noise perturbation of observed data (key in stochastic EnKF)
    obs_perturb = noise_R_param(y_obs_j.expand(*bs, N_ensem, y_dim))
    noise_R = noise_R_param.full()
    
    if simulation_type == 0:
      if Cf_track is None:
        C_uu = 1/(N_ensem-1) * X_ct.transpose(-1, -2) @ X_ct  # (*bs, x_dim, x_dim)
      else:
        C_uu = Cf_track[j]
      
      if localization_radius is not None:
        C_uu = taper * C_uu
      if linear_obs:
        H = obs_func_j.H # (y_dim, x_dim)
      else:
        H_vec = torch.autograd.functional.jacobian(obs_func_j, X_m.view(-1, x_dim), create_graph=False, strict=False, vectorize=True) # (bs_mul, y_dim, bs_mul, x_dim)
        H = H_vec.diagonal(dim1=0, dim2=2).permute(2, 0, 1).view(*bs, y_dim, x_dim) # (*bs, y_dim, x_dim)   see https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571
      HX = X @ H.transpose(-1, -2) # (*bs, N_ensem, y_dim)
      HX_m = X_m @ H.transpose(-1, -2) # (*bs, 1, y_dim)
      HC = H @ C_uu # (*bs, y_dim, x_dim)
      HCH_T = HC @ H.transpose(-1, -2) # (*bs, y_dim, y_dim)
      HCH_TR_chol = torch.linalg.cholesky(HCH_T + noise_R) # (*bs, y_dim, y_dim), lower-tril
      if compute_likelihood: #and j >= likelihood_warmup:
        d = torch.distributions.MultivariateNormal(HX_m.squeeze(-2), scale_tril=HCH_TR_chol) # (*bs, y_dim) and (*bs, y_dim, y_dim)
        neg_log_likelihood += d.log_prob(y_obs_j.squeeze(-2)) # (*bs)
        # d = torch.distributions.MultivariateNormal(torch.zeros(y_dim, device=device), scale_tril=noise_R_chol)
        # neg_log_likelihood += -torch.log(torch.exp(d.log_prob(y_obs_j - HX)).mean(dim=-1)) # (*bs, N_ensem) -> (*bs)
      pre = (obs_perturb - HX) @ torch.cholesky_inverse(HCH_TR_chol) # (*bs, N_ensem, y_dim)
      X = X + pre @ HC # (*bs, N_ensem, x_dim)

      if detach_every is not None and (j+1) % detach_every == 0:
        X = X.detach()


      if save_filter_step and smooth_lag > 0:
        with torch.no_grad():
          t_start = torch.max(t_cur - smooth_lag, torch.tensor(t0))
          j_start = torch.argmax((t_obs>=t_start).type(torch.uint8)) # e.g., t_obs[5] = 0.5, t_obs[4]=0.4, smooth_lag=0.1 --> j_start = 4
          XS = X_track[j_start:j] # (L, *bs, N_ensem, x_dim)
          XS_m = XS.mean(dim=-2, keepdim=True) # (L, *bs, 1, x_dim)
          CSH_T = 1/(N_ensem-1) * (XS - XS_m).transpose(-1, -2) @ (HX - HX_m) # (L, *bs, x_dim, y_dim)
          XS = XS + pre @ CSH_T.transpose(-1, -2)
          # XS = XS + (obs_perturb - HX) @ torch.cholesky_solve(CSH_T.transpose(-1, -2), HCH_TR_chol) # (L, *bs, N_ensem, x_dim) This might be more stable!
          X_track[j_start:j] = XS
          if save_intermediate_step:
            n_start_im = round(((t_start - t0)/step_size).item())
            XS = X_intermediate[n_start_im:n_cur]
            XS_m = XS.mean(dim=-2, keepdim=True) # (L, *bs, 1, x_dim)
            CSH_T = 1/(N_ensem-1) * (XS - XS_m).transpose(-1, -2) @ (HX - HX_m) # (L, *bs, x_dim, y_dim)
            XS = XS + pre @ CSH_T.transpose(-1, -2) # (L, *bs, N_ensem, x_dim)
            X_intermediate[n_start_im:n_cur] = XS


    elif simulation_type == 1:
      HX = obs_func_j(X)   # (*bs, N_ensem, y_dim)
      HX_m = HX.mean(dim=-2).unsqueeze(-2)  # (*bs, 1, y_dim)
      HX_ct = HX - HX_m
      C_uw = 1/(N_ensem-1) * X_ct.transpose(-1, -2) @ HX_ct  # (*bs, x_dim, y_dim)
      C_ww = 1/(N_ensem-1) * HX_ct.transpose(-1, -2) @ HX_ct  # (*bs, y_dim, y_dim)
      C_ww_R_chol = torch.linalg.cholesky(C_ww + noise_R) # (*bs, y_dim, y_dim), lower-tril
      pre = (obs_perturb - HX) @ torch.cholesky_inverse(C_ww_R_chol) # (*bs, N_ensem, y_dim)
      if compute_likelihood and j >= likelihood_warmup:
        d = torch.distributions.MultivariateNormal(HX_m.squeeze(-2), scale_tril=C_ww_R_chol)  # (*bs, y_dim) and (*bs, y_dim, y_dim)
        neg_log_likelihood += d.log_prob(y_obs_j.squeeze(-2)) # (*bs)
      X = X + pre @ C_uw.transpose(-1, -2)  # (*bs, N_ensem, x_dim)
      ########### TODO: Smoothing ##############
    if save_filter_step:
      X_track[j+1] = X.detach()
    if save_intermediate_step:
      X_intermediate[n_cur] = X.detach().clone()
  if not save_first:
    X_track = X_track[1:]
  return X, X_track, X_intermediate, neg_log_likelihood


def KF(ode_func, obs_func, t_obs, y_obs, init_m, init_C_param, model_Q_param, noise_R_param, device, compute_likelihood=False, save_first=False, time_varying_obs=False, **unused_kwargs):
  """
  Args:
    t_obs: 1D tensor. Time points where observations are available. # (n_obs,)
    y_obs: observed values at t_eval. # (n_obs, *bs, y_dim)
    obs_func: nn.Module if time_varying_obs=False
          list(nn.Module) if time_varying_obs=True. # (n_obs, )
    init_m, init_C: Initial mean and covariance. # (x_dim,)  (x_dim, x_dim)
    model_Q_param: (Additive) model error covariance. Can be regarded as additive covariance inflation # (x_dim, x_dim)
    model_Q_type: If "scalar", then \sigma = exp(model_Q_param)
            If "diag", then \sigma_1,...\sigma_n = exp(model_Q_param)
            If "tril", then cov = LL^T, where L is lower_triangular, and diag(L) is positive, e.g., exp(...). Here model_Q_param=L
    noise_R_param: Noise covariance. # (obs_dim, obs_dim)
    noise_R_type: Same as above

  Returns:
    mu_track, C_track: Filtered states at time t_obs. # (n_obs, *bs, x_dim), (n_obs, *bs, x_dim, x_dim)
    neg_log_likelihood: Negative log likelihood # (*bs)
  """
  # if not isinstance(ode_func, NNModel.Linear_ODE) and not isinstance(obs_func, NNModel.Linear):
  #   raise ValueError('Please ensure that both dynamic and observation models are linear.')
  
  # A = ode_func.A() # (x_dim, x_dim)

  x_dim = ode_func.A().shape[0]
  y_dim = y_obs.shape[-1]
  n_obs = y_obs.shape[0]
  bs = y_obs.shape[1:-1]

  neg_log_likelihood = torch.zeros(bs, device=device) if compute_likelihood else None  # (*bs),  tensor(0.) if no batch dimension

  m_track = torch.empty(n_obs+1, *bs, x_dim, dtype=init_m.dtype, device=device) # (n_obs, *bs, x_dim)
  C_track = torch.empty(n_obs+1, *bs, x_dim, x_dim, dtype=init_m.dtype, device=device) # (n_obs, *bs, x_dim, x_dim)
  Cf_track = torch.empty(n_obs, *bs, x_dim, x_dim, dtype=init_m.dtype, device=device) # (n_obs, *bs, x_dim, x_dim)

  m = init_m.repeat(*bs, 1)  # (*bs, x_dim)
  C = init_C_param.full()
  m_track[0] = m.detach().clone()
  C_track[0] = C.detach().clone()

  for j in range(n_obs):
    ################ Forecast step ##################
    m = m @ ode_func.A().t() # (*bs, x_dim)
    C = ode_func.A() @ C @ ode_func.A().t() # (*bs, x_dim, x_dim)
    if model_Q_param is not None:
      C += model_Q_param.full()
    Cf_track[j] = C.detach().clone()

    ################ Analysis step ##################
    obs_func_j = obs_func[j] if time_varying_obs else obs_func
    y_obs_j = y_obs[j].unsqueeze(-2)  # (*bs, 1, y_dim)
    H = obs_func_j.H # (y_dim, x_dim)
    HX = (m @ H.t()).unsqueeze(-2) # (*bs, 1, y_dim)
    HC = H @ C # (*bs, y_dim, x_dim)
    HCH_T = HC @ H.t() # (*bs, y_dim, y_dim)
    noise_R = noise_R_param.full()
    HCH_TR_chol = torch.linalg.cholesky(HCH_T + noise_R) # (*bs, y_dim, y_dim)
    K_T = torch.cholesky_inverse(HCH_TR_chol) @ HC # (*bs, y_dim, x_dim)
    if compute_likelihood:
      d = torch.distributions.MultivariateNormal(HX.squeeze(-2), scale_tril=HCH_TR_chol)
      neg_log_likelihood += d.log_prob(y_obs_j.squeeze(-2)) # (*bs)
    m = m + ((y_obs_j - HX) @ K_T).squeeze(-2) # (*bs, x_dim)
    C = C @ (torch.eye(x_dim, device=device) - H.t() @ K_T) # (*bs, x_dim, x_dim)
    ########### TODO: Smoothing ##############
    m_track[j+1] = m.detach().clone()
    C_track[j+1] = C.detach().clone()

  if not save_first:
    m_track = m_track[1:]
    C_track = C_track[1:]
  return m_track, C_track, Cf_track, neg_log_likelihood

def BootstrapPF(ode_func, obs_func, t_obs, y_obs, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, proposal='bootstrap', proposal_param=None, init_w=None, init_X=None, detach_every=None,
                        ode_method='rk4', ode_options=None, adjoint=False, adjoint_method=None, adjoint_options=None, save_intermediate_step=False, 
                        smooth_lag=0, t0=0., var_inflation=None, localization_radius=None, compute_likelihood=False, likelihood_warmup=0, linear_obs=True, time_varying_obs=False, save_first=False, adaptive_resampling=False, **ode_kwargs):
  """
  Args:
    t_obs: 1D tensor. Time points where observations are available. # (n_obs,)
    y_obs: observed values at t_eval. # (n_obs, *bs, y_dim)
    obs_func: nn.Module if time_varying_obs=False
          list(nn.Module) if time_varying_obs=True. # (n_obs, )
    init_m, init_C: Initial mean and covariance. # (x_dim,)  (x_dim, x_dim)
    model_Q_param: (Additive) model error covariance. Can be regarded as additive covariance inflation # (x_dim, x_dim)
    model_Q_type: If "scalar", then \sigma = exp(model_Q_param)
            If "diag", then \sigma_1,...\sigma_n = exp(model_Q_param)
            If "tril", then cov = LL^T, where L is lower_triangular, and diag(L) is positive, e.g., exp(...). Here model_Q_param=L
    noise_R_param: Noise covariance. # (obs_dim, obs_dim)
    noise_R_type: Same as above

  Returns:
    w_track, X_track: Filtered states at time t_obs. # (n_obs, *bs, N_ensem), (n_obs, *bs, N_ensem, x_dim)
    neg_log_likelihood: Negative log likelihood # (*bs)
  """
  if save_intermediate_step and 'step_size' not in ode_options:
    raise ValueError('Specify step size first to save intermediate steps.')

  ode_integrator = odeint_adjoint if adjoint else odeint

  x_dim = init_m.shape[0]
  y_dim = y_obs.shape[-1]
  n_obs = y_obs.shape[0]
  bs = y_obs.shape[1:-1]

  neg_log_likelihood = torch.zeros(bs, device=device) if compute_likelihood else None  # (*bs),  tensor(0.) if no batch dimension

  w_track = torch.empty(n_obs+1, *bs, N_ensem, device=device) # (n_obs+1, *bs, N_ensem)
  X_track = torch.empty(n_obs+1, *bs, N_ensem, x_dim, dtype=init_m.dtype, device=device) # (n_obs+1, *bs, N_ensem, x_dim)

  w_intermediate = None
  X_intermediate = None


  if init_w is not None:
    w = init_w.detach()
    X = init_X.detach()
  else:
    w = 1. / N_ensem * torch.ones(*bs, N_ensem, device=device) # (*bs, N_ensem)
    X = init_C_param(init_m.expand(*bs, N_ensem, x_dim))
  
  w_track[0] = w.detach().clone()
  X_track[0] = X.detach().clone()
  if save_intermediate_step:
    step_size = ode_options['step_size']
    n_intermediate = round(((t_obs[-1] - t0) / step_size).item()) if n_obs > 0 else 0
    w_intermediate = torch.empty(n_intermediate+1, *bs, N_ensem, device=device)
    X_intermediate = torch.empty(n_intermediate+1, *bs, N_ensem, x_dim, dtype=init_m.dtype, device=device)
    w_intermediate[0] = w.detach().clone()
    X_intermediate[0] = X.detach().clone()
    n_cur = 0

  t_cur = t0


  
  for j in range(n_obs):
    ################ Forecast step ##################
    X_prev = X
    if not save_intermediate_step:
      if adjoint:
        _, X = ode_integrator(ode_func, X, torch.tensor([t_cur, t_obs[j]], device=device), method=ode_method, options=ode_options, adjoint_method=adjoint_method, adjoint_options=adjoint_options, **ode_kwargs)
      else:
        _, X = ode_integrator(ode_func, X, torch.tensor([t_cur, t_obs[j]], device=device), method=ode_method, options=ode_options, **ode_kwargs)
    else:
      n_intermediate_j = round(((t_obs[j] - t_cur) / step_size).item())
      if adjoint:
        X_intermediate_j = ode_integrator(ode_func, X, torch.linspace(t_cur, t_obs[j], n_intermediate_j+1, device=device), method=ode_method, options=ode_options, adjoint_method=adjoint_method, adjoint_options=adjoint_options)
      else:
        X_intermediate_j = ode_integrator(ode_func, X, torch.linspace(t_cur, t_obs[j], n_intermediate_j+1, device=device), method=ode_method, options=ode_options)
      X_intermediate[n_cur+1:n_cur+n_intermediate_j] = X_intermediate_j[1:-1].detach().clone()
      w_intermediate[n_cur+1:n_cur+n_intermediate_j] = w.detach().clone().expand(n_intermediate_j-1, *bs, N_ensem)
      X = X_intermediate_j[-1]
      n_cur += n_intermediate_j
    t_cur = t_obs[j]

    if model_Q_param is not None:
      model_Q_chol = model_Q_param.chol(X_prev) # (x_dim, x_dim) or (*bs, N_ensem, x_dim, x_dim)
      model_Q = model_Q_param.full(X_prev) # (x_dim, x_dim) or (*bs, N_ensem, x_dim, x_dim)
    if noise_R_param is not None:
      noise_R_chol = noise_R_param.chol(X_prev)
      noise_R = noise_R_param.full(X_prev)

    if proposal == 'bootstrap':
      if model_Q_param is not None:
        X = model_Q_param(X, X_prev)

      ################ Analysis step ##################
      obs_func_j = obs_func[j] if time_varying_obs else obs_func
      y_obs_j = y_obs[j].unsqueeze(-2)  # (*bs, 1, y_dim)
      # Noise perturbation of observed data (key in stochastic EnKF)
      HX = obs_func_j(X) # (*bs, N_ensem, y_dim)
      obs_d = torch.distributions.MultivariateNormal(torch.zeros(y_dim, device=device), scale_tril=noise_R_chol)
      logits = obs_d.log_prob(y_obs_j - HX) + torch.log(w) # (*bs, N_ensem)
      
    elif proposal == 'optimal':
      obs_func_j = obs_func[j] if time_varying_obs else obs_func
      y_obs_j = y_obs[j].unsqueeze(-2)  # (*bs, 1, y_dim)
      if linear_obs:
        H = obs_func_j.H # (y_dim, x_dim)
      else:
        H_vec = torch.autograd.functional.jacobian(obs_func_j, X_m.view(-1, x_dim), create_graph=False, strict=False, vectorize=True) # (bs_mul, y_dim, bs_mul, x_dim)
        H = H_vec.diagonal(dim1=0, dim2=2).permute(2, 0, 1).view(*bs, y_dim, x_dim) # (*bs, y_dim, x_dim)   see https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571
      HX = X @ H.transpose(-1, -2) # (*bs, N_ensem, y_dim)
      HQ = H @ model_Q # (y_dim, x_dim) or (*bs, N_ensem, y_dim, x_dim)
      HQH_T = HQ @ H.transpose(-1, -2) # (y_dim, y_dim)
      HQH_TR_chol = torch.linalg.cholesky(HQH_T + noise_R) # (y_dim, y_dim) or (*bs, N_ensem, y_dim, y_dim)
      K_T = torch.cholesky_inverse(HQH_TR_chol) @ HQ # (y_dim, x_dim) or (*bs, N_ensem, y_dim, x_dim)
      obs_d = torch.distributions.MultivariateNormal(torch.zeros(y_dim, device=device), scale_tril=HQH_TR_chol)
      logits = obs_d.log_prob(y_obs_j - HX) + torch.log(w) # (*bs, N_ensem)

      X = X + ((y_obs_j - HX).unsqueeze(-2) @ K_T).squeeze(-2) # (*bs, N_ensem, x_dim)
      C = model_Q @ (torch.eye(x_dim, device=device) - H.transpose(-1, -2) @ K_T) # (x_dim, x_dim) or (*bs, N_ensem, x_dim, x_dim)
      C_chol = torch.linalg.cholesky(C)
      if len(C.shape) == 2:
        X = X + torch.distributions.MultivariateNormal(torch.zeros(x_dim, device=device), scale_tril=C_chol).sample((*bs, N_ensem)) # (*bs, N_ensem, x_dim)
      else:
        X = X + torch.distributions.MultivariateNormal(torch.zeros(x_dim, device=device), scale_tril=C_chol).sample() # (*bs, N_ensem, x_dim)

    elif proposal == 'variational':
      obs_func_j = obs_func[j] if time_varying_obs else obs_func
      y_obs_j = y_obs[j].unsqueeze(-2)  # (*bs, 1, y_dim)
      
      X_new, log_rphi = proposal_param(X, y_obs_j)
      log_dyn = torch.distributions.MultivariateNormal(torch.zeros(x_dim, device=device), scale_tril=model_Q_chol).log_prob(X_new-X) # (*bs, N_ensem)
      HX_new = obs_func_j(X_new)
      log_obs = torch.distributions.MultivariateNormal(torch.zeros(y_dim, device=device), scale_tril=noise_R_chol).log_prob(y_obs_j-HX_new) # (*bs, N_ensem)
      logits = log_dyn + log_obs - log_rphi + torch.log(w)
      X = X_new




    w = torch.nn.functional.softmax(logits, dim=-1) # (*bs, N_ensem)
    if compute_likelihood:
      neg_log_likelihood += torch.logsumexp(logits, dim=-1) # (*bs)      

    X_track[j+1] = X.detach().clone()
    w_track[j+1] = w.detach().clone()
    # w_track[j+1] = 1. / N_ensem * torch.ones(*bs, N_ensem) # (*bs, N_ensem)
    if save_intermediate_step:
      w_intermediate[n_cur] = w.detach().clone()
      X_intermediate[n_cur] = X.detach().clone()

    ##### Resampling #####
    ess = utils.ess(w)
    if not adaptive_resampling or (ess<N_ensem/2).sum() > 0:
      rs_d = torch.distributions.Categorical(logits=logits)
      rs_idx = rs_d.sample((N_ensem,)).view(N_ensem,-1).transpose(0,1).view(*bs,N_ensem) # (N_ensem, *bs) --> (*bs, N_ensem)
      rs_idx = rs_idx.unsqueeze(-1).expand(*bs, N_ensem, x_dim) # (*bs, N_ensem, x_dim)
      X = X.gather(-2, rs_idx)
      # X = X.detach()
      w = 1. / N_ensem * torch.ones(*bs, N_ensem, device=device) # (*bs, N_ensem)

    if detach_every is not None and (j+1) % detach_every == 0:
      X = X.detach()
      w = w.detach()
      
      
  if not save_first:
    X_track = X_track[1:]
    w_track = w_track[1:]
  # if tbptt is not None:
  #   neg_log_likelihood = neg_log_likelihood_out
  return X, w, X_track, X_intermediate, w_track, w_intermediate, neg_log_likelihood















