import math
import torch
from torchEnKF import nn_templates
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
from tqdm import tqdm
import os
import sys
sys.path.append(os.getcwd()) # Fix Python path


def generate(ode_func, obs_func, t_obs, x0, model_Q_param, noise_R_param, device,
             ode_method='rk4', ode_options=None, t0=0., time_varying_obs=False, tqdm=None):
    """
    Generate state and observation data from a given state space model.

    Key args:
        ode_func (torch.nn.Module): Vector field f(t,x).
        obs_func (torch.nn.Module): Observation model h(x).
                If time varying_obs==True, can take a list of torch.nn.Module's
        t_obs (tensor): 1D-Tensor of shape (n_obs,). Time points where observations are available
        x0 (tensor): Tensor of shape (*bs, x_dim). Initial positions x0.
                '*bs' can be arbitrary batch dimension (or empty).
        model_Q_param (Noise.AddGaussian): model error covariance
        noise_R_param (Noise.AddGaussian): observation error covariance

    Optional args:
        ode_method: Numerical scheme for forward equation. We use 'euler' or 'rk4'. Other solvers are available. See https://github.com/rtqichen/torchdiffeq
        ode_options: Set it to dict(step_size=...) for fixed step solvers. Adaptive solvers are also available - see the link above.

    Returns:
        x_truth (tensor): Tensor of shape (n_obs, *bs, x_dim). States of the reference model.
        y_obs (tensor): Tensor of shape (n_obs, *bs, y_dim). Observations of the reference model.
    """

    x_dim = x0.shape[-1]
    n_obs = t_obs.shape[0]
    bs = x0.shape[:-1]

    x_truth = torch.empty(n_obs, *bs, x_dim, dtype=x0.dtype, device=device)  # (n_obs, *bs, x_dim)
    X = x0

    y_obs = None
    if obs_func is not None:
        obs_func_0 = obs_func[0] if time_varying_obs else obs_func
        y_dim = obs_func_0(x0).shape[-1]
        y_obs = torch.empty(n_obs, *bs, y_dim, dtype=x0.dtype, device=device)


    t_cur = t0

    pbar = tqdm(range(n_obs), desc="Generating data", leave=True) if tqdm is not None else range(n_obs)
    for j in pbar:
        _, X = odeint(ode_func, X, torch.tensor([t_cur, t_obs[j]], device=device), method=ode_method, options=ode_options)
        t_cur = t_obs[j]

        if model_Q_param is not None:
            X = model_Q_param(X)

        x_truth[j] = X

        if obs_func is not None:
            obs_func_j = obs_func[j] if time_varying_obs else obs_func
            HX = obs_func_j(X)
            y_obs[j] = noise_R_param(HX)

    return x_truth, y_obs

