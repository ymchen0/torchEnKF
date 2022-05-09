from tqdm import tqdm
import os
import sys
sys.path.append(os.getcwd()) # Fix Python path

from torchEnKF import da_methods, nn_templates, noise
from examples import generate_data, utils

import random
import torch
import numpy as np

from torchdiffeq import odeint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"
print(f"device: {device}")
# torch.backends.cudnn.benchmark = True

seed = 44
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

######### Define reference model #########
xx_dim = 36
xy_dim = 10
x_dim = xx_dim * (xy_dim + 1)
true_param = torch.tensor([10., 1., 10., 10.])
true_ode_func = nn_templates.Lorenz96_FS(true_param, device, xx_dim, xy_dim).to(device)

######### Warmup: Draw x0 from the L96 limit cycle. Can be ignored for problems with a smaller scale. #########
print("Warming up...")
train_size = 10
with torch.no_grad():
    xx_cov = 25 * torch.eye(xx_dim)
    xy_cov = 0.25 * torch.eye(xx_dim * xy_dim)
    cov = torch.block_diag(xx_cov, xy_cov).to(device)
    x0_warmup = torch.distributions.MultivariateNormal(torch.zeros(x_dim, device=device), covariance_matrix=cov).sample().to(device) # <-
    t_warmup = 20 * torch.arange(0., train_size + 1).to(device)
    x0 = odeint(true_ode_func, x0_warmup, t_warmup, method='rk4', options=dict(step_size=0.005))[1:]

######### Generate training data from the reference model #########
t0 = 0.
t_obs_step = 0.1
n_obs = 100
t_obs = t_obs_step * torch.arange(1, n_obs+1).to(device)
model_Q_true = None  # No noise in dynamics

def true_obs_func(X):  # define the observation model
    # (*bs, x_dim) -> (*bs, y_dim)
    bs = X.shape[:-1]
    to_cat = []
    to_cat.append(X[..., :xx_dim])
    Y_bar = X[..., xx_dim:].reshape(*bs, xx_dim, xy_dim).mean(dim=-1)
    to_cat.append(Y_bar)
    to_cat.append(X[..., :xx_dim]**2)
    to_cat.append(X[..., :xx_dim] * Y_bar)
    to_cat.append(((X[..., xx_dim:].reshape(*bs, xx_dim, xy_dim))**2).mean(dim=-1).view(*bs, xx_dim))
    return torch.cat(to_cat, dim=-1)
y_dim = true_obs_func(x0).shape[-1]
xo_cov = 0.1 * torch.eye(xx_dim)
yo_cov = 0.1 * torch.eye(xx_dim)
xxo_cov = 0.1 * torch.eye(xx_dim)
xyo_cov = 0.1 * torch.eye(xx_dim)
yyo_cov = 0.1 * torch.eye(xx_dim)
obs_cov = torch.block_diag(xo_cov, yo_cov, xxo_cov, xyo_cov, yyo_cov).to(device)
noise_R_true = noise.AddGaussian(y_dim, obs_cov, param_type='full').to(device)


with torch.no_grad():
    x_truth, y_obs = generate_data.generate(true_ode_func, true_obs_func, t_obs, x0, model_Q_true, noise_R_true, device=device,
                                            ode_method='rk4', ode_options=dict(step_size=0.0025), tqdm=tqdm)  # Shape: (n_obs, train_size, y_dim)

######### Parameter estimation #########
N_ensem = 100
init_m = torch.zeros(x_dim, device=device)
init_C_param = noise.AddGaussian(x_dim, cov, 'full').to(device)
init_coeff = torch.tensor([8., 0., 2., 2.],device=device)
init_Q = 0.2 * torch.ones(x_dim)
learned_ode_func = nn_templates.Lorenz96_FS(init_coeff, device, xx_dim, xy_dim).to(device)
learned_model_Q = noise.AddGaussian(x_dim, init_Q, 'diag').to(device)
optimizer = torch.optim.Adam([{'params':learned_ode_func.parameters(), 'lr':1e-1},
                            {'params':learned_model_Q.parameters(), 'lr':1e-1}])
lambda1 = lambda2 = lambda epoch: (epoch-49)**(-0.5) if epoch >=50 else 1
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
L = 1 # subsequence length in AD-EnKF-T
warm_up = 0
for epoch in tqdm(range(100), desc="Training", leave=False):
    train_log_likelihood = torch.zeros(train_size, device=device)
    t_start = t0
    X = init_C_param(init_m.expand(train_size, N_ensem, x_dim))

    # Warm-up phase. Time interval at the beginning that the gradients will not be recorded. But the filtered states will. (This is not presented in paper)
    with torch.no_grad():
        X, res, log_likelihood = da_methods.EnKF(learned_ode_func, true_obs_func, t_obs[:warm_up], y_obs[:warm_up], N_ensem, init_m, init_C_param, learned_model_Q, noise_R_true, device,
                                                     save_filter_step={}, t0=t_start, init_X=X, ode_options=dict(step_size=0.0025), adjoint_options=dict(step_size=0.01), linear_obs=False)
        train_log_likelihood += log_likelihood
    t_start = t_obs[warm_up - 1] if warm_up >= 1 else t0

    for start in range(warm_up, n_obs, L):
        optimizer.zero_grad()
        end = min(start + L, n_obs)
        X, res, log_likelihood = da_methods.EnKF(learned_ode_func,true_obs_func, t_obs[start:end], y_obs[start:end], N_ensem, init_m, init_C_param, learned_model_Q, noise_R_true,device,
                                             save_filter_step={}, t0=t_start, init_X=X, ode_options=dict(step_size=0.0025), adjoint_options=dict(step_size=0.01), linear_obs=False)
        t_start = t_obs[end - 1]
        (-log_likelihood).mean().backward()
        train_log_likelihood += log_likelihood.detach().clone()
        optimizer.step()
    scheduler.step()

    if epoch % 1 == 0:
        tqdm.write(f"Epoch {epoch}, Training log-likelihood: {train_log_likelihood.mean().item()}")
        tqdm.write(f"Learned coefficients: {learned_ode_func.param.data.cpu().numpy()}")
        tqdm.write(f"Learned q: {torch.sqrt(torch.trace(learned_model_Q.full())/x_dim).item()}")