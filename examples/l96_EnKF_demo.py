# from core_training import train_loop_diff, train_loop_em_new
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


seed = 40
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


######### Define reference model #########
x_dim = 40
true_F = 8.
true_coeff = torch.tensor([8., 0., 0., -1, 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 1., 0.], device=device)
true_ode_func = nn_templates.Lorenz96_dict_param(true_coeff, device, x_dim).to(device)


######### Warmup: Draw an initial point x0 from the L96 limit cycle. Can be ignored for problems with a smaller scale. #########
with torch.no_grad():
    x0_warmup = torch.distributions.MultivariateNormal(torch.zeros(x_dim), covariance_matrix=50 * torch.eye(x_dim)).sample().to(device)
    t_warmup = torch.tensor([0., 120.]).to(device)
    x0 = odeint(true_ode_func, x0_warmup, t_warmup, method='rk4', options=dict(step_size=0.05))[1]  # Shape (x_dim,)


######### Generate data from the reference model #########
t0 = 0.
t_obs_step = 0.05
n_obs = 300
t_obs = t_obs_step * torch.arange(1, n_obs+1).to(device)
model_Q_true = None  # No noise in dynamics
indices = [i for i in range(x_dim)]
y_dim = len(indices)
H_true = torch.eye(x_dim)[indices]
true_obs_func = nn_templates.Linear(x_dim, y_dim, H=H_true).to(device)  # Full observation
noise_R_true = noise.AddGaussian(y_dim, torch.tensor(1.), param_type='scalar').to(device)  # Gaussian observation noise with std 1
with torch.no_grad():
    x_truth, y_obs = generate_data.generate(true_ode_func, true_obs_func, t_obs, x0, model_Q_true, noise_R_true, device=device,
                                            ode_method='rk4', ode_options=dict(step_size=0.01), tqdm=tqdm)


########## Run EnKF with the reference model, compute log-likelihood and gradient #########
N_ensem = 50
init_m = torch.zeros(x_dim, device=device)
init_C_param = noise.AddGaussian(x_dim, 50 * torch.eye(x_dim), 'full').to(device)
X, res, log_likelihood = da_methods.EnKF(true_ode_func,true_obs_func, t_obs, y_obs, N_ensem, init_m, init_C_param, model_Q_true, noise_R_true,device,
                                              save_filter_step={'mean'}, localization_radius=5, tqdm=tqdm)
print(f"log-likelihood estimate: {log_likelihood}")
burn_in = n_obs // 5
print(f"Filter accuracy (RMSE): {torch.sqrt(utils.mse_loss(res['mean'][burn_in:], x_truth[burn_in:]))}")
print("Computing gradient...")
log_likelihood.backward()
print(f"Gradient: {true_ode_func.coeff.grad}")

### The log-likelihood estimates and gradient estimates can be used for various purposes!