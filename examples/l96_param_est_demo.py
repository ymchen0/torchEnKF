from tqdm import tqdm
import os
import sys
sys.path.append(os.getcwd()) # Fix Python path

from torchEnKF import da_methods, nn_templates, noise
from examples import generate_data, utils

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"
print(f"device: {device}")


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

######### Define reference model #########
x_dim = 40
true_F = 8.
true_coeff = torch.tensor([8., 0., 0., -1, 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 1., 0.], device=device)
true_ode_func = nn_templates.Lorenz96_dict_param(true_coeff, device, x_dim).to(device)


######### Warmup: Draw x0 from the L96 limit cycle. Can be ignored for problems with a smaller scale. #########
train_size = 4
with torch.no_grad():
    x0_warmup = torch.distributions.MultivariateNormal(torch.zeros(x_dim), covariance_matrix=25 * torch.eye(x_dim)).sample().to(device)  # <-
    t_warmup = 120 * torch.arange(0., train_size + 1).to(device)
    x0 = odeint(true_ode_func, x0_warmup, t_warmup, method='rk4', options=dict(step_size=0.05))[1:] # Shape: (train_size, x_dim)


######### Generate training data from the reference model #########
t0 = 0.
t_obs_step = 0.05
n_obs = 150
t_obs = t_obs_step * torch.arange(1, n_obs+1).to(device)
model_Q_true = None  # No noise in dynamics
indices = [i for i in range(x_dim)]
y_dim = len(indices)
H_true = torch.eye(x_dim)[indices]
true_obs_func = nn_templates.Linear(x_dim, y_dim, H=H_true).to(device)  # Observe every coordinates
noise_R_true = noise.AddGaussian(y_dim, torch.tensor(1.), param_type='scalar').to(device)  # Gaussian perturbation with std 1
with torch.no_grad():
    x_truth, y_obs = generate_data.generate(true_ode_func, true_obs_func, t_obs, x0, model_Q_true, noise_R_true, device=device,
                                            ode_method='rk4', ode_options=dict(step_size=0.01), tqdm=tqdm)  # Shape: (n_obs, train_size, y_dim)


######### Parameter estimation #########
N_ensem = 50
init_m = torch.zeros(x_dim, device=device)
init_C_param = noise.AddGaussian(x_dim, 25 * torch.eye(x_dim), 'full').to(device)
init_coeff = torch.zeros(18)
init_Q = 2 * torch.ones(x_dim)
learned_ode_func = nn_templates.Lorenz96_dict_param(init_coeff, device, x_dim).to(device)
learned_model_Q = noise.AddGaussian(x_dim, init_Q, 'diag').to(device)
optimizer = torch.optim.Adam([{'params':learned_ode_func.parameters(), 'lr':1e-1},
                            {'params':learned_model_Q.parameters(), 'lr':1e-1}])
lambda1 = lambda2 = lambda epoch: (epoch-9)**(-0.5) if epoch >=10 else 1
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
L = 20 # subsequence length in AD-EnKF-T
monitor = []
for epoch in tqdm(range(150), desc="Training", leave=False):
    train_log_likelihood = torch.zeros(train_size, device=device)
    t_start = t0
    X = init_C_param(init_m.expand(train_size, N_ensem, x_dim))

    # Training phase
    for start in range(0, n_obs, L):
        optimizer.zero_grad()
        end = min(start + L, n_obs)
        X, res, log_likelihood = da_methods.EnKF(learned_ode_func,true_obs_func, t_obs[start:end], y_obs[start:end], N_ensem, init_m, init_C_param, learned_model_Q, noise_R_true,device,
                                             save_filter_step={}, t0=t_start, init_X=X, ode_options=dict(step_size=0.01), adjoint_options=dict(step_size=0.05), localization_radius=5, tqdm=None)
        t_start = t_obs[end - 1]
        (-log_likelihood).mean().backward()
        train_log_likelihood += log_likelihood.detach().clone()
        optimizer.step()
    scheduler.step()

    # Printing
    if epoch % 5 == 0:
        tqdm.write(f"Epoch {epoch}, Training log-likelihood: {train_log_likelihood.mean().item()}")
        tqdm.write(f"Learned coefficients: {learned_ode_func.coeff.data.cpu().numpy()}")
    with torch.no_grad():
        q_scale = torch.sqrt(torch.trace(learned_model_Q.full())/x_dim)
        curr_output = learned_ode_func.coeff.tolist() + [q_scale.item()] + [train_log_likelihood.mean().item()]
        monitor.append(curr_output)

# Reproducing Figure 6, EnKF results
monitor = np.asarray(monitor)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i in range(18):
    if i in {0,3,11,16}:
        axes[0,0].plot(monitor[:,i])
    else:
        axes[0,1].plot(monitor[:,i])
axes[1,0].plot(monitor[:,-2])
axes[1,1].plot(monitor[:,-1])
axes[0,0].set_ylabel("Coefficients, non-zero entries")
axes[0,1].set_ylabel("Coefficients, zero entries")
axes[1,0].set_ylabel("Error level")
axes[1,1].set_ylabel("Training log-likelihood")
plt.show()