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
train_size = 8
test_size = 4
with torch.no_grad():
    x0_warmup = torch.distributions.MultivariateNormal(torch.zeros(x_dim), covariance_matrix=25 * torch.eye(x_dim)).sample().to(device)
    t_warmup = 60 * torch.arange(0., train_size + test_size + 1).to(device)
    x0_train_and_test = odeint(true_ode_func, x0_warmup, t_warmup, method='rk4', options=dict(step_size=0.05))[1:]
    x0_train = x0_train_and_test[:train_size] # Shape: (train_size, x_dim)
    x0_test = x0_train_and_test[train_size:] # Shape: (test_size, x_dim)

######### For computing forecast RMSE only: Draw P=500 points from the L96 limit cycle to computer forecast RMSE. ########
with torch.no_grad():
    x0_warmup = torch.distributions.MultivariateNormal(torch.zeros(x_dim), covariance_matrix=25 * torch.eye(x_dim)).sample((50,)).to(device)
    t_warmup = torch.cat((torch.tensor([0.]), torch.arange(80.,280.,20.))).to(device)
    x0_fc = odeint(true_ode_func, x0_warmup, t_warmup, method='rk4', options=dict(step_size=0.05))[1:].reshape(-1, x_dim) # Shape: (P, x_dim)


######### Generate training data from the reference model #########
t0 = 0.
t_obs_step = 0.05
n_obs = 1200
t_obs = t_obs_step * torch.arange(1, n_obs+1).to(device)
model_Q_true = None  # No noise in dynamics
indices = [i for i in range(x_dim)]
y_dim = len(indices)
H_true = torch.eye(x_dim)[indices]
true_obs_func = nn_templates.Linear(x_dim, y_dim, H=H_true).to(device)  # Observe every coordinates
noise_R_true = noise.AddGaussian(y_dim, torch.tensor(1.), param_type='scalar').to(device)  # Gaussian perturbation with std 1
with torch.no_grad():
    x_truth_train, y_obs_train = generate_data.generate(true_ode_func, true_obs_func, t_obs, x0_train, model_Q_true, noise_R_true, device=device,
                                            ode_method='rk4', ode_options=dict(step_size=0.01), tqdm=tqdm)  # Shape: (n_obs, train_size, y_dim)
    x_truth_test, y_obs_test = generate_data.generate(true_ode_func, true_obs_func, t_obs, x0_test, model_Q_true, noise_R_true, device=device,
                                            ode_method='rk4', ode_options=dict(step_size=0.01),tqdm=tqdm)  # Shape: (n_obs, test_size, y_dim)

######## Define NN correction model #############
pert_0, pert_1, pert_2 = 1, 0.1, 0.01
pert_diag = torch.cat( (pert_0 * torch.ones(1), pert_1 * torch.ones(5), pert_2 * torch.ones(12)) ).to(device)
pert_cov = torch.diag(pert_diag)
pert = torch.distributions.MultivariateNormal(torch.zeros(18, device=device), covariance_matrix=pert_cov).sample()
init_coeff = true_coeff + pert
learned_ode_func = nn_templates.Lorenz96_correction(init_coeff, x_dim).to(device)

######### NN training #########
N_ensem = 50
init_m = torch.zeros(x_dim, device=device)
init_C_param = noise.AddGaussian(x_dim, 25 * torch.eye(x_dim), 'full').to(device)
init_Q = 2 * torch.ones(x_dim)
learned_model_Q = noise.AddGaussian(x_dim, init_Q, 'diag').to(device)
optimizer = torch.optim.Adam([{'params':learned_ode_func.parameters(), 'lr':1e-3},
                            {'params':learned_model_Q.parameters(), 'lr':1e-1}])
lambda1 = lambda2 = lambda epoch: (epoch-9)**(-0.75) if epoch >=10 else 1
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
L = 20 # subsequence length in AD-EnKF-T
monitor = []
for epoch in tqdm(range(50), desc="Training", leave=False):
    train_log_likelihood = torch.zeros(train_size, device=device)
    train_state_est_loss = torch.tensor(0., device=device)
    t_start = t0
    X = init_C_param(init_m.expand(train_size, N_ensem, x_dim))

    # Training phase
    for start in range(0, n_obs, L):
        optimizer.zero_grad()
        end = min(start + L, n_obs)
        X, res, log_likelihood = da_methods.EnKF(learned_ode_func,true_obs_func, t_obs[start:end], y_obs_train[start:end], N_ensem, init_m, init_C_param, learned_model_Q, noise_R_true,device,
                                             save_filter_step={'mean'}, t0=t_start, init_X=X, ode_options=dict(step_size=0.01), adjoint_options=dict(step_size=0.05), localization_radius=5, tqdm=None)
        t_start = t_obs[end - 1]
        (-log_likelihood).mean().backward()
        train_log_likelihood += log_likelihood.detach().clone()
        train_state_est_loss += utils.mse_loss(res['mean'], x_truth_train[start:end]) * (end-start)
        optimizer.step()
    scheduler.step()

    # Testing
    with torch.no_grad():
        filter_rmse = torch.sqrt(train_state_est_loss / n_obs)
        _, _, test_log_likelihood = da_methods.EnKF(learned_ode_func, true_obs_func, t_obs, y_obs_test, N_ensem, init_m, init_C_param, learned_model_Q, noise_R_true, device,
                                                     save_filter_step={}, ode_options=dict(step_size=0.01), adjoint_options=dict(step_size=0.05), localization_radius=5, tqdm=None)
        true_fc, _ = generate_data.generate(true_ode_func, None, torch.tensor([t_obs_step], device=device), x0_fc, None, None, device, ode_options=dict(step_size=0.01))
        fc, _ = generate_data.generate(learned_ode_func, None, torch.tensor([t_obs_step], device=device), x0_fc, None, None, device, ode_options=dict(step_size=0.01))
        forecast_rmse = torch.sqrt(utils.mse_loss(true_fc, fc))

        curr_output = [forecast_rmse.item(), filter_rmse.item(), test_log_likelihood.mean().item()]
        monitor.append(curr_output)


    # Printing
    if epoch % 1 == 0:
        tqdm.write(f"Epoch {epoch}, Training log-likelihood: {train_log_likelihood.mean().item()}")
        tqdm.write(f"Epoch {epoch}, Test log-likelihood: {test_log_likelihood.mean().item()}")
        tqdm.write(f"Filter RMSE: {filter_rmse.item()}")
        tqdm.write(f"Forecast RMSE: {forecast_rmse.item()}")


# Reproducing Figure 7, EnKF results
monitor = np.asarray(monitor)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].plot(monitor[:,0])
axes[1].plot(monitor[:,1])
axes[2].plot(monitor[:,2])
axes[0].set_ylabel("Forecast RMSE")
axes[1].set_ylabel("Filter RMSE")
axes[2].set_ylabel("Test log likelihood")
plt.show()