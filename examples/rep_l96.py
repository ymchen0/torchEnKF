import utils
from core_training import train_loop_diff, train_loop_em_new
from tqdm import tqdm
# from tqdm.auto import tqdm

from models import DEModel, DAModel, NNModel, Noise

from utils import Timer
import copy
import math
import random
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device="cpu"
print(f"device: {device}")


seed = 40
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

x_dim = 40
true_F = 8.
true_coeff = torch.tensor([8., 0., 0., -1, 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 1., 0.], device=device)
true_ode_func = NNModel.Lorenz96_dict_param(true_coeff, device, x_dim).to(device)

init_m = torch.zeros(x_dim, device=device)
init_C_param = Noise.AddGaussian(x_dim, 50*torch.eye(x_dim), 'full').to(device)

train_and_test = 6
train_size = 4

with torch.no_grad():
    u0_warmup = torch.distributions.MultivariateNormal(torch.zeros(x_dim), covariance_matrix=25 * torch.eye(x_dim)).sample().to(device)  # <-
    t_warmup = torch.cat((torch.tensor([0.]), 120 * torch.arange(1, train_and_test + 1))).to(device)
    out_warmup = odeint(true_ode_func, u0_warmup, t_warmup, method='rk4', options=dict(step_size=0.05))
    u0 = out_warmup[1:]  # (*bs, x_dim)

t0 = 0.
t_obs_step = 0.05
t_obs_end = 15.
t_obs = torch.arange(t0, t_obs_end + t_obs_step, t_obs_step)[1:].to(device)

model_Q_true = None

indices = [i for i in range(x_dim)]  # if nm == 0 else ( [3 * i for i in range((x_dim-1)//3+1)] + [3 * i + 1 for i in range((x_dim+1)//3)] )
y_dim = len(indices)
time_varying_obs = False
if time_varying_obs:
    true_obs_func = []
    for j in range(t_obs.shape[0]):
        indices = np.random.choice(x_dim, y_dim, False)
        H_true = torch.eye(x_dim)[indices]
        true_obs_func.append(NNModel.Linear(x_dim, y_dim, H=H_true).to(device))
else:
    H_true = torch.eye(x_dim)[indices]
    true_obs_func = NNModel.Linear(x_dim, y_dim, H=H_true).to(device)

noise_R_true = Noise.AddGaussian(y_dim, torch.tensor(1.), param_type='scalar').to(device)

with torch.no_grad():
    out, y_obs, out_intermediate, t_intermediate = DEModel.generate_data(true_ode_func, true_obs_func, t_obs, u0, model_Q_true, noise_R_true, device=device, ode_method='rk4',
                                                                         ode_options=dict(step_size=0.01),adjoint=False, save_intermediate_step=True, t0=0., time_varying_obs=time_varying_obs)

