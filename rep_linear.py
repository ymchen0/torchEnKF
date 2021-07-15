import utils
from core_training import train_loop_diff, train_loop_em_new
from viz_loss import loss_grad_vis_1d, loss_grad_vis_2d
# from tqdm import tqdm
from tqdm.auto import tqdm

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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device="cpu"
print(f"device: {device}")


# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)


# N_ensems = np.rint(np.power(1.2, np.arange(18,50))).astype(int)
N_ensems = 25 * 2 ** (np.arange(0, 14-2))
n_epochs = N_ensems.shape[0]

n_runs=30
# n_epochs = x_axis.shape[0]

n_metaruns = 1
x_dims = [20]
monitor_EnKFs = []
monitor_KFs = []
monitor_PFs = []

for nm in range(n_metaruns):
  x_dim = x_dims[nm]
  # if nm == 0:
  #   x_dim = x_dims[ne]
  # true_a = torch.tensor(0.42)
  # true_ode_func = NNModel.Linear_ODE_single_var(x_dim, true_a).to(device)
  # true_a = torch.tensor([0.3, 0.4,0.1,0.15,0.05])
  true_a = torch.tensor([0.3, 0.6,0.1])
  a_dim = true_a.shape[0]
  true_ode_func = NNModel.Linear_ODE_diag(x_dim, true_a).to(device)

  u0 = 2*torch.ones(x_dim, device=device)
  t0 = 0.
  t_obs_step = 1.
  t_obs_end = 10.
  t_obs = torch.arange(t0, t_obs_end+t_obs_step, t_obs_step)[1:].to(device)

  # q = 0.5
  # model_Q_true = Noise.AddGaussian(x_dim, torch.sqrt(torch.tensor(q)), "scalar")
  q = 0.5
  model_Q_true = Noise.AddGaussian(x_dim, torch.sqrt(torch.tensor(q)), "scalar", q_shape=torch.tensor(1.))

  init_m = torch.zeros(x_dim, device=device)
  init_C_param = Noise.AddGaussian(x_dim, 4*torch.eye(x_dim), "full")

  indices = [i for i in range(x_dim)]
  # indices = [0]
  y_dim = len(indices)
  H_true = torch.eye(x_dim)[indices]
  true_obs_func = NNModel.Linear(x_dim, y_dim, H_true).to(device)

  noise_R_true = Noise.AddGaussian(y_dim, torch.sqrt(torch.tensor(0.5)), "scalar")

  seed = 42
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  with torch.no_grad():
    out, y_obs, out_intermediate, t_intermediate = DEModel.generate_data(true_ode_func, true_obs_func, t_obs, u0, model_Q_true, noise_R_true,
                                                  device=device, ode_method='euler', ode_options=dict(step_size=t_obs_step), adjoint=False, save_intermediate_step=True, t0=t0, time_varying_obs=False)

  monitor_EnKF = torch.zeros(n_runs, n_epochs, 3+a_dim)
  monitor_PF = torch.zeros(n_runs, n_epochs, 3+a_dim)
  monitor_KF = torch.zeros(1, n_epochs, 3+a_dim)

  for ne in range(n_epochs):
    print(ne,end='')

    N_ensem = N_ensems[ne]
    # test_a = torch.tensor([0.3, 0.4,0.1,0.15,0.05])
    test_a = torch.tensor([0.3, 0.6, 0.1])
    test_ode_func = NNModel.Linear_ODE_diag(x_dim, test_a).to(device)
    q = 0.5
    test_model_Q = Noise.AddGaussian(x_dim, torch.sqrt(torch.tensor(q)), "scalar", q_shape=torch.tensor(1.))

    enkf_kwargs = dict(ode_method='euler', ode_options=dict(step_size=t_obs_step), adjoint=False, save_intermediate_step=True, smooth_lag=0, t0=t0,
          var_inflation=None, localization_radius=None, compute_likelihood=True, likelihood_warmup=0,
          linear_obs=True, time_varying_obs=False, save_first=True, simulation_type=0)

    for nr in range(n_runs):
      X, X_track, X_intermediate, neg_log_likelihood = DAModel.EnKF(test_ode_func, true_obs_func, t_obs, y_obs.unsqueeze(-2).repeat(1,1,1), N_ensem=N_ensem, init_m=init_m, init_C_param=init_C_param, model_Q_param=test_model_Q, noise_R_param=noise_R_true, device=device, detach_every=None, Cf_track=None, **enkf_kwargs)
      neg_log_likelihood.mean().backward()
      monitor_EnKF[nr,ne,0] = neg_log_likelihood.detach()
      monitor_EnKF[nr,ne,1:1+a_dim] = test_ode_func.a.grad.detach()
      monitor_EnKF[nr,ne,1+a_dim] = test_model_Q.post_grad().item()
      monitor_EnKF[nr,ne,2+a_dim] = test_model_Q.q_shape.grad.item()
      test_ode_func.a.grad.zero_()
      test_model_Q.q.grad.zero_()
      test_model_Q.q_shape.grad.zero_()

    m_track, C_track, Cf_track, neg_log_likelihood = DAModel.KF(test_ode_func, true_obs_func, t_obs, y_obs, init_m, init_C_param, test_model_Q, noise_R_true, device, **enkf_kwargs)
    neg_log_likelihood.mean().backward()
    monitor_KF[0,ne,0] = neg_log_likelihood.detach()
    monitor_KF[0,ne,1:1+a_dim] = test_ode_func.a.grad.detach()
    monitor_KF[0,ne,1+a_dim] = test_model_Q.post_grad().item()
    monitor_KF[0,ne,2+a_dim] = test_model_Q.q_shape.grad.item()
    test_ode_func.a.grad.zero_()
    test_model_Q.q.grad.zero_()
    test_model_Q.q_shape.grad.zero_()

    for nr in range(n_runs):
      X, w, X_track, X_intermediate, w_track, w_intermediate, neg_log_likelihood = DAModel.BootstrapPF(test_ode_func, true_obs_func, t_obs, y_obs.unsqueeze(-2).repeat(1,1,1), N_ensem=N_ensem, init_m=init_m, init_C_param=init_C_param, model_Q_param=test_model_Q, noise_R_param=noise_R_true, device=device, proposal='optimal', adaptive_resampling=False, detach_every=None, **enkf_kwargs)
      # X, X_track, X_intermediate, neg_log_likelihood = DAModel.EnKF(test_ode_func, true_obs_func, t_obs, y_obs.unsqueeze(-2).repeat(1,1,1), N_ensem=N_ensem, init_m=init_m, init_C_param=init_C_param, model_Q_param=model_Q_true, noise_R_param=noise_R_true, device=device, detach_every=None, Cf_track=Cf_track, **enkf_kwargs)
      neg_log_likelihood.mean().backward()
      monitor_PF[nr,ne,0] = neg_log_likelihood.detach()
      monitor_PF[nr,ne,1:1+a_dim] = test_ode_func.a.grad.detach()
      monitor_PF[nr,ne,1+a_dim] = test_model_Q.post_grad().item()
      monitor_PF[nr,ne,2+a_dim] = test_model_Q.q_shape.grad.item()
      test_ode_func.a.grad.zero_()
      test_model_Q.q.grad.zero_()
      test_model_Q.q_shape.grad.zero_()

  monitor_EnKFs.append(monitor_EnKF)
  monitor_KFs.append(monitor_KF)
  monitor_PFs.append(monitor_PF)


mpl.rc_file_defaults()
rc('font', **{'family':'serif','serif':'Computer Modern'})
rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
mpl.rcParams['axes.titlesize']= 24
mpl.rcParams['axes.grid']= False
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['axes.labelpad'] = 0.5
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['lines.markeredgewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize']= 18
mpl.rcParams['legend.fontsize']= 20


save = False

# Plotting here
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8 * 3, 6 * 1), constrained_layout=False)

colors1 = cm.Blues(np.linspace(0.6,0.9,n_metaruns))
colors2 = cm.Greens(np.linspace(0.6,0.9,n_metaruns))

shapes = ['o', 'v', 's']
# Preprocessing here

for nm in range(0,n_metaruns):
  monitor_EnKF = monitor_EnKFs[nm]
  monitor_KF = monitor_KFs[nm]
  monitor_PF = monitor_PFs[nm]
  p1 = torch.abs(monitor_EnKF - monitor_KF)**2
  p1_new = p1.detach().clone()
  p1_new[:,:,0] = p1[:,:,0] / monitor_KF[0,0,0]**2
  p1_new[:,:,1] = torch.mean(p1[:,:,1:1+a_dim], dim=-1) / torch.mean(monitor_KF[0,0,1:1+a_dim]**2)
  p1_new[:,:,2] = torch.mean(p1[:,:,1+a_dim:], dim=-1) / torch.mean(monitor_KF[0,0,1+a_dim:]**2)
  p2 = torch.abs(monitor_PF - monitor_KF)**2
  p2_new = p2.detach().clone()
  p2_new[:,:,0] = p2[:,:,0] / monitor_KF[0,0,0]**2
  p2_new[:,:,1] = torch.mean(p2[:,:,1:1+a_dim], dim=-1) / torch.mean(monitor_KF[0,0,1:1+a_dim]**2)
  p2_new[:,:,2] = torch.mean(p2[:,:,1+a_dim:], dim=-1) / torch.mean(monitor_KF[0,0,1+a_dim:]**2)
  m1 = torch.mean(p1_new, dim=0) ** 0.5
  m2 = torch.mean(p2_new, dim=0) ** 0.5
  # m2, s2 = utils.mean_and_std(p2, 0)




  axes[0].plot(N_ensems, m1[:,0], shapes[nm]+'-',mfc='w', color=colors1[nm], label=rf"EnKF, $d_x={x_dims[nm]}$")
  axes[0].plot(N_ensems, m2[:,0], shapes[nm]+'-',mfc='w', color=colors2[nm], label=rf"PF, $d_x={x_dims[nm]}$")

  axes[1].plot(N_ensems, m1[:,1], shapes[nm]+'-',mfc='w', color=colors1[nm], label=f"EnKF, $d_x={x_dims[nm]}$")
  axes[1].plot(N_ensems, m2[:,1], shapes[nm]+'-',mfc='w', color=colors2[nm], label=f"PF, $d_x={x_dims[nm]}$")


  axes[2].plot(N_ensems, m1[:,2], shapes[nm]+'-',mfc='w', color=colors1[nm], label=f"EnKF, $d_x={x_dims[nm]}$")
  axes[2].plot(N_ensems, m2[:,2], shapes[nm]+'-',mfc='w', color=colors2[nm], label=f"PF, $d_x={x_dims[nm]}$")


x = N_ensems
axes[0].plot(x, 0.05/np.power(x, 0.5), 'r--', linewidth=3, label=r"$O(N^{-1/2})$")
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel(r"$N$")
axes[0].set_ylabel(r"$L^2$ estimation error (${\mathcal{L}}$)")
# axes[0].legend(loc='lower center',bbox_to_anchor=(0.5, 1.05),
#           ncol=3, fancybox=False, shadow=False)



axes[1].plot(x, 2/4/np.power(x, 0.5), 'r--', linewidth=3, label=r"$O(N^{-1/2})$")
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel(r"$N$")
axes[1].set_ylabel(r"$L^2$ estimation error ($\nabla_\alpha {\mathcal{L}}$)")

leg=axes[1].legend(loc='lower center',bbox_to_anchor=(0.5, -0.36),
          ncol=4, fancybox=False, shadow=False)
# leg._legend_box.align = "right"


axes[2].plot(x, 4/6/np.power(x, 0.5), 'r--', linewidth=3, label=r"$O(N^{-1/2})$")
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].set_xlabel(r"$N$")
axes[2].set_ylabel(r"$L^2$ estimation error ($\nabla_\beta {\mathcal{L}}$)")

# axes[2].legend(loc='lower left')

# Adjustments here
if save:
  plt.savefig(f"/content/drive/MyDrive/project_Yuming/codes/final_figures/Linear/20_40_80_off.pdf", bbox_inches='tight')