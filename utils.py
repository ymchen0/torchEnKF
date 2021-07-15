import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
import math

import matplotlib




from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
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

def softplus(t):
  return torch.log(1. + torch.exp(t))

def softplus_inv(t):
  return torch.log(-1. + torch.exp(t))

def softplus_grad(t):
  return torch.exp(t) / (1. + torch.exp(t))

def flat2matrix(t, truth=None):
  # t: (*batch_dims, d*d) --> (*batch_dim, d, d)
  batch_dim = t.shape[:-1]
  mat_dim = int(math.sqrt(t.shape[-1]))
  return torch.linalg.norm(t.view(*batch_dim, mat_dim, mat_dim) - truth, 'fro', dim=(-1,-2))


def visualize_matrix(mat, symmetric_error_bar=False):
  mat_plot = mat.detach().cpu().numpy()
  fig_ratio = mat_plot.shape[0] / mat_plot.shape[1]
  if symmetric_error_bar:
    vmax = max(mat_plot.max(), -mat_plot.min())
  fig = plt.figure(figsize=(8/(fig_ratio) , 8))
  ax = fig.add_subplot(111)
  if symmetric_error_bar:
    cf = ax.imshow(mat_plot, cmap=cm.bwr, vmax=vmax, vmin=-vmax)
  else:
    cf = ax.imshow(mat_plot, cmap=cm.bwr)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(cf, cax=cax)
  plt.show()


def unique_labels(ax):
  handles, labels = ax.get_legend_handles_labels()
  labels, ids = np.unique(labels, return_index=True)
  handles = [handles[i] for i in ids]
  return handles, labels

def mse_loss(input, target):
  return torch.mean((input - target) ** 2)

def mse_loss_last_dim(input, target):
  last_dim = input.shape[-1]
  return torch.mean((input.reshape(-1, last_dim) - target.reshape(-1, last_dim)) ** 2, dim=0)

def particle_mse_loss(input, target, weight):
  weight = weight.unsqueeze(-1)
  mean = torch.sum(weight * input, dim=-2) # (n_obs, *bs, x_dim)
  return mse_loss(mean, target)

def particle_mse_loss_last_dim(input, target, weight):
  weight = weight.unsqueeze(-1)
  mean = torch.sum(weight * input, dim=-2) # (n_obs, *bs, x_dim)
  return mse_loss_last_dim(mean, target)

def weighted_mse_loss(input, target, weight):
  return torch.mean(weight * (input - target) ** 2)

def ess(weight):
  # (*bdims, weight) -> (*bdims)
  return 1 / (weight**2).sum(dim=-1)

def mean_and_std(t, axis=None):
  t_np = t.detach().cpu().numpy()
  if axis is None:
    return np.mean(t_np), np.std(t_np)
  else:
    return np.mean(t_np, axis=axis), np.std(t_np, axis=axis)

def construct_exp(x_dim):
  exp = torch.zeros(x_dim, x_dim)
  for i in range(x_dim):
    for j in range(x_dim):
      exp[i, j] = -1. * abs(i-j)
  return exp

def shrink_batch_dim(t):
  # (a, *bs, b, c) --> (a, -1, b, c)
  return t.view(t.shape[0], -1, t.shape[-2], t.shape[-1])

def mean_over_all_but_last_k_dims(t, k):
  num_dims = len(t.shape)
  for _ in range(num_dims - k):
    t = t.mean(dim=0)
  return t

def set_lim_ticks(ax, low, high, n_ticks, ratio=0.05):
  res = (high-low)*ratio
  ax.set_ylim(low-res, high+res)
  ax.yaxis.set_ticks(np.linspace(low, high, n_ticks))
  return

def plot_monitor_res_new(monitor_res, methods, titles, truths, logscale, logscalex=None, truths_legends=None, truths2=None, truths2_legends=None, plot_truth_first=True, start_from_one=None, gridspec_kw=None, legend_order=None, groups=None, error_bar=False, error_bar_style="std", plots_to_show=None, n_cols=3, 
              subplots_adjust=None, subplot_width=8, subplot_height=6, subplot_groups=None, ax_d=None, linewidth=2, colors_list=None, custom_axes=None, x_axiss=None, save_location=None, file_name=None, load_location=None):
  '''
  Args:
    monitor_res: (list of n_methods *) (n_runs, n_epochs, n_monitors)
      e.g., for a 'method' Diff-EnKF, I have 4 repeated runs, statistics over 100 training epochs, of 6 different statistics (training loss, test loss, etc.)
      For each 'method', n_runs, n_epochs, n_monitors are the same
      For different 'method's, make sure each monitor has the same meaning
    method: (list of) method_names
    titles: (list of) monitor_names

  '''
  
  if monitor_res is None and load_location is not None:
    d = torch.load(load_location)
    monitor_res = d['monitor_res']
    x_axiss = d['x_axiss']

  if not isinstance(monitor_res, list):
    monitor_res = [monitor_res]
  if not isinstance(methods, list):
    methods = [methods]
  
  n_methods = len(monitor_res)
  n_subplots = 0
  for me in range(n_methods):
    if torch.is_tensor(monitor_res[me]):
      monitor_res[me] = monitor_res[me].detach().cpu().numpy()
    n_subplots = max(n_subplots, monitor_res[me].shape[2])
  
  if groups is None:
    groups = [(i,i+1) for i in range(n_subplots)]
  else:
    new_group = []
    for g in range(len(groups)):
      if g == 0:
        new_group += [(i,i+1) for i in range(groups[g][0])]
      else:
        new_group += [(i,i+1) for i in range(groups[g-1][1], groups[g][0])]
      new_group += [groups[g]]
    new_group += [(i,i+1) for i in range(groups[-1][1], n_subplots)]
    groups = new_group
    n_subplots = len(groups)  
  
  if plots_to_show is not None:
    n_subplots = len(plots_to_show)
    groups = [groups[i] for i in plots_to_show]
    titles = [titles[i] for i in plots_to_show]
  def unravel(i, n_cols):
    return (i//n_cols, i%n_cols)

  if subplot_groups is not None:
    row_range, col_range = subplot_groups
    tmp = n_subplots - 1 + (row_range[1]-row_range[0])*(col_range[1]-col_range[0])
    n_rows = (tmp-1)//n_cols + 1
  else:
    n_rows = (n_subplots-1)//n_cols + 1
  

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(subplot_width * n_cols, subplot_height * n_rows), constrained_layout=False, gridspec_kw=gridspec_kw)

  if n_rows == 1:
    axes = np.array([axes])
  if n_cols == 1:
    axes = axes[:,np.newaxis]


  if subplot_groups is not None:
    gs = axes[row_range[0],col_range[0]].get_gridspec()
    for row in range(row_range[0], row_range[1]):
      for col in range(col_range[0], col_range[1]):
        axes[row, col].remove()
    axbig = fig.add_subplot(gs[row_range[0]:row_range[1], col_range[0]:col_range[1]])
    ax_d = {sp:axes[ax_d[sp]] for sp in range(1, n_subplots)}
    ax_d[0] = axbig
  else:
    ax_d = {sp:axes[unravel(sp,n_cols)] for sp in range(n_subplots)}
    

  # colors_list = [cm.Wistia, cm.Blues, cm.Greens]
  # colors_list = [cm.Reds, cm.Blues, cm.Greens]
  colors_list = [cm.Blues, cm.Greens, cm.Reds] if colors_list is None else colors_list
  
  for sp in range(n_subplots):
    if start_from_one is not None and sp in start_from_one:
      start = 1
    else:
      start = 0
    idx = unravel(sp, n_cols)
    for me in range(n_methods):
      cur_res = monitor_res[me] # (n_runs, n_epochs, n_monitors)
      cur_res_mean = cur_res.mean(axis=0)
      cur_res_std = cur_res.std(axis=0)
      cur_res_75_quantile = np.quantile(cur_res, 0.75, axis=0)
      cur_res_50_quantile = np.quantile(cur_res, 0.5, axis=0)
      cur_res_25_quantile = np.quantile(cur_res, 0.25, axis=0)
      n_runs, n_epochs, n_monitors = cur_res.shape
      if x_axiss is None:
        x_axis = np.arange(n_epochs)
      else:
        x_axis = x_axiss[sp]

      for m in range(groups[sp][0], groups[sp][1]):
        if m >= n_monitors:
          continue
        colors = colors_list[me](np.linspace(0.5,0.7,n_runs+1))
        if plot_truth_first:
          if m < len(truths) and truths[m] is not None:
            if (truths_legends is None or truths_legends[m] is None):
              label=None
            elif truths_legends[m] == 't':
              label = "Truth"
            else:
              label = truths_legends[m]
            ax_d[sp].axhline(y=truths[m], color='r', linestyle='--', linewidth=2.5, label=label) 
          if truths2 is not None and m < len(truths2) and truths2[m] is not None:
            if (truths2_legends is None or truths2_legends[m] is None):
              label=None
            elif truths2_legends[m] == 't':
              label = "Truth"
            else:
              label = truths2_legends[m]
            ax_d[sp].axhline(y=truths2[m], color='r', linestyle='--', linewidth=2.5, label=label)

        

        if not error_bar:
          for r in range(n_runs):
            ax_d[sp].plot(x_axis[start:], cur_res[r,start:x_axis.shape[0],m], color=colors[-r-1],linewidth=linewidth, label=methods[me])
        else:
          if error_bar_style == "std":
            ax_d[sp].plot(x_axis[start:], cur_res_mean[start:x_axis.shape[0],m], color=colors[-1],linewidth=linewidth, label=methods[me])
            ax_d[sp].fill_between(x_axis[start:], cur_res_mean[start:x_axis.shape[0],m]+2*cur_res_std[start:x_axis.shape[0],m], cur_res_mean[start:x_axis.shape[0],m]-2*cur_res_std[start:x_axis.shape[0],m], facecolor=colors[-1], edgecolor=None, alpha=0.2)
          elif error_bar_style == "quantile":
            ax_d[sp].plot(x_axis[start:], cur_res_50_quantile[start:x_axis.shape[0],m], color=colors[-1],linewidth=linewidth, label=methods[me])
            ax_d[sp].fill_between(x_axis[start:], cur_res_75_quantile[start:x_axis.shape[0],m], cur_res_25_quantile[start:x_axis.shape[0],m], facecolor=colors[-1], edgecolor=None, alpha=0.2)
        
        if not plot_truth_first:
          if m < len(truths) and truths[m] is not None:
            if (truths_legends is None or truths_legends[m] is None):
              label=None
            elif truths_legends[m] == 't':
              label = "Truth"
            else:
              label = truths_legends[m]
            ax_d[sp].axhline(y=truths[m], color='r', linestyle='--', linewidth=2.5, label=label) 
          if truths2 is not None and m < len(truths2) and truths2[m] is not None:
            if (truths2_legends is None or truths2_legends[m] is None):
              label=None
            elif truths2_legends[m] == 't':
              label = "Truth"
            else:
              label = truths2_legends[m]
            ax_d[sp].axhline(y=truths2[m], color='k', linestyle='-', linewidth=2.5, label=label) 
    if titles[sp] is not None:
      ax_d[sp].set_title(f"{titles[sp]}")#, {n_runs} independent run(s)
    if sp in logscale:
      ax_d[sp].set_yscale('log')
    if logscalex is not None and sp in logscalex:
      ax_d[sp].set_xscale('log')
      ax_d[sp].invert_xaxis()
    # if truths[i] is not None:
    #   axes[idx].axhline(y=truths[i], color='r', linestyle='--', linewidth=3, label="True value")

    # Handle legends
    handles, labels = unique_labels(ax_d[sp])
    if legend_order is None:
      ax_d[sp].legend(handles, labels, loc='upper right')
    else:
      ax_d[sp].legend([handles[idx] for idx in legend_order],[labels[idx] for idx in legend_order], loc='upper right')


  if custom_axes is not None:
    custom_axes(axes, ax_d)
  
  if subplots_adjust is not None:
    plt.subplots_adjust(wspace=subplots_adjust)

  if save_location is not None:
    # torch.save({'monitor_res': monitor_res, 'x_axiss':x_axiss},
    #        save_location)
    plt.savefig(save_location+".pdf", bbox_inches='tight')

  plt.show()



def plot_dynamic(t_eval, out, t_obs=None, y_obs=None, online_color=None, fig_num_limit=None, text="obs", contour_plot=False):
  t_eval = t_eval.cpu().numpy()
  if t_obs is None:
    t_obs = t_eval
  x_dim = out.shape[1]
  if fig_num_limit is not None and x_dim > fig_num_limit:
    x_dim = fig_num_limit
  n_eval = t_eval.shape[0]
  out_plot = out.detach().cpu().numpy()
  if y_obs is not None:
    obs_dim = y_obs.shape[1]
    if fig_num_limit is not None and obs_dim > fig_num_limit:
      obs_dim = fig_num_limit
    obs_plot = y_obs.detach().cpu().numpy()
  fig, axes = plt.subplots(x_dim, 1, sharex=True, figsize=(6, 3*x_dim), constrained_layout=True)
  if x_dim == 1:
    axes = np.array([axes])
  fig.suptitle('True dynamic (for each coordinate)', fontsize=15)
  for i, ax in enumerate(axes):
    # Note that: ax = axes[i]
    ax.plot(t_eval, out_plot[:, i], 'r', linewidth=2, label="Truth")
    y_lim=ax.get_ylim()
    ax.set_ylim(y_lim)
    if y_obs is not None and i < obs_dim:
      ax.plot(t_obs, obs_plot[:, i], 'y', label=text)
    ax.set_xlabel('$t$', fontsize=10)
    # ax.set_ylabel('$u_1$', rotation=0, fontsize=15)
    ax.legend(loc='upper right')
  
  if x_dim == 2 or x_dim == 3:
    fig2 = plt.figure(figsize=(12,6))
    fig2.suptitle('True dynamic', fontsize=15)
    if x_dim == 2:
      ax21 = fig2.add_subplot(121)
      ax21.plot(out_plot[:, 0], out_plot[:, 1], 'r', linewidth=2, label="Truth")
    elif x_dim == 3:
      ax21 = fig2.add_subplot(121, projection='3d')
      ax21.plot(out_plot[:, 0], out_plot[:, 1], out_plot[:, 2], c='r', linewidth=2, label="Truth")
      ax21.scatter(out_plot[0, 0], out_plot[0, 1], out_plot[0, 2], marker='*', s=25, linestyle="None", color="k", label="start")
      ax21.scatter(out_plot[-1, 0], out_plot[-1, 1], out_plot[-1, 2], marker='^', s=25, linestyle="None", color="k", label="end")
    ax21.legend(loc='upper right')
    if y_obs is not None:
      if obs_dim == 1:
        ax22 = fig2.add_subplot(122)
        ax22.plot(t_obs, obs_plot, 'y', label=text)
      elif obs_dim == 2:
        ax22 = fig2.add_subplot(122)
        # ax22.plot(obs_plot[:, 0], obs_plot[:, 1], 'y', label=text)
        ax22.plot(obs_plot[0, 0], obs_plot[0, 1], marker='*', markersize=12, linestyle="None", color="k", label=f"start")
        ax22.plot(obs_plot[-1, 0], obs_plot[-1, 1], marker='^', markersize=12, linestyle="None", color="k", label=f"end")
        ax22.quiver(obs_plot[:-1, 0], obs_plot[:-1, 1], obs_plot[1:, 0]-obs_plot[:-1, 0], obs_plot[1:, 1]-obs_plot[:-1, 1], np.linspace(0,1,len(t_obs)-1), scale_units='xy', angles='xy', scale=1)
        if x_dim == 2:
          plt.setp(ax22, xlim=ax21.get_xlim(),ylim=ax21.get_ylim())
      elif obs_dim == 3:
        ax22 = fig2.add_subplot(122, projection='3d')
        if online_color is None:
          ax22.plot(obs_plot[:, 0], obs_plot[:, 1], obs_plot[:, 2], c='y', label=text)
        else:
          online_color = online_color.detach().cpu().numpy().squeeze()
          online_color = (online_color - np.min(online_color))/np.ptp(online_color)
          n_obs = obs_plot.shape[0]
          for i in range(0, n_obs-1):
            ax22.plot(obs_plot[i:i+2, 0], obs_plot[i:i+2, 1], obs_plot[i:i+2, 2], color=plt.cm.Blues(online_color[i]*0.75+0.25))
        # for i in range(len(t_obs)-1):
        #   ax22.plot(obs_plot[i:i+2, 0], obs_plot[i:i+2, 1], obs_plot[i:i+2, 2], color=plt.cm.jet(255*i/len(t_obs)))
        ax22.scatter(obs_plot[0, 0], obs_plot[0, 1], obs_plot[0, 2], marker='*', s=25, linestyle="None", color="k", label=f"start")
        ax22.scatter(obs_plot[-1, 0], obs_plot[-1, 1], obs_plot[-1, 2], marker='^', s=25, linestyle="None", color="k", label=f"end")
        # ax22.quiver(obs_plot[:-1, 0], obs_plot[:-1, 1], obs_plot[:-1, 2], obs_plot[1:, 0]-obs_plot[:-1, 0], obs_plot[1:, 1]-obs_plot[:-1, 1], obs_plot[1:, 2]-obs_plot[:-1, 2], np.linspace(0,1,n_eval-1))
        if x_dim == 3:
          plt.setp(ax22, xlim=ax21.get_xlim(),ylim=ax21.get_ylim(), zlim=ax21.get_zlim())
      ax22.legend(loc='upper right')

  if contour_plot:
    vert = list(range(out.shape[1]))
    fig3 = plt.figure(figsize=(18,6))
    fig3.suptitle('Time evolution', fontsize=15)
    ax31 = fig3.add_subplot(111)
    vmin = out_plot.min()
    vmax = out_plot.max()
    cf1 = ax31.contourf(t_eval, vert, out_plot.T, levels=np.linspace(vmin, vmax, 8))
    
    plt.colorbar(cf1)
    if y_obs is not None and y_obs.shape[1] == out.shape[1]:
      vert_obs = list(range(y_obs.shape[1]))
      fig4 = plt.figure(figsize=(18,6))
      ax41 = fig4.add_subplot(111)
      cf2 = ax41.contourf(t_eval, vert_obs, obs_plot.T, levels=np.linspace(vmin, vmax, 8), extend='both')
      plt.colorbar(cf2)
      fig5 = plt.figure(figsize=(18,6))
      ax51 = fig5.add_subplot(111)
      cf3 = ax51.contourf(t_eval, vert_obs, out_plot.T - obs_plot.T, cmap=cm.bwr, levels=np.linspace(-vmax, vmax, 20), extend='both')
      plt.colorbar(cf3)
  plt.show()

# def plot_filter_old(t_eval, out, E, name="Ensemble", plot_all=True, compare_F_S=False, fig_num_limit=None):
#   t_eval = t_eval.cpu().numpy()
#   x_dim = out.shape[1]
#   if fig_num_limit is not None and x_dim > fig_num_limit:
#     x_dim = fig_num_limit
#   out_plot = out.detach().cpu().numpy()
#   n_cols = 2 if compare_F_S else 1
#   fig, axes = plt.subplots(x_dim, n_cols, sharex='col', sharey='row', figsize=(8*n_cols, 4*x_dim), constrained_layout=True)
  
#   if x_dim == 1:
#     axes = np.array([axes])
#   if not compare_F_S:
#     axes = axes[:, np.newaxis]

#   if name == "Ensemble":
#     X_track_plot = E.X_track.detach().cpu().numpy()
#     if compare_F_S:
#       X_smooth_plot = E.X_smooth.detach().cpu().numpy()
#     else:
#       X_smooth_plot = E.X_track.detach().cpu().numpy()
#     N_ensem = X_track_plot.shape[1]
#     track_mean = X_track_plot.mean(axis=1)
#     track_std = np.std(X_track_plot, axis=1)
#     smooth_mean = X_smooth_plot.mean(axis=1)
#     smooth_std = np.std(X_smooth_plot, axis=1)
#     fig.suptitle("Ensemble Kalman", fontsize=15)
#   elif name == "Kalman":
#     track_mean = E.mu_track.detach().cpu().numpy()
#     V_track = E.V_track.detach().cpu().numpy()
#     track_std = np.sqrt(np.diagonal(V_track, axis1=1, axis2=2))
#     smooth_mean = E.mu_smooth.detach().cpu().numpy()
#     V_smooth = E.V_smooth.detach().cpu().numpy()
#     smooth_std = np.sqrt(np.diagonal(V_smooth, axis1=1, axis2=2))
#     fig.suptitle("Kalman", fontsize=15)

#   for i, ax in enumerate(axes):
#     if name == "Ensemble" and plot_all:
#       for n in range(N_ensem):
#         ax[0].plot(np.insert(t_eval, 0, 0.), X_smooth_plot[:, n, i], 'b', linewidth=0.5, label='EnKS')
#         if compare_F_S:
#           ax[1].plot(np.insert(t_eval, 0, 0.), X_track_plot[:, n, i], 'b', linewidth=0.5, label='EnKF')
#     else:
#       ax[0].plot(np.insert(t_eval, 0, 0.), smooth_mean[:, i], 'b--', linewidth=2, label='Smoother mean')
#       ax[0].plot(np.insert(t_eval, 0, 0.), smooth_mean[:, i] + 2*smooth_std[:, i], 'b', linewidth=0.5, label='+2 std')
#       ax[0].plot(np.insert(t_eval, 0, 0.), smooth_mean[:, i] - 2*smooth_std[:, i], 'b', linewidth=0.5, label='-2 std')
#       if compare_F_S:
#         ax[1].plot(np.insert(t_eval, 0, 0.), track_mean[:, i], 'b--', linewidth=2, label='Filter mean')
#         ax[1].plot(np.insert(t_eval, 0, 0.), track_mean[:, i] + 2*track_std[:, i], 'b', linewidth=0.5, label='+2 std')
#         ax[1].plot(np.insert(t_eval, 0, 0.), track_mean[:, i] - 2*track_std[:, i], 'b', linewidth=0.5, label='-2 std')
#     ax[0].plot(t_eval, out_plot[:, i], 'r', linewidth=3, label="Truth")
#     ax[0].set_xlabel('$t$', fontsize=10)
#     handles, labels = unique_labels(ax[0])
#     ax[0].legend(handles, labels, loc='upper right')
#     # ax[0].set_ylim(-30, 60)

#     if compare_F_S:
#       ax[1].plot(t_eval, out_plot[:, i], 'r', linewidth=3, label="Truth")
#       ax[1].set_xlabel('$t$', fontsize=10)
#       handles, labels = unique_labels(ax[1])
#       ax[1].legend(handles, labels, loc='upper right')
#   plt.show()

def plot_loss(n_epochs, loss_1_track, loss_2_track):
  fig = plt.figure(figsize=(16, 6))
  ax1 = fig.add_subplot(121)
  ax2 = fig.add_subplot(122)
  ax1.plot(list(range(n_epochs)), loss_1_track, label='observation model loss')
  ax2.plot(list(range(n_epochs)), loss_2_track, label='dynamic model loss')
  ax1.set_xlabel('$EM iter$', fontsize=10)
  ax2.set_xlabel('$EM iter$', fontsize=20)
  ax1.legend()
  ax2.legend()
  # ax1.set_yscale('log')
  # ax2.set_yscale('log')
  plt.show()

def plot_monitor_res(monitor_res, method, monitor_res_cmp, method_cmp, titles, truths, logscale):
  '''
  Args:
    monitor_res: list of n_methods * (n_runs, n_epochs, n_monitors) or (n_runs, n_epochs, n_monitors)
    method: list of method_names
    titles: list of monitor_names
  '''
  
  # monitor_res: (n_runs, n_epochs, n_monitors)
  monitor_res = monitor_res.detach().cpu().numpy()
  n_runs = monitor_res.shape[0]
  n_epochs = monitor_res.shape[1]
  n_monitors = monitor_res.shape[2]

  n_subplots = n_monitors
  if monitor_res_cmp is not None:
    monitor_res_cmp = monitor_res_cmp.detach().cpu().numpy()
    n_subplots = max(n_subplots, monitor_res_cmp.shape[2])
  
  n_cols = 3
  n_rows = (n_subplots-1)//n_cols + 1
  def unravel(i, n_cols):
    return (i//n_cols, i%n_cols)

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), constrained_layout=False)
  if n_rows == 1:
    axes = np.array([axes])
  colors = cm.Blues(np.linspace(0.5,1,n_runs))
  for i in range(n_monitors):
    idx = unravel(i, n_cols)
    for j in range(n_runs):
      axes[idx].plot(np.arange(n_epochs), monitor_res[j, :, i], color=colors[j], label=method)
    axes[idx].set_title(f"{titles[i]}, {n_runs} run(s)", fontsize=20)
    if i in logscale:
      axes[idx].set_yscale('log')
    if truths[i] is not None:
      axes[idx].axhline(y=truths[i], color='r', linestyle='--', linewidth=3, label="True value")

  if monitor_res_cmp is not None:
    colors_cmp = cm.Wistia(np.linspace(0.5, 1, n_runs))
    n_monitors_cmp = monitor_res_cmp.shape[2]
    for i in range(n_monitors_cmp):
      idx = unravel(i, n_cols)
      for j in range(n_runs):
        axes[idx].plot(np.arange(n_epochs), monitor_res_cmp[j, :, i], color=colors_cmp[j], label=method_cmp)
      axes[idx].set_title(f"{titles[i]}, {n_runs} run(s)", fontsize=20)

  # Handle legends
  for i in range(n_monitors):
    idx = unravel(i, n_cols)
    handles, labels = unique_labels(axes[idx])
    axes[idx].legend(handles, labels, loc='upper right')

  # axes[0,0].set_ylim(-0.4, 0.7)
  # axes[0,2].set_ylim(30, 60)
  # axes[1,0].set_ylim(30, 60)
  # axes[1,1].set_ylim(7500, 20000)
  # axes[1,2].set_ylim(0.75, 3)
  # axes[2,0].set_ylim(0.75, 3)
  # axes[2,1].set_ylim(0, 70)
  plt.show()


def plot_1d_nll_with_grad(x_axis, nll, nll_kalman, N_ensem_list, x_label, x_truth, grad=None, grad_kalman=None, recon=None, dx=0.7, grad_every=10, delta=10, x_scale=None):
  # nll: (Ne, n_pts, n_trial)
  x_axis = x_axis.detach().cpu().numpy()
  nll = nll.detach().cpu().numpy()
  # recon = recon.detach().cpu().numpy()
  ne_trials = nll.shape[0]
  n_pts = nll.shape[1]
  n_trials = nll.shape[2]
  indices_grad = list(range(0,n_pts,grad_every))
  if grad is not None:
    grad = grad.detach().cpu().numpy()


  fig = plt.figure(figsize=(24, 9))
  ax1 = fig.add_subplot(121)
  ax2 = fig.add_subplot(122)
  # ax3 = fig.add_subplot(223)
  colors = cm.viridis_r(np.linspace(0,1,ne_trials))

  for ne in range(ne_trials):
    nll_mean = nll[ne].mean(axis=1)
    nll_std = nll[ne].std(axis=1)
    # recon_mean = recon.mean(axis=0)
    # recon_std = np.std(recon, axis=0)
    if grad is not None:
      grad_mean = grad[ne].mean(axis=1)
      grad_std = grad[ne].std(axis=1)
    
    ax1.plot(x_axis, nll_mean, linewidth=3, label=f"EnKF, N = {N_ensem_list[ne]}")
    ax1.fill_between(x_axis, nll_mean - 2*nll_std, nll_mean + 2*nll_std, alpha=0.3)
    ax1.set_xlabel(x_label, fontsize=15)
    ax1.set_title(f"nll versus {x_label}, averaged {n_trials} EnKF trials", fontsize=20)
    # ax3.plot(x_axis, recon_mean, 'k', label='recon $\pm$ 2std')
    # ax3.fill_between(x_axis, recon_mean - 2*recon_std, recon_mean + 2*recon_std, alpha=0.3)
    # ax3.set_xlabel(x_label, fontsize=15)
    # ax3.set_title(f"recon_error versus F, averaged {n_trials} EnKF trials", fontsize=15)
    if delta is not None and ne == ne_trials - 1:
      ylim1 = [(nll_mean - 2 * nll_std).min() - delta[0], (nll_mean + 2 * nll_std).max()+delta[0]]
      


    if grad is not None:
      ax2.plot(x_axis, grad_mean, linewidth=3, label=f"EnKF, N = {N_ensem_list[ne]}")
      ax2.fill_between(x_axis, grad_mean - 2*grad_std, grad_mean + 2*grad_std, alpha=0.3)
      ax2.set_xlabel(x_label, fontsize=15)
      ax2.set_title(f"nll_grad versus {x_label}, w/ autograd, averaged {n_trials} EnKF trials", fontsize=20)
      if delta is not None and ne == ne_trials - 1:
        ylim2 = [(grad_mean - 2 * grad_std).min() - delta[1], (grad_mean + 2 * grad_std).max()+delta[1]]

      # for i in indices_grad:
      #   ax3.quiver(x_axis[i]*np.ones(n_trials), nll_mean[i]*np.ones(n_trials), dx*np.ones(n_trials), dx*grad[ne, i], angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=3, headlength=5, color="blue")
      # ax3.fill_between(x_axis, nll_mean - 2*nll_std, nll_mean + 2*nll_std, alpha=0.3)
      # ax3.set_xlabel(x_label, fontsize=15)
      # ax3.set_title(f"nll_grad versus {x_label}, {n_trials} EnKF trials (w/ backprop)", fontsize=15)

  if nll_kalman is not None:
    nllk = nll_kalman.detach().cpu().numpy()
    ax1.plot(x_axis, nllk, 'r', linewidth=3, label=f"KF")
    if grad_kalman is not None:
      gradk = grad_kalman.detach().cpu().numpy()
      ax2.plot(x_axis, gradk, 'r', linewidth=3, label=f"KF")

  ax1.axvline(x=x_truth, color='k', linestyle='--', linewidth=3, label=f"True {x_label}")
  ax2.axvline(x=x_truth, color='k', linestyle='--', linewidth=3, label=f"True {x_label}")
  # ax3.axvline(x=x_truth, color='r', linestyle='--', linewidth=3, label=f"True {x_label}")

  if delta is not None:
    ax1.set_ylim(ylim1)
  ax1.legend(loc='upper right')
  if grad is not None:
    if delta is not None:
      ax2.set_ylim(ylim2)
    ax2.legend(loc='upper right')
    # if delta is not None:
    #   ax3.set_ylim(ylim1)

  ax1.xaxis.set_tick_params(labelsize=20)
  ax1.yaxis.set_tick_params(labelsize=20)
  ax2.xaxis.set_tick_params(labelsize=20)
  ax2.yaxis.set_tick_params(labelsize=20)
  if x_scale == "log":
    ax1.set_xscale('log')
    ax2.set_xscale('log')
  # ax3.xaxis.set_tick_params(labelsize=20)
  # ax3.yaxis.set_tick_params(labelsize=20)
  plt.show()


def plot_2d_nll_with_grad(x_axis, y_axis, nll, nll_kalman, N_ensem, xy_label, xy_truth, grad=None, grad_kalman=None, recon=None, arrow_scale=1, dx=0.7, grad_every=10, xy_scale=[None,None]):
  # nll: (n_ptsx, n_ptsy, n_trial)
  # grad: (n_ptsx, n_ptsy, 2, n_trial)
  x_axis = x_axis.detach().cpu().numpy()
  y_axis = y_axis.detach().cpu().numpy()
  nll = nll.detach().cpu().numpy()
  # recon = recon.detach().cpu().numpy()
  n_ptsx = nll.shape[0]
  n_ptsy = nll.shape[1]
  n_trials = nll.shape[2]
  indices_gradx = np.arange(0,n_ptsx, grad_every)
  indices_grady = np.arange(0,n_ptsy, grad_every)
  if grad is not None:
    grad = grad.detach().cpu().numpy()

  fig, axes = plt.subplots(2, 3, figsize=(24, 16), constrained_layout=False)
  
  # colors = cm.viridis_r(np.linspace(0,1,ne_trials))

  nll_mean = nll.mean(axis=-1)
  nll_std = nll.std(axis=-1)
  # recon_mean = recon.mean(axis=0)
  # recon_std = np.std(recon, axis=0)
  if grad is not None:
    gradx_mean = grad.mean(axis=-1)[:, :, 0]
    grady_mean = grad.mean(axis=-1)[:, :, 1]
    grad_std = np.sqrt((grad**2).sum(axis=-2)).std(axis=-1)
  

  vmax = nll_mean.max()
  vmin = nll_mean.min()
  if nll_kalman is not None:
    nllk = nll_kalman.detach().cpu().numpy()
    vmax = nllk.max()
    vmin = nllk.min()
    cf02 = axes[0,2].contourf(x_axis, y_axis, nllk.T, levels=np.linspace(vmin, vmax, 25))
    axes[0,2].set_xlabel(xy_label[0], fontsize=15)
    axes[0,2].set_ylabel(xy_label[1], fontsize=15)
    axes[0,2].set_title(f"nll versus ({xy_label[0]}, {xy_label[1]}), Kalman Filter", fontsize=20)
    divider = make_axes_locatable(axes[0,2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cf02, cax=cax)
    if grad_kalman is not None:
      gradkx = grad_kalman[:, :, 0]
      gradky = grad_kalman[:, :, 1]
      axes[1,2].quiver(x_axis[0:n_ptsx:grad_every], y_axis[0:n_ptsy:grad_every], -gradkx[0:n_ptsx:grad_every,0:n_ptsy:grad_every].T, -gradky[0:n_ptsx:grad_every,0:n_ptsy:grad_every].T, scale_units='xy', angles='xy', scale=arrow_scale)
      axes[1,2].set_xlabel(xy_label[0], fontsize=15)
      axes[1,2].set_ylabel(xy_label[1], fontsize=15)
      axes[1,2].set_title(f"nll_grad versus ({xy_label[0]}, {xy_label[1]}), Kalman Filter", fontsize=20)

  cf00 = axes[0,0].contourf(x_axis, y_axis, nll_mean.T, levels=np.linspace(vmin, vmax, 25), extend='both')
  axes[0,0].set_xlabel(xy_label[0], fontsize=15)
  axes[0,0].set_ylabel(xy_label[1], fontsize=15)
  axes[0,0].set_title(f"nll versus ({xy_label[0]}, {xy_label[1]}), N={N_ensem}, averaged {n_trials} EnKF trials", fontsize=20)
  divider = make_axes_locatable(axes[0,0])
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(cf00, cax=cax)
  cf01 = axes[0,1].contourf(x_axis, y_axis, nll_std.T)
  axes[0,1].set_xlabel(xy_label[0], fontsize=15)
  axes[0,1].set_ylabel(xy_label[1], fontsize=15)
  axes[0,1].set_title(f"std", fontsize=20)
  divider = make_axes_locatable(axes[0,1])
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(cf01, cax=cax)
  # ax3.plot(x_axis, recon_mean, 'k', label='recon $\pm$ 2std')
  # ax3.fill_between(x_axis, recon_mean - 2*recon_std, recon_mean + 2*recon_std, alpha=0.3)
  # ax3.set_xlabel(x_label, fontsize=15)
  # ax3.set_title(f"recon_error versus F, averaged {n_trials} EnKF trials", fontsize=15)
  
  if grad is not None:
    axes[1,0].quiver(x_axis[0:n_ptsx:grad_every], y_axis[0:n_ptsy:grad_every], -gradx_mean[0:n_ptsx:grad_every,0:n_ptsy:grad_every].T, -grady_mean[0:n_ptsx:grad_every,0:n_ptsy:grad_every].T, scale_units='xy', angles='xy', scale=arrow_scale)
    axes[1,0].set_xlabel(xy_label[0], fontsize=15)
    axes[1,0].set_ylabel(xy_label[1], fontsize=15)
    axes[1,0].set_title(f"nll_grad versus ({xy_label[0]}, {xy_label[1]}), N={N_ensem}, averaged {n_trials} EnKF trials", fontsize=20)
    cf11 = axes[1,1].contourf(x_axis, y_axis, grad_std.T)
    axes[1,1].set_xlabel(xy_label[0], fontsize=15)
    axes[1,1].set_ylabel(xy_label[1], fontsize=15)
    axes[1,1].set_title(f"std of norm", fontsize=20)
    divider = make_axes_locatable(axes[1,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cf11, cax=cax)

  for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
      axes[i,j].axvline(x=xy_truth[0], color='k', linestyle='--', linewidth=3, label=f"True {xy_label[0]}")
      axes[i,j].axhline(y=xy_truth[1], color='k', linestyle='-.', linewidth=3, label=f"True {xy_label[1]}")
      axes[i,j].xaxis.set_tick_params(labelsize=20)
      axes[i,j].yaxis.set_tick_params(labelsize=20)
      if xy_scale[0] == "log":
        axes[i,j].set_xscale('log')
      if xy_scale[1] == "log":
        axes[i,j].set_yscale('log')

  # f = plt.figure(figsize=(9,9))
  # ax = plt.axes(projection='3d')
  # cf = ax.contour3D(x_axis, y_axis, nll_mean.T, levels=np.linspace(vmin, vmax, 25), extend='both')
  # divider = make_axes_locatable(ax)
  # cax = divider.append_axes("right", size="5%", pad=0.05)
  # plt.colorbar(cf, cax=cax)

  plt.show()
  return

def plot_filter(t_eval, out, filter_track, name="Ensemble", plot_all=False, compare_F_S=False, fig_num_limit=None):

  t_eval = t_eval.cpu().numpy()
  x_dim = out.shape[1]
  if fig_num_limit is not None and x_dim > fig_num_limit:
    x_dim = fig_num_limit
  out_plot = out.detach().cpu().numpy()
  n_cols = 2 if compare_F_S else 1
  fig, axes = plt.subplots(x_dim, n_cols, sharex='col', sharey='row', figsize=(8*n_cols, 4*x_dim), constrained_layout=True)
  
  if x_dim == 1:
    axes = np.array([axes])
  if not compare_F_S:
    axes = axes[:, np.newaxis]

  if name == "Ensemble":
    X_track_plot = filter_track.detach().cpu().numpy()
    X_smooth_plot = filter_track.detach().cpu().numpy()
    N_ensem = X_track_plot.shape[1]
    track_mean = X_track_plot.mean(axis=1)
    track_std = np.std(X_track_plot, axis=1)
    smooth_mean = X_smooth_plot.mean(axis=1)
    smooth_std = np.std(X_smooth_plot, axis=1)
    fig.suptitle("Ensemble Kalman", fontsize=15)
  elif name == "Kalman":
    track_mean = filter_track[0].detach().cpu().numpy()
    V_track = filter_track[1].detach().cpu().numpy()
    track_std = np.sqrt(np.diagonal(V_track, axis1=1, axis2=2))
    smooth_mean = filter_track[0].detach().cpu().numpy()
    V_smooth = filter_track[1].detach().cpu().numpy()
    smooth_std = np.sqrt(np.diagonal(V_smooth, axis1=1, axis2=2))
    fig.suptitle("Kalman", fontsize=15)
  elif name == "Particle":
    w_track = filter_track[0].detach().cpu().numpy() # n_obs, N_ensem
    X_track = filter_track[1].detach().cpu().numpy() # n_obs, N_ensem, x_dim
    w_track = w_track[...,np.newaxis] # (n_obs, N_ensem, 1)
    smooth_mean = np.sum(w_track * X_track, axis=1, keepdims=True) # (n_obs, 1, x_dim)
    smooth_std = np.sum(w_track * (X_track - smooth_mean)**2, axis=1) # (n_obs, x_dim)
    smooth_mean = np.squeeze(smooth_mean, 1)
    fig.suptitle("Particle", fontsize=15)

  for i, ax in enumerate(axes):
    ax[0].plot(t_eval, out_plot[:, i], 'r', linewidth=3, label="Truth")
    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()
    if name == "Ensemble" and plot_all:
      for n in range(N_ensem):
        ax[0].plot(t_eval, X_smooth_plot[:, n, i], 'b', linewidth=0.5, label='EnKS')
        if compare_F_S:
          ax[1].plot(t_eval, X_track_plot[:, n, i], 'b', linewidth=0.5, label='EnKF')
    else:
      ax[0].plot(t_eval, smooth_mean[:, i], 'b--', linewidth=1, label='Smoother mean')
      ax[0].plot(t_eval, smooth_mean[:, i] + 2*smooth_std[:, i], 'b', linewidth=0.5, label='+2 std')
      ax[0].plot(t_eval, smooth_mean[:, i] - 2*smooth_std[:, i], 'b', linewidth=0.5, label='-2 std')
      if compare_F_S:
        ax[1].plot(t_eval, track_mean[:, i], 'b--', linewidth=1, label='Filter mean')
        ax[1].plot(t_eval, track_mean[:, i] + 2*track_std[:, i], 'b', linewidth=0.5, label='+2 std')
        ax[1].plot(t_eval, track_mean[:, i] - 2*track_std[:, i], 'b', linewidth=0.5, label='-2 std')
    plt.setp(ax[0], xlim=xlim,ylim=ylim)
    ax[0].set_xlabel('$t$', fontsize=10)
    handles, labels = unique_labels(ax[0])
    ax[0].legend(handles, labels, loc='upper right')
    # ax[0].set_ylim(-30, 60)

    if compare_F_S:
      ax[1].plot(t_eval, out_plot[:, i], 'r', linewidth=3, label="Truth")
      ax[1].set_xlabel('$t$', fontsize=10)
      handles, labels = unique_labels(ax[1])
      ax[1].legend(handles, labels, loc='upper right')
  plt.show()










