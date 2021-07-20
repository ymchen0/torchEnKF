import torch
import math
from torchEnKF import misc
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint

from tqdm import tqdm
import time

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

def EnKF(ode_func, obs_func, t_obs, y_obs, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, 
                    init_X=None, ode_method='rk4', ode_options=None, adjoint=True, adjoint_method='rk4', adjoint_options=None, save_filter_step=True,
                    smooth_lag=0, t0=0., var_inflation=None, localization_radius=None, compute_likelihood=True, linear_obs=True, time_varying_obs=False,
                    save_first=False, tqdm=None, **ode_kwargs):
    """
    EnKF with stochastic perturbation.

    Key args:
        ode_func (torch.nn.Module): Vector field f(t,x)
                Tip: Wrap all parameters of interest that you want to evaluate gradient by torch.nn.Parameter()
        obs_func (torch.nn.Module): Observation model h(x), assumed to be linear h(x) = Hx.
                If time varying_obs==True, can take a list of torch.nn.Module's
        t_obs (tensor): 1D-Tensor of shape (n_obs,). Time points where observations are available
                This does NOT need to be time-uniform. By default, t0 is NOT included.
        y_obs (tensor): Tensor of shape (n_obs, *bs, y_dim). Observed values at t_obs.
                '*bs' can be arbitrary batch dimension (or empty).
                Observations are assumed to have the same dimension 'y_dim'. However, observation model can be time-varying.
        N_ensem: Number of particles.
        init_m (tensor): Tensor of shape (x_dim, ). Mean of the initial distribution.
        init_C (noise.AddGaussian):  covariance of initial distribution
        model_Q_param (noise.AddGaussian): model error covariance
        noise_R_param (noise.AddGaussian): observation error covariance

    Optional args:
        init_X (tensor): Tensor of shape (*bs, N_ensem, x_dim). Initial ensemble if pre-specified.
        ode_method: Numerical scheme for forward equation. We use 'euler' or 'rk4'. Other solvers are available. See https://github.com/rtqichen/torchdiffeq
        ode_options: Set it to dict(step_size=...) for fixed step solvers. Adaptive solvers are also available - see the link above.
        adjoint (bool): Whether to compute gradient via adjoint equation or direct backpropagation through the solver.
        adjoint_method: Numerical scheme for adjoint equation if adjoint==True.
        ode_kwargs: additional kwargs for neuralODE.
        smooth_lag: Length of smoothing time interval.
                This is NOT needed for AD-EnKF, but might be needed for Expectation-Maximization.
        var_inflation: See discussion in paper. Typical value is between 1 and 1.1. None by default.
        localization_radius: See discussion in paper. Typical value is 5. None by default.

    Returns:
        X (tensor): Tensor of shape (*bs, N_ensem, x_dim). Final ensemble.
        X_track (tensor): Tensor of shape (n_obs, *bs, N_ensem, x_dim) if save_filter_step==True. Trajectories of ensemble across t_obs.
        log_likelihood (tensor): Log likelihood estimate # (*bs)
    """

    ode_integrator = odeint_adjoint if adjoint else odeint

    x_dim = init_m.shape[0]
    y_dim = y_obs.shape[-1]
    n_obs = y_obs.shape[0]
    bs = y_obs.shape[1:-1]  

    log_likelihood = torch.zeros(bs, device=device) if compute_likelihood else None  # (*bs),  tensor(0.) if no batch dimension

    if localization_radius is not None:
        taper = construct_Gaspari_Cohn(localization_radius, x_dim, device)

    if init_X is not None:
        X = init_X.detach()
    else:
        X = init_C_param(init_m.expand(*bs, N_ensem, x_dim))

    X_track = None
    if save_filter_step:
        X_track = torch.empty(n_obs+1, *bs, N_ensem, x_dim, dtype=init_m.dtype, device=device)
        X_track[0] = X.detach().clone()


    t_cur = t0

    pbar = tqdm(range(n_obs), leave=False) if tqdm is not None else range(n_obs)
    for j in pbar:
        ################ Forecast step ##################
        if adjoint:
            _, X = ode_integrator(ode_func, X, torch.tensor([t_cur, t_obs[j]], device=device), method=ode_method, options=ode_options, adjoint_method=adjoint_method, adjoint_options=adjoint_options, **ode_kwargs)
        else:
            _, X = ode_integrator(ode_func, X, torch.tensor([t_cur, t_obs[j]], device=device), method=ode_method, options=ode_options, **ode_kwargs)
        t_cur = t_obs[j]

        if model_Q_param is not None:
            X = model_Q_param(X)

        X_m = X.mean(dim=-2).unsqueeze(-2)  # (*bs, 1, x_dim)
        X_ct = X - X_m

        if var_inflation is not None:
            X = var_inflation * (X - X_m) + X_m

        ################ Analysis step ##################
        obs_func_j = obs_func[j] if time_varying_obs else obs_func
        y_obs_j = y_obs[j].unsqueeze(-2)  # (*bs, 1, y_dim)
        # Noise perturbation of observed data (key in stochastic EnKF)
        obs_perturb = noise_R_param(y_obs_j.expand(*bs, N_ensem, y_dim))
        noise_R = noise_R_param.full()

        C_uu = 1/(N_ensem-1) * X_ct.transpose(-1, -2) @ X_ct  # (*bs, x_dim, x_dim). Note: It can be made memory-efficient by not computing this explicity. See discussion in paper.

        if localization_radius is not None:
            C_uu = taper * C_uu

        if linear_obs:
            H = obs_func_j.H # (y_dim, x_dim)
        else: # Compute Jacobian of h evaluated at the ensemble mean
            H_vec = torch.autograd.functional.jacobian(obs_func_j, X_m.view(-1, x_dim), create_graph=False, strict=False, vectorize=True) # (bs_mul, y_dim, bs_mul, x_dim)
            H = H_vec.diagonal(dim1=0, dim2=2).permute(2, 0, 1).view(*bs, y_dim, x_dim) # (*bs, y_dim, x_dim)   see https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571
        HX = X @ H.transpose(-1, -2) # (*bs, N_ensem, y_dim)
        HX_m = X_m @ H.transpose(-1, -2) # (*bs, 1, y_dim)
        HC = H @ C_uu # (*bs, y_dim, x_dim)
        HCH_T = HC @ H.transpose(-1, -2) # (*bs, y_dim, y_dim)
        HCH_TR_chol = torch.linalg.cholesky(HCH_T + noise_R) # (*bs, y_dim, y_dim), lower-tril
        if compute_likelihood: 
            d = torch.distributions.MultivariateNormal(HX_m.squeeze(-2), scale_tril=HCH_TR_chol) # (*bs, y_dim) and (*bs, y_dim, y_dim)
            log_likelihood += d.log_prob(y_obs_j.squeeze(-2)) # (*bs)
        pre = (obs_perturb - HX) @ torch.cholesky_inverse(HCH_TR_chol) # (*bs, N_ensem, y_dim)
        X = X + pre @ HC # (*bs, N_ensem, x_dim)


        if save_filter_step and smooth_lag > 0:
            with torch.no_grad():
                t_start = torch.max(t_cur - smooth_lag, torch.tensor(t0))
                j_start = torch.argmax((t_obs>=t_start).type(torch.uint8)) # e.g., t_obs[5] = 0.5, t_obs[4]=0.4, smooth_lag=0.1 --> j_start = 4
                XS = X_track[j_start:j] # (L, *bs, N_ensem, x_dim)
                XS_m = XS.mean(dim=-2, keepdim=True) # (L, *bs, 1, x_dim)
                CSH_T = 1/(N_ensem-1) * (XS - XS_m).transpose(-1, -2) @ (HX - HX_m) # (L, *bs, x_dim, y_dim)
                XS = XS + pre @ CSH_T.transpose(-1, -2)
                X_track[j_start:j] = XS

        if save_filter_step:
            X_track[j+1] = X.detach()

    if not save_first:
        X_track = X_track[1:]
    return X, X_track, log_likelihood

