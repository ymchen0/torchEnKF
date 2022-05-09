import torch
import math
from torchEnKF import misc
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
from torch.nn.functional import normalize

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

def power_iter(A, n_iter=1):
    device = A.device
    u_shape = A[...,0:1].shape  # (*bs, N_ensem, 1)
    v_shape = A[...,0:1,:].shape  # (*bs, 1, y_dim)
    u = normalize(A.new_empty(u_shape).normal_(0, 1), dim=-2)  # (*bs, N_ensem, 1)
    v = normalize(A.new_empty(v_shape).normal_(0, 1), dim=-1)  # (*bs, 1, y_dim)
    for i in range(n_iter):
        v = normalize(A.transpose(-1, -2) @ u, dim=-2) # (*bs, y_dim, 1)
        u = normalize(A @ v, dim=-2) # (*bs, N_ensem, 1)
        sigma = u.transpose(-1, -2) @ A @ v
        # A = A / sigma
        v = v.transpose(-1,-2) # (*bs, 1, y_dim)
    return sigma

def inv_logdet(v, Y_ct, R, R_inv, logdet_R):
    # Returns matrix-vector product (Y Y^T + R)^{-1} times v for matrix Y=Y_ct and any choice of vector/matrix v. Also returns the log-determinant of (Y Y^T + R).
    # Supports batch operation
    # Y_ct: (*bs, N_ensem, y_dim), R: (y_dim, y_dim), v: (*bs, bs2, y_dim)
    # out (invv): (*bs, bs2, y_dim)
    device = Y_ct.device
    N_ensem = Y_ct.shape[-2]
    y_dim = Y_ct.shape[-1]
    if N_ensem >= y_dim:
        YYT_R = Y_ct.transpose(-1, -2) @ Y_ct + R
        YYT_R_chol = torch.linalg.cholesky(YYT_R)
        logdet = 2 * YYT_R_chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        invv = torch.cholesky_solve(v.transpose(-1,-2), YYT_R_chol).transpose(-1,-2)
    else:
        YTRinv = Y_ct @ R_inv  # (*bs, N_ensem, y_dim)
        YTRinvv = YTRinv @ v.transpose(-1, -2)  # (*bs, N_ensem, bs2)
        I_YTRinvY = torch.eye(N_ensem, device=device) + YTRinv @ Y_ct.transpose(-1, -2)  # (*bs, N_ensem, N_ensem)
        sc = power_iter(I_YTRinvY,n_iter=1)
        I_YTRinvY_sc = I_YTRinvY/sc
        I_YTRinvY_chol_sc = torch.linalg.cholesky(I_YTRinvY_sc)  # (*bs, N_ensem, N_ensem)
        I_YTRinvY_inv_YTRinvv = 1/sc * torch.cholesky_solve(YTRinvv, I_YTRinvY_chol_sc)   # (*bs, N_ensem, bs2)
        invv = v @ R_inv - I_YTRinvY_inv_YTRinvv.transpose(-1,-2) @ YTRinv  # (*bs, bs2, y_dim)
        logdet = y_dim * torch.log(sc).squeeze(-1).squeeze(-1) + 2 * I_YTRinvY_chol_sc.diagonal(dim1=-2, dim2=-1).log().sum(-1) + logdet_R
    return invv, logdet  # (*bs, bs2, y_dim), (*bs)

def EnKF(ode_func, obs_func, t_obs, y_obs, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, 
                    init_X=None, ode_method='rk4', ode_options=None, adjoint=True, adjoint_method='rk4', adjoint_options=None, save_filter_step={'mean'},
                    smooth_lag=0, t0=0., var_inflation=None, localization_radius=None, compute_likelihood=True, linear_obs=True, time_varying_obs=False,
                    save_first=False, tqdm=None, **ode_kwargs):
    """
    EnKF with stochastic perturbation.

    Key args:
        ode_func (torch.nn.Module): Vector field f(t,x)
                Tip: Wrap all parameters of interest that you want to evaluate gradient by torch.nn.Parameter()
                NOTE: This implicitly assume the underlying latent model is an ODE. For generic type of latent evolutions x_{t+1}=F(x_t), slight modifications of the forcast step are required.
        obs_func (torch.nn.Module): Observation model h(x), assumed to be linear h(x) = Hx.
                If time varying_obs==True, can take a list of torch.nn.Module's
        t_obs (tensor): 1D-Tensor of shape (n_obs,). Time points where observations are available.
                This does NOT need to be time-uniform. By default, t0 is NOT included. Must be monotonic increasing.
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
        ode_options: Set it to dict(step_size=...) for fixed step solvers for the forward equation. Adaptive solvers are also available - see the link above.
        adjoint (bool): Whether to compute gradient via adjoint equation or direct backpropagation through the solver.
        adjoint_method: Numerical scheme for adjoint equation if adjoint==True.
        adjoint options: Set it to dict(step_size=...) for fixed step solvers for the adjoint equation. Adaptive solvers are also available - see the link above.
        ode_kwargs: additional kwargs for neuralODE.
        save_filter_step:
            If contains 'mean', then particle means will be saved.
            If contains 'particles', then all particles will be saved.
            (Note: the up-to-date/final particles will always be returned seperately)
        t0: The timestamp at which the ensemble is initialized.
            By default, we DO NOT assume observation is available at t0. Slight modifications of the code are needed to handle this situation.
        var_inflation: See discussion in paper. Typical value is between 1 and 1.1. None by default.
        localization_radius: See discussion in paper. Typical value is 5. None by default.
        compute_likelihood: Whether to compute data log-likelihood in the filtering process.
                            Must be set to True for AD-EnKF.
        linear_obs: If set to True, then obs_func must be 'nn_templates.Linear' class. The observation model is y = Hx + noise where H is a matrix.
                    If set to False, then obs_func can be any differentiable function/module in PyTorch. The observation model is y = obs_func(x) + noise
        time_varying_obs: If set to False, the observation model is time-invariant. A single nn.Module/function is sufficient for the obs_func argument.
                        If set to True, the observation model can be different across time. A list of nn.Module/functions is needed for obs_func argument and has the same length as t_obs.
        save_first: Set it to True to save the initial ensemble.
        tqdm: Set tqdm=tqdm to use the tqdm format for presenting.

    Returns:
        X (tensor): Tensor of shape (*bs, N_ensem, x_dim). Final ensemble.
        res (dict):
            If save_filter_step contains 'mean', then res['mean'] will be tensor of shape (n_obs, *bs, x_dim)
            If save_filter_step contains 'particles', then res['particles'] will be tensor of shape (n_obs, *bs, N_ensem, x_dim)
        log_likelihood (tensor): Log likelihood estimate # (*bs)
    """

    ode_integrator = odeint_adjoint if adjoint else odeint

    x_dim = init_m.shape[0]
    y_dim = y_obs.shape[-1]
    n_obs = y_obs.shape[0]
    bs = y_obs.shape[1:-1]  

    if ode_options is None:
        if n_obs > 0:
            step_size = (t_obs[1:] - t_obs[:-1]).min()  # This computes the minimum length of time intervals in t_obs. However it's more preferred to manually provide a quantity for the step_size to avoid issues like non-divisibility.
            ode_options = dict(step_size=step_size)

    log_likelihood = torch.zeros(bs, device=device) if compute_likelihood else None  # (*bs),  tensor(0.) if no batch dimension

    if linear_obs and localization_radius is not None:
        taper = construct_Gaspari_Cohn(localization_radius, x_dim, device)

    if init_X is not None:
        X = init_X.detach()
    else:
        X = init_C_param(init_m.expand(*bs, N_ensem, x_dim))


    res = {}
    if 'particles' in save_filter_step:
        res['particles'] = torch.empty(n_obs + 1, *bs, N_ensem, x_dim, dtype=init_m.dtype, device=device)
        res['particles'][0] = X
    if 'mean' in save_filter_step:
        X_m = X.mean(dim=-2)
        res['mean'] = torch.empty(n_obs + 1, *bs, x_dim, dtype=init_m.dtype, device=device)
        res['mean'][0] = X_m.detach()


    step_size = ode_options['step_size']

    t_cur = t0

    pbar = tqdm(range(n_obs), desc="Running EnKF", leave=False) if tqdm is not None else range(n_obs)
    for j in pbar:
        ################ Forecast step ##################
        n_intermediate_j = round(((t_obs[j] - t_cur) / step_size).item())
        if adjoint:
            X = ode_integrator(ode_func, X, torch.linspace(t_cur, t_obs[j], n_intermediate_j + 1, device=device), method=ode_method, adjoint_method=adjoint_method, adjoint_options=adjoint_options, **ode_kwargs)[-1]
        else:
            X = ode_integrator(ode_func, X, torch.linspace(t_cur, t_obs[j], n_intermediate_j + 1, device=device), method=ode_method,  **ode_kwargs)[-1]
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
        noise_R_inv = noise_R_param.inv()
        logdet_noise_R = noise_R_param.logdet()



        if linear_obs and localization_radius is not None:
            H = obs_func_j.H  # (y_dim, x_dim)
            C_uu = 1 / (N_ensem - 1) * X_ct.transpose(-1,-2) @ X_ct  # (*bs, x_dim, x_dim). Note: It can be made memory-efficient by not computing this explicity. See discussion in paper.
            C_uu = taper * C_uu
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
        else:
            HX = obs_func_j(X)  # (*bs, N_ensem, y_dim)
            HX_m = HX.mean(dim=-2).unsqueeze(-2)  # (*bs, 1, y_dim)
            HX_ct = HX - HX_m
            C_ww_sqrt = 1/math.sqrt(N_ensem-1) * HX_ct  # (*bs, N_ensem, y_dim)
            v1 = obs_perturb - HX  # (*bs, N_ensem, y_dim)
            v2 = y_obs_j - HX_m  # (*bs, 1, y_dim)
            v = torch.cat((v1, v2), dim=-2)  # (*bs, N_ensem+1, y_dim)
            C_ww_R_invv, C_ww_R_logdet = inv_logdet(v, C_ww_sqrt, noise_R, noise_R_inv, logdet_noise_R)  # (*bs, N_ensem+1, y_dim), (*bs)
            pre = C_ww_R_invv[..., :N_ensem, :]  # (*bs, N_ensem, y_dim)  # (*bs, N_ensem, y_dim)
            if compute_likelihood:
                part1 = -1 / 2 * (y_dim * math.log(2 * math.pi) + C_ww_R_logdet)  # (1,)
                part2 = -1 / 2 * C_ww_R_invv[..., N_ensem:, :] @ (y_obs_j - HX_m).transpose(-1, -2)  # (*bs, 1, 1,)
                log_likelihood += (part1 + part2.squeeze(-1).squeeze(-1))  # (*bs)
            X = X + 1 / math.sqrt(N_ensem - 1) * (pre @ C_ww_sqrt.transpose(-1, -2)) @ X_ct  # (*bs, N_ensem, x_dim)

        if 'particles' in save_filter_step:
            res['particles'][j+1] = X
        if 'mean' in save_filter_step:
            X_m = X.mean(dim=-2)
            res['mean'][j+1] = X_m.detach()

    if not save_first:
        for key in res.keys():
            res[key] = res[key][1:]
    return X, res, log_likelihood

