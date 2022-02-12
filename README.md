# Auto-differentiable Ensemble Kalman Filters (AD-EnKF)

Joint learning of latent dynamics and states from noisy observations, by auto-differentiating through an Ensemble Kalman Filter (EnKF) using PyTorch.

Getting started:
- `l96_EnKF_demo.py`: Computation of parameter log-likelihood and gradient estimates with EnKF (Lorenz-96 model).
- `l96_param_est_demo.py`: Parameter estimation in Lorenz-96 model with AD-EnKF (cf. Section 5.2.1 of paper).
- `l96_NN_demo.py`: Learning Lorenz-96 dynamics and states with neural network and AD-EnKF (cf. Section 5.2.2 of paper).
- `l96_correction_demo.py`: Correcting imperfect Lorenz-96 model with neural network and AD-EnKF (cf. Section 5.2.3 of paper).
- `l96_multiscale_param_est.py`: Parameter estimation in multiscale Lorenz-96 model with AD-EnKF (working paper).
