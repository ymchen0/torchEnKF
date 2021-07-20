import numpy as np
import torch

def ess(weight):
  # (*bdims, weight) -> (*bdims)
  return 1 / (weight**2).sum(dim=-1)

def softplus(t):
  return torch.log(1. + torch.exp(t))

def softplus_inv(t):
  return torch.log(-1. + torch.exp(t))

def softplus_grad(t):
  return torch.exp(t) / (1. + torch.exp(t))