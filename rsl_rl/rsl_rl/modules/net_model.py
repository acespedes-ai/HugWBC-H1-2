import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as td
from torch import autograd
from torch.distributions import Normal, Categorical
from rsl_rl.utils import unpad_trajectories
from isaacgym.torch_utils import *

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
    
def MLP(input_dim, output_dim, hidden_dims, activation, output_activation=None):
    activation = get_activation(activation)  
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    layers.append(activation)
    for l in range(len(hidden_dims)):
        if l == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[l], output_dim))
            if output_activation is not None:
                output_activation = get_activation(output_activation)
                layers.append(output_activation)
        else:
            layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
            layers.append(activation)

    return layers

class BaseAdaptModel(nn.Module):
    def __init__(self,
                 act_dim,
                 proprioception_dim,
                 cmd_dim,
                 privileged_dim,
                 terrain_dim,
                 latent_dim,
                 privileged_recon_dim,
                 actor_hidden_dims,
                 activation,
                 output_activation):

        super().__init__()
        self.is_recurrent = False
        self.act_dim = act_dim
        self.proprioception_dim = proprioception_dim
        self.cmd_dim = cmd_dim
        self.terrain_dim = terrain_dim
        self.privileged_dim = privileged_dim

        self.privileged_recon_loss = 0
        self.privileged_recon_dim = privileged_recon_dim
        self.z = 0

        self.state_estimator = nn.Sequential(*MLP(latent_dim, self.privileged_recon_dim, [64, 32], activation))        
        self.low_level_net = nn.Sequential(*MLP(proprioception_dim + latent_dim + self.cmd_dim + privileged_recon_dim, act_dim,
                                                actor_hidden_dims, activation, output_activation))
        
    def forward(self, x, privileged_obs=None, env_mask=None, sync_update = False, **kwargs):
        pro_obs_seq = x[..., :self.proprioception_dim]
        cmd = x[..., -1, self.proprioception_dim:self.proprioception_dim+self.cmd_dim]

        mem = self.memory_encoder(pro_obs_seq, **kwargs)
        privileged_pred_now = self.state_estimator(mem)
        jp_out = self.low_level_net(torch.cat((mem, privileged_pred_now, x[..., -1, :self.proprioception_dim], cmd), dim=-1))

        if sync_update:
            privileged_now = privileged_obs[..., self.proprioception_dim+self.cmd_dim:
                            self.proprioception_dim+self.cmd_dim+self.privileged_recon_dim]

            self.privileged_recon_loss = 2 * (privileged_pred_now - privileged_now.detach()).pow(2).mean()                         

        action = jp_out

        return action

    def memory_encoder(self, pro_obs_seq, **kwargs):
        raise NotImplementedError

    def compute_adaptation_pred_loss(self, metrics):
        if self.privileged_recon_loss != 0:
            metrics['privileged_recon_loss'] += self.privileged_recon_loss.item()
        return self.privileged_recon_loss


class SimpleMlpModel(nn.Module):
    """Plain MLP actor — no history, no teacher-student. Used for walk comparison tasks."""
    is_recurrent = False

    def __init__(self, obs_dim, act_dim,
                 hidden_dims=(512, 256, 128),
                 activation='elu',
                 output_activation=None,
                 **kwargs):
        super().__init__()
        self.is_recurrent = False
        self.privileged_recon_loss = 0
        layers = MLP(obs_dim, act_dim, list(hidden_dims), activation, output_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x)

    def compute_adaptation_pred_loss(self, metrics):
        return 0

class MlpAdaptModel(BaseAdaptModel):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 proprioception_dim,
                 cmd_dim,
                 privileged_dim,
                 terrain_dim,
                 latent_dim = 64,
                 privileged_recon_dim=3,
                 actor_hidden_dims=[256, 128, 64],
                 activation='elu',
                 output_activation=None,
                 max_length=10,
                 mlp_hidden_dims=[128, 64],
                 **kwargs):

        super().__init__(act_dim, proprioception_dim, cmd_dim, privileged_dim,
                         terrain_dim, latent_dim, privileged_recon_dim, actor_hidden_dims, 
                         activation, output_activation)

        self.max_length = max_length
        self.short_length = max_length
        self.mem_encoder = nn.Sequential(*MLP(proprioception_dim * self.short_length, latent_dim, mlp_hidden_dims, activation))

    def memory_encoder(self, pro_obs_seq, **kwargs):
        short_term_mem = pro_obs_seq[..., -self.short_length:, :self.proprioception_dim].flatten(-2, -1)
        return self.mem_encoder(short_term_mem)



    



