"""
modified from:
# https://github.com/spitis/mrl/blob/master/mrl/utils/networks.py
# https://github.com/ShangtongZhang/DeepRL/tree/master/deep_rl/network
"""

import torch
import torch.nn as nn

from vcrl.policies.base import Policy
from vcrl.torch.core import PyTorchModule, eval_np


def orthogonal_init(layer):
    nn.init.orthogonal_(layer.weight.data)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class Mlp(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        hidden_activation=nn.GELU,
        output_activation=nn.Identity,
        hidden_init=orthogonal_init,
        output_init=orthogonal_init,
        layer_norm=nn.LayerNorm, # nn.Identity
    ):
        super(Mlp, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_sizes 
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_init = hidden_init
        self.output_init = output_init
        self.layer_norm = layer_norm

        layers = []
        dim_in = input_size
        for dim_out in hidden_sizes:
            layers += [
                hidden_init(nn.Linear(dim_in, dim_out)),
                layer_norm(dim_out), 
                hidden_activation(),
            ]
            dim_in = dim_out
        layers += [
            output_init(nn.Linear(dim_in, output_size)),
            output_activation()
        ]

        self.layers = nn.Sequential(*layers)
        
    def forward(self, input):
        return self.layers(input)


class ConcatMlp(Mlp):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)


class MlpQf(ConcatMlp):
    def __init__(
            self,
            *args,
            obs_normalizer=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, actions, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, actions, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=nn.Tanh, **kwargs)