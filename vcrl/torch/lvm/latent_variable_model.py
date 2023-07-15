import abc
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import vcrl.torch.pytorch_util as ptu
from vcrl.policies.base import ExplorationPolicy
from vcrl.torch.core import torch_ify, elem_or_tuple_to_numpy
from vcrl.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)
from vcrl.torch.networks import Mlp, CNN
from vcrl.torch.networks.basic import MultiInputSequential
from vcrl.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)
from vcrl.torch.sac.policies.base import (
    TorchStochasticPolicy,
    PolicyFromDistributionGenerator,
    MakeDeterministic,
)


class LatentVariableModel(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
