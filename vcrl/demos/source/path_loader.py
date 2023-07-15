from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import vcrl.torch.pytorch_util as ptu
from vcrl.core.eval_util import create_stats_ordered_dict
from vcrl.torch.torch_rl_algorithm import TorchTrainer

from vcrl.util.io import load_local_or_remote_file

import random
from vcrl.torch.core import np_to_pytorch_batch
from vcrl.data_management.path_builder import PathBuilder

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from vcrl.core import logger

import glob

class PathLoader:
    """
    Loads demonstrations and/or off-policy data into a Trainer
    """

    def load_demos(self, ):
        pass
