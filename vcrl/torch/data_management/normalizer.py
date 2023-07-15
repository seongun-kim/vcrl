# Support MeanStd Normalizer
# modified from:
# https://github.com/spitis/mrl/blob/master/mrl/modules/normalizer.py
import torch
import torch.nn as nn
import vcrl.torch.pytorch_util as ptu
import numpy as np

from vcrl.data_management.normalizer import Normalizer, FixedNormalizer


class TorchNormalizer(Normalizer):
    """
    Update with np array, but de/normalize pytorch Tensors.
    """
    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self.synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = ptu.from_numpy(self.mean)
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            # Unsqueeze along the batch use automatic broadcasting
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        if not self.synchronized:
            self.synchronize()
        mean = ptu.from_numpy(self.mean)
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean + v * std


class TorchFixedNormalizer(FixedNormalizer):
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = ptu.from_numpy(self.mean)
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            # Unsqueeze along the batch use automatic broadcasting
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def normalize_scale(self, v):
        """
        Only normalize the scale. Do not subtract the mean.
        """
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            std = std.unsqueeze(0)
        return v / std

    def denormalize(self, v):
        mean = ptu.from_numpy(self.mean)
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean + v * std

    def denormalize_scale(self, v):
        """
        Only denormalize the scale. Do not add the mean.
        """
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            std = std.unsqueeze(0)
        return v * std


class MeanStdNormalizer(nn.Module):
    def __init__(self, size, epsilon=1e-4, clip_before=200.0, clip_after=5.0):
        super(MeanStdNormalizer, self).__init__()

        self.size = size
        self.epsilon = epsilon
        self.clip_before = clip_before
        self.clip_after = clip_after
        
        self.register_buffer('mean', torch.zeros(self.size))
        self.register_buffer('var', torch.ones(self.size) + self.epsilon)
        self.register_buffer('count', torch.zeros(1) + self.epsilon)
        
    @property
    def std(self):
        return self.var.sqrt()
        
    def update(self, x):
        assert len(x.shape) == 2
        batch_mean = x.mean(dim=0, keepdims=True)
        batch_var = x.var(dim=0, keepdims=True, unbiased=False)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.pow(delta, 2) * self.count * batch_count / (tot_count)
        new_var = M2 / (tot_count)

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def forward(self, x):
        return self.normalize(x)
    
    def normalize(self, x):
        x = torch.clamp(x, -self.clip_before, self.clip_before)
        x = (x - self.mean) / (self.std + self.epsilon)
        return torch.clamp(x, -self.clip_after, self.clip_after)

    def denormalize(self, x):
        return x * self.std + self.mean
        
    def get_diagnostics(self):
        return {}