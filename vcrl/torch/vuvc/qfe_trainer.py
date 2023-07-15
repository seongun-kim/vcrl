import numpy as np
from collections import OrderedDict, namedtuple

import torch
import torch.optim as optim
from torch import nn as nn

import vcrl.torch.pytorch_util as ptu
from vcrl.core.loss import LossFunction
from vcrl.torch.torch_rl_algorithm import TorchTrainer


class QfeTrainer(TorchTrainer, LossFunction):
    def __init__(
        self,
        policy,
        qfe,
        target_qfe,
        *args,
        discount=0.99,
        reward_scale=1.0,
        qf_lr=1e-3,
        optimizer_class=optim.Adam,
        soft_target_tau=1e-2,
        target_update_period=1,
        **kwargs
    ):
        super(QfeTrainer, self).__init__()

        self.policy = policy 
        self.qfe = qfe
        self.target_qfe = target_qfe
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        
        self.qf_criterion = nn.MSELoss()

        self.optimizers = []
        for qf in self.qfe.qfs:
            self.optimizers.append(
                optimizer_class(
                    qf.parameters(),
                    lr=qf_lr,
                )
            )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train_from_torch(self, batch):
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        for loss, optimizer in zip(losses, self.optimizers):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        
    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        for qf, target_qf in zip(self.qfe.qfs, self.target_qfe.qfs):
            ptu.soft_update_from_to(
                qf, target_qf, self.soft_target_tau
            )

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # new_next_actions = self.policy(next_obs) # ddpg

        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)

        q_preds = self.qfe(obs, actions)
        target_q_values = self.target_qfe(next_obs, new_next_actions)

        qf_losses = []
        for q_pred, target_q_value in zip(q_preds, target_q_values):
            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_value
            qf_loss = self.qf_criterion(q_pred, q_target.detach())
            qf_losses.append(qf_loss)

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            for i, q_loss in enumerate(qf_losses):
                eval_statistics['QF{} Loss'.format(i + 1)] = np.mean(ptu.get_numpy(qf_loss))
            
        return qf_losses, eval_statistics

    @property
    def networks(self):
        return [
            self.qfe,
            self.target_qfe,
        ]

    def get_snapshot(self):
        return dict(
            qfe=self.qfe,
            target_qfe=self.target_qfe,
        )

    def get_diagnostics(self):
        return self.eval_statistics