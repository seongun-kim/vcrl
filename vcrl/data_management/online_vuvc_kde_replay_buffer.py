import numpy as np

import torch

import vcrl.torch.pytorch_util as ptu
from vcrl.data_management.online_kde_replay_buffer import OnlineKdeRelabelingBuffer


class OnlineVuvcKdeRelabelingBuffer(OnlineKdeRelabelingBuffer):
    def __init__(
        self,
        qfe,
        policy,
        fixed_reset_obs,
        disagreement_method='std',
        *args,
        **kwargs
    ):
        super(OnlineVuvcKdeRelabelingBuffer, self).__init__(*args, **kwargs)

        self.qfe = qfe
        self.policy = policy
        self.fixed_reset_obs = fixed_reset_obs
        self.disagreement_method = disagreement_method

        self._disagreements = None
        self._vuvc_sample_probs = None

    def add_path(self, path):
        super().add_path(path)

    def get_diagnostics(self):
        return {}

    def refresh(self, epoch):
        """update kde_sample_probs + disagreement
        """
        super().refresh(epoch)

        self._disagreements = compute_disagreement(
            self.qfe,
            self.policy,
            self._next_obs[self.achieved_goal_key][:self._size],
            self.fixed_reset_obs[self.achieved_goal_key],
            self.disagreement_method
        )

        self._vuvc_sample_probs = self._kde_sample_probs * self._disagreements
        self._vuvc_sample_probs = self._vuvc_sample_probs / np.sum(self._vuvc_sample_probs)

    def sample_weighted_indices(self, batch_size):
        if (
            self._vuvc_sample_probs is not None
        ):
            indices = np.random.choice(
                len(self._vuvc_sample_probs),
                batch_size,
                p=self._vuvc_sample_probs,
            )
            assert (
                np.max(self._vuvc_sample_probs) <= 1 and
                np.min(self._vuvc_sample_probs) >= 0
            )
        else:
            indices = self._sample_indices(batch_size)
        return indices

    def sample_buffer_goals(self, batch_size):
        if self._size == 0:
            # desired_goals
            return self.env.wrapped_env.sample_goals(batch_size)
        weighted_idxs = self.sample_weighted_indices(batch_size)
        next_obs = self._next_obs[self.achieved_goal_key][weighted_idxs]
        return {
            self.desired_goal_key:  next_obs,
        }


def compute_disagreement(
    qfe,
    policy,
    ags, # np, achieved_goals
    reset_achieved_goal, # np dict
    disagreement_method='std',
):
    batch_size = 512
    next_idx = min(batch_size, ags.shape[0])
    cur_idx = 0
    disagreements = np.zeros(ags.shape[0])
    while cur_idx < ags.shape[0]:
        idxs = np.arange(cur_idx, next_idx)
        batch_ags = ptu.from_numpy(ags[idxs])

        batch_observations = ptu.from_numpy(
            np.tile(reset_achieved_goal, [batch_ags.shape[0], 1])
        )

        # compute q value from reset_observation to batch_ags
        batch_obs = torch.cat((
            batch_observations, batch_ags), dim=1
        )
        dist = policy(batch_obs)
        actions = dist.sample()
        # actions = policy(batch_obs) # ddpg
        q_preds = qfe(batch_obs, actions)
      
        if disagreement_method == 'std':
            disagreement = torch.std(
                torch.cat(q_preds, dim=1), dim=1
            )
        elif disagreement_method == 'var':
            disagreement = torch.var(
                torch.cat(q_preds, dim=1), dim=1
            )
        else:
            raise NotImplementedError
        disagreement = ptu.get_numpy(disagreement)
        disagreements[idxs] = disagreement

        cur_idx = next_idx
        next_idx += batch_size
        next_idx = min(next_idx, ags.shape[0])
    return disagreements
