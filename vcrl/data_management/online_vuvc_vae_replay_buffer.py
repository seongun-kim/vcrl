"""Modified:
* remove parralleism
"""
import numpy as np
import torch

import vcrl.torch.pytorch_util as ptu
from vcrl.core.eval_util import create_stats_ordered_dict
from vcrl.data_management.online_vae_replay_buffer import OnlineVaeRelabelingBuffer


class OnlineVuvcVaeRelabelingBuffer(OnlineVaeRelabelingBuffer):
    def __init__(
            self,
            qfe,
            policy,
            fixed_reset_obs,
            disagreement_method='std',
            *args,
            **kwargs
    ):
        super(OnlineVuvcVaeRelabelingBuffer, self).__init__(*args, **kwargs)        

        self.qfe = qfe
        self.policy = policy
        self.fixed_reset_obs = fixed_reset_obs
        self.disagreement_method = disagreement_method

        self._disagreements = None
        self._vuvc_sample_probs = None

    def add_path(self, path):
        super().add_path(path)

    def get_diagnostics(self):
        if self._vae_sample_probs is None or self._vae_sample_priorities is None or self._vuvc_sample_probs is None:
            stats = create_stats_ordered_dict(
                'VAE Sample Weights',
                np.zeros(self._size),
            )
            stats.update(create_stats_ordered_dict(
                'VAE Sample Probs',
                np.zeros(self._size),
            ))
            stats.update(create_stats_ordered_dict(
                'VUVC Sample Probs',
                np.zeros(self._size),
            ))
            stats.update(create_stats_ordered_dict(
                'VUVC Disagreements',
                np.zeros(self._size),
            ))
        else:
            vae_sample_priorities = self._vae_sample_priorities[:self._size]
            vae_sample_probs = self._vae_sample_probs[:self._size]
            vuvc_sample_probs = self._vuvc_sample_probs[:self._size]
            vuvc_disagreements = self._disagreements[:self._size]
            stats = create_stats_ordered_dict(
                'VAE Sample Weights',
                vae_sample_priorities,
            )
            stats.update(create_stats_ordered_dict(
                'VAE Sample Probs',
                vae_sample_probs,
            ))
            stats.update(create_stats_ordered_dict(
                'VUVC Sample Probs',
                vuvc_sample_probs,
            ))
            stats.update(create_stats_ordered_dict(
                'VUVC Disagreement',
                vuvc_disagreements,
            ))
        return stats

    def refresh_latents(self, epoch):
        """
        update vae_sample_probs + disagreement
        """
        super().refresh_latents(epoch)

        self._disagreements = self.compute_disagreement(
            self.qfe,
            self.policy,
            self._next_obs[self.achieved_goal_key][:self._size],
            self.fixed_reset_obs[self.decoded_obs_key],
            self.disagreement_method
        )

        self._vuvc_sample_probs = self._vae_sample_probs * self._disagreements
        # self._vuvc_sample_probs = self._vae_sample_probs * np.power(self._disagreements, 4)
        self._vuvc_sample_probs = self._vuvc_sample_probs / np.sum(self._vuvc_sample_probs)

    def sample_weighted_indices(self, batch_size):
        if (
            self._prioritize_vae_samples and
            self._vuvc_sample_probs is not None and
            self.skew
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

    def compute_disagreement(
        self,
        qfe,
        policy,
        ags,
        reset_achieved_goal,
        disagreement_method='std',
    ):
        batch_size = 512
        next_idx = min(batch_size, ags.shape[0])
        cur_idx = 0
        disagreements = np.zeros(ags.shape[0])

        reset_achieved_goal = ptu.from_numpy(reset_achieved_goal).unsqueeze(dim=0)
        reset_mu, reset_logvar = self.vae.encode(reset_achieved_goal)

        while cur_idx < ags.shape[0]:
            idxs = np.arange(cur_idx, next_idx)
            batch_ags = ptu.from_numpy(ags[idxs])

            # batch_observations = ptu.from_numpy(
            #     np.tile(reset_achieved_goal, [batch_ags.shape[0], 1])
            # )
            batch_observations = reset_mu.repeat([batch_ags.shape[0], 1])

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
