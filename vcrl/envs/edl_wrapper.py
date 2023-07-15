import gym
import random
import numpy as np
from collections import OrderedDict
import vcrl.torch.pytorch_util as ptu
from gym.spaces import Box, Dict
from multiworld.core.image_env import normalize_image
from vcrl.torch.vae.vae_trainer import (
    compute_log_p_log_q_log_d
)


class EDLWrappedEnv(object):
    """
        exploration policy requires env which 
        * does not have goal
        * has exploration reward (density-based)
    """
    def __init__(self, wrapped_env, vae):
        self.wrapped_env = wrapped_env
        self.vae = vae
        self.representation_size = self.vae.representation_size
        empty_space = Box(-np.inf, np.inf, shape=(0,), dtype=np.float32)
        spaces = self.wrapped_env.observation_space.spaces
        spaces['desired_goal'] = empty_space
        spaces['image_desired_goal'] = empty_space
        spaces['latent_desired_goal'] = empty_space
        self.observation_space = Dict(spaces)
        self.action_space = self.wrapped_env.action_space

    def get_diagnostics(self, paths, **kwargs):
        return self.wrapped_env.get_diagnostics(paths, **kwargs)

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs)[0]

    def compute_rewards(self, actions, obs):
        # assert 'image_achieved_goal' in obs.keys(), 'Observation/goal key should include image in the script code'
        normalized_img = normalize_image(obs['image_achieved_goal'])
        log_p, log_q, log_d = compute_log_p_log_q_log_d(
            self.vae,
            normalized_img,
            decoder_distribution='gaussian_identity_variance',
            num_latents_to_sample=10,
            sampling_method='importance_sampling'        
        )
        log_p_x = (log_p - log_q + log_d).mean(dim=1)
        log_p_x = ptu.get_numpy(log_p_x)
        exploration_reward = -log_p_x
        return exploration_reward

    def step(self, action):
        obs, _, done, info = self.wrapped_env.step(action)
        obs['desired_goal'] = np.zeros(0)
        obs['image_desired_goal'] = np.zeros(0)
        obs['latent_desired_goal'] = np.zeros(0)
        exploration_reward = self.compute_reward(
            action, obs
        )
        return obs, exploration_reward, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        obs['desired_goal'] = np.zeros(0)
        obs['image_desired_goal'] = np.zeros(0)
        obs['latent_desired_goal'] = np.zeros(0)
        return obs

    def sample_goals(self, batch_size):
        return {'desired_goal': np.zeros([batch_size, 0]),
                'image_desired_goal': np.zeros([batch_size, 0]),
                'latent_desired_goal': np.zeros([batch_size, 0])}

    def _encode(self, imgs):
        if imgs.size == 0:
            # If encoding image observation...
            return imgs
        else:
            # If encoding image desired goal...
            return self.wrapped_env._encode(imgs)

    def _decode(self, latents):
        # Empty desired goals are expected, so return empty decoded desired goals.
        assert latents.size == 0, 'Desired goal should be empty.'
        return latents