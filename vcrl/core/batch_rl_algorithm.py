"""Modified:
* update obsnormalizer when we interact with expl_env
"""
import abc
import numpy as np

import gtimer as gt

import vcrl.torch.pytorch_util as ptu
from vcrl.core.rl_algorithm import BaseRLAlgorithm
from vcrl.data_management.replay_buffer import ReplayBuffer
from vcrl.samplers.data_collector import PathCollector
from vcrl.torch.data_management.normalizer import MeanStdNormalizer


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            start_epoch=0, # negative epochs are offline, positive epochs are online
            obs_normalizer=None,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = start_epoch
        self.obs_normalizer = obs_normalizer

    def train(self):
        """Negative epochs are offline, positive epochs are online"""
        for self.epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.offline_rl = self.epoch < 0
            self._begin_epoch(self.epoch)
            self._train()
            self._end_epoch(self.epoch)

    def _train(self):
        if self.epoch == 0 and self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            if not self.offline_rl:
                self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            if self.obs_normalizer is not None:
                update_obs_normalizer(
                    init_expl_paths,
                    self.obs_normalizer,
                    self.replay_buffer.observation_key,
                    self.replay_buffer.desired_goal_key
                )

        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
        gt.stamp('evaluation sampling')

        for _ in range(self.num_train_loops_per_epoch):
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
            )
            if self.obs_normalizer is not None:
                update_obs_normalizer(
                    new_expl_paths,
                    self.obs_normalizer,
                    self.replay_buffer.observation_key,
                    self.replay_buffer.desired_goal_key
                )
            gt.stamp('exploration sampling', unique=False)

            if not self.offline_rl:
                self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)

            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                self.trainer.train(train_data)
            gt.stamp('training', unique=False)
            self.training_mode(False)


def update_obs_normalizer(paths, obs_normalizer, observation_key, desired_goal_key):
    assert isinstance(obs_normalizer, MeanStdNormalizer)
    for path in paths:
        for obs in path['next_observations']:
            obs = np.hstack((obs[observation_key], obs[desired_goal_key]))[None]
            obs_normalizer.update(ptu.from_numpy(obs))