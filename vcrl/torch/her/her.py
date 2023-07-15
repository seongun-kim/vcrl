import torch

from vcrl.torch.torch_rl_algorithm import TorchTrainer


class HERTrainer(TorchTrainer):
    def __init__(self, base_trainer: TorchTrainer):
        super().__init__()
        self._base_trainer = base_trainer

    def train_from_torch(self, data):
        obs = data['observations']
        next_obs = data['next_observations']
        goals = data['resampled_goals']
        data['observations'] = torch.cat((obs, goals), dim=1)
        data['next_observations'] = torch.cat((next_obs, goals), dim=1)
        self._base_trainer.train_from_torch(data)

    def get_diagnostics(self):
        return self._base_trainer.get_diagnostics()

    def end_epoch(self, epoch):
        self._base_trainer.end_epoch(epoch)

    @property
    def networks(self):
        return self._base_trainer.networks

    def get_snapshot(self):
        return self._base_trainer.get_snapshot()


class HERTrainerDXP(HERTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_from_torch(self, data):
        obs = data['observations']
        next_obs = data['next_observations']
        goals = data['resampled_goals']
        obs = self._postprocess_obs(obs, goals)
        next_obs = self._postprocess_obs(next_obs, goals)
        data['observations'] = torch.cat((obs, goals), dim=1)
        data['next_observations'] = torch.cat((next_obs, goals), dim=1)
        self._base_trainer.train_from_torch(data)

    def _postprocess_obs(self, obs, goals):
        scan_stack = obs[:, :-7]
        other = obs[:, -7:]
        prev_pose = other[:, :2]
        pose = other[:, 2:4]
        vel = other[:, 4:6]
        yaw = other[:, 6]

        goal_distance = self._compute_distance(pose, goals)
        goal_angle = torch.atan2(goals[:, 1] - pose[:, 1],
                                 goals[:, 0] - pose[:, 0])
        goal_angle = self._angle_correction(goal_angle)
        heading = self._angle_correction(goal_angle, yaw)

        return torch.cat((scan_stack, goal_distance, heading, vel), dim=1)

    def _compute_distance(self, achieved_goal, goal):
        return torch.norm(achieved_goal-goal, dim=1)

    def _angle_correction(self, angle):
        if angle > np.pi:
            angle -= 2 * np.pi 
        elif angle < -np.pi:
            angle += 2 * np.pi 
        return angle
