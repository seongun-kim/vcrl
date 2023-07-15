import os
import argparse
import gym
import multiworld
import time
import torch

import multiworld.envs.mujoco as mwmj
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0
import vcrl.torch.pytorch_util as ptu
from vcrl.util.io import load_local_or_remote_file
from vcrl.launchers.launcher_util import setup_logger, set_seed
from vcrl.launchers.skewfit_experiments import generate_vae_dataset
from vcrl.pythonplusplus import identity
import vcrl.torch.vae.vae_schedules as vae_schedules
from vcrl.torch.vae.conv_vae import ConvVAE, imsize48_default_architecture
from vcrl.data_management.online_vae_replay_buffer import OnlineVaeRelabelingBuffer
from multiworld.core.image_env import ImageEnv
from vcrl.envs.vae_wrapper import VAEWrappedEnv
from vcrl.envs.edl_wrapper import EDLWrappedEnv
from vcrl.torch.networks import ConcatMlp
from vcrl.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from vcrl.torch.sac.sac import SACTrainer
from vcrl.torch.her.her import HERTrainer
from vcrl.torch.vae.vae_trainer import ConvVAETrainer
from vcrl.samplers.data_collector.vae_env import VAEWrappedEnvPathCollector
from vcrl.samplers.data_collector import GoalConditionedPathCollector
from vcrl.torch.skewfit.online_vae_algorithm import OnlineVaeAlgorithm
from vcrl.torch.edl.online_edl_vae_algorithm import OnlineEDLVaeAlgorithm


def train_vae(args):
    variant = dict(
        args=dict(
            env_id='SawyerDoorHookResetFreeEnv-v1',
            base_logdir='/data1/',
            log_dir='{}/vcrl_logs/{}/edl/{}/seed_{}',
            render=False,
            use_gpu=True,
            gpu_id=0,
            snapshot_mode='gap_and_last',
            snapshot_gap=20,
            seed=None,
            spec='vae',
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        vae_kwargs=dict(
            representation_size=16,
            decoder_output_activation=identity,
            decoder_distribution='gaussian_identity_variance',
            input_channels=3,
            architecture=imsize48_default_architecture,
        ),
        replay_buffer_kwargs=dict(
            start_skew_epoch=10,
            max_size=int(100000),
            fraction_goals_rollout_goals=1.,
            fraction_goals_env_goals=0.,
            exploration_rewards_type='None',
            vae_priority_type='vae_prob',
            priority_function_kwargs=dict(
                sampling_method='importance_sampling',
                decoder_distribution='gaussian_identity_variance',
                num_latents_to_sample=10,
            ),
            power=0.,
            relabeling_goal_sampling_mode='custom_goal_sampler',
        ),
        sac_trainer_kwargs=dict(
            reward_scale=1,
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        vae_trainer_kwargs=dict(
            beta=20,
            lr=1e-3,
        ),
        image_env_kwargs=dict(
            imsize=48,
            init_camera=sawyer_door_env_camera_v0,
            transpose=True,
            normalize=True,
            non_presampled_goal_img_is_garbage=True,
        ),
        vae_wrapped_env_kwargs=dict(
            sample_from_true_prior=True,
            reward_params=dict(
                type='latent_distance',
            ),
        ),
        algo_kwargs=dict(
            batch_size=1024,
            num_epochs=1000, # 170,
            num_eval_steps_per_epoch=500,
            num_expl_steps_per_train_loop=500,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=10000,
            vae_training_schedule=vae_schedules.custom_schedule,
            oracle_data=False,
            vae_save_period=50,
            parallel_vae_train=False,
            max_path_length=100,
        ),
        generate_vae_dataset_kwargs=dict(
            N=2,
            test_p=.9,
            use_cached=True,
            show=False,
            oracle_dataset=False,
            n_random_steps=1,
            env_id='SawyerDoorHookResetFreeEnv-v1',
            imsize=48,
            init_camera=sawyer_door_env_camera_v0,
        ),
    )

    # Overwrite arguments.
    for key, value in vars(args).items():
        if value is not None and value is not False:
            try:
                variant['args'][key] = value
            except:
                pass
    args = variant['args']
    if not args['spec']:
        log_dir = args['log_dir'].format(args['base_logdir'], args['env_id'], time.strftime('%Y_%m_%d_%H_%M_%S'), args['seed'])
    else:
        log_dir = args['log_dir'].format(args['base_logdir'], args['env_id'], args['spec'], args['seed'])

    setup_logger(
        variant=variant, 
        log_dir=log_dir,
        snapshot_mode=args['snapshot_mode'],
        snapshot_gap=args['snapshot_gap']
    )
    ptu.set_gpu_mode(args['use_gpu'], args['gpu_id'])

    if args['seed'] is None:
        args['seed'] = random.randint(0, 100000)
    set_seed(args['seed'])

    vae = ConvVAE(
        **variant['vae_kwargs']
    )
    vae.to(ptu.device)

    print(args['env_id'])
    multiworld.register_all_envs()
    env = gym.make(args['env_id'])

    render = args['render']
    presampled_goals = None

    image_env = ImageEnv(
        env,
        presampled_goals=presampled_goals,
        **variant.get('image_env_kwargs', {})
    )
    vae_env = VAEWrappedEnv(
        image_env,
        vae,
        imsize=image_env.imsize,
        decode_goals=render,
        render_goals=render,
        render_rollouts=render,
        presampled_goals=presampled_goals,
        **variant.get('vae_wrapped_env_kwargs', {})
    )
    env = vae_env
    env = EDLWrappedEnv(env, vae)

    observation_key = 'latent_observation'
    desired_goal_key = 'latent_desired_goal'
    achieved_goal_key = desired_goal_key.replace('desired', 'achieved')

    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
        + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size

    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=env.vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    train_data, test_data, info = generate_vae_dataset(variant['generate_vae_dataset_kwargs'])
    vae_trainer = ConvVAETrainer(
        train_data,
        test_data,
        model=env.vae,
        **variant['vae_trainer_kwargs']
    )
    max_path_length = variant['algo_kwargs']['max_path_length']
    expl_path_collector = GoalConditionedPathCollector(
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=None
    )
    eval_path_collector = GoalConditionedPathCollector(
        env,
        MakeDeterministic(policy),
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        goal_sampling_mode=None
    )

    algorithm = OnlineEDLVaeAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        vae=vae,
        vae_trainer=vae_trainer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

    torch.save(vae.state_dict(), os.path.join(log_dir, 'vae.pt'))


def train_policy(args):
    variant = dict(
        args=dict(
            env_id='SawyerDoorHookResetFreeEnv-v1',
            base_logdir='/data1/',
            log_dir = '/{}/vcrl_logs/{}/edl/{}/seed_{}', # base_logdir, env_id, spec, seed
            render=False,
            use_gpu=True,
            gpu_id=0,
            snapshot_mode='gap_and_last',
            snapshot_gap=20,
            seed=None,
            spec=None,
        ),
        edl_variant=dict(
            exploration_goal_sampling_mode='vae_prior', # 'edl_goal_sampler',
            evaluation_goal_sampling_mode='presampled',
            custom_goal_sampler='replay_buffer',
            presampled_goals_path=os.path.join(
                os.path.dirname(mwmj.__file__),
                "goals",
                "door_goals.npy",
            ),
            presample_goals=True,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        # go_explore_kwargs
        vae_kwargs=dict(
            representation_size=16,
            decoder_output_activation=identity,
            decoder_distribution='gaussian_identity_variance',
            input_channels=3,
            architecture=imsize48_default_architecture,
        ),
        replay_buffer_kwargs=dict(
            start_skew_epoch=10,
            max_size=int(100000),
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
            exploration_rewards_type='None',
            vae_priority_type='vae_prob',
            priority_function_kwargs=dict(
                sampling_method='importance_sampling',
                decoder_distribution='gaussian_identity_variance',
                num_latents_to_sample=10,
            ),
            power=-0.5,
            relabeling_goal_sampling_mode='custom_goal_sampler',
        ),
        sac_trainer_kwargs=dict(
            reward_scale=1,
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        vae_trainer_kwargs=dict(
            beta=20,
            lr=1e-3,
        ),
        image_env_kwargs=dict(
            imsize=48,
            init_camera=sawyer_door_env_camera_v0,
            transpose=True,
            normalize=True,
            non_presampled_goal_img_is_garbage=True,
        ),
        vae_wrapped_env_kwargs=dict(
            sample_from_true_prior=True,
            reward_params=dict(
                type='latent_distance',
            ),
        ),
        algo_kwargs=dict(
            batch_size=1024,
            num_epochs=170,
            num_eval_steps_per_epoch=500,
            num_expl_steps_per_train_loop=500,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=10000,
            vae_training_schedule=vae_schedules.edl_schedule,
            oracle_data=False,
            vae_save_period=50,
            parallel_vae_train=False,
            max_path_length=100,
        ),
        generate_vae_dataset_kwargs=dict(
            N=2,
            test_p=.9,
            use_cached=True,
            show=False,
            oracle_dataset=False,
            n_random_steps=1,

            env_id='SawyerDoorHookResetFreeEnv-v1',
            imsize=48,
            init_camera=sawyer_door_env_camera_v0,
        ),
    )

    # Overwrite arguments.
    for key, value in vars(args).items():
        if value is not None and value is not False:
            try:
                variant['args'][key] = value
            except:
                pass
    args = variant['args']
    if not args['spec']:
        log_dir = args['log_dir'].format(args['base_logdir'], args['env_id'], time.strftime('%Y_%m_%d_%H_%M_%S'), args['seed'])
    else:
        log_dir = args['log_dir'].format(args['base_logdir'], args['env_id'], args['spec'], args['seed'])

    setup_logger(
        variant=variant,
        snapshot_mode=args['snapshot_mode'],
        snapshot_gap=args['snapshot_gap'],
        log_dir=log_dir,
    )
    ptu.set_gpu_mode(args['use_gpu'], args['gpu_id'])

    if args['seed'] is None:
        args['seed'] = random.randint(0, 100000)
    set_seed(args['seed'])

    vae = ConvVAE(
        **variant['vae_kwargs']
    )
    vae.to(ptu.device)
    vae_dir = args['log_dir'].format(args['base_logdir'], args['env_id'], 'vae', 0)
    assert os.path.exists(vae_dir), 'Pre-trained VAE model doesn\'t exist.'
    vae.load_state_dict(torch.load(os.path.join(vae_dir, 'vae.pt')))

    multiworld.register_all_envs()
    env = gym.make(args['env_id'])

    render = args['render']
    presample_goals = variant['edl_variant'].get('presample_goals', False)
    presampled_goals_path = variant['edl_variant'].get('presampled_goals_path', None)
    presampled_goals = load_local_or_remote_file(presampled_goals_path).item() if presample_goals else None

    image_env = ImageEnv(
        env,
        presampled_goals=presampled_goals,
        **variant.get('image_env_kwargs', {})
    )
    vae_env = VAEWrappedEnv(
        image_env,
        vae,
        imsize=image_env.imsize,
        decode_goals=render,
        render_goals=render,
        render_rollouts=render,
        presampled_goals=presampled_goals,
        **variant.get('vae_wrapped_env_kwargs', {})
    )

    env = vae_env

    observation_key = 'latent_observation'
    desired_goal_key = 'latent_desired_goal'
    achieved_goal_key = desired_goal_key.replace('desired', 'achieved')
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=env.vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    train_data, test_data, info = generate_vae_dataset(variant['generate_vae_dataset_kwargs'])
    vae_trainer = ConvVAETrainer(
        train_data,
        test_data,
        model=env.vae,
        **variant['vae_trainer_kwargs']
    )
    max_path_length = variant['algo_kwargs']['max_path_length']
    expl_path_collector = VAEWrappedEnvPathCollector(
        variant['edl_variant']['exploration_goal_sampling_mode'],
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    eval_path_collector = VAEWrappedEnvPathCollector(
        variant['edl_variant']['evaluation_goal_sampling_mode'],
        env,
        MakeDeterministic(policy),
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )

    if variant['edl_variant']['custom_goal_sampler'] == 'replay_buffer':
        env.custom_goal_sampler = replay_buffer.sample_buffer_goals

    algorithm = OnlineVaeAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        vae=vae,
        vae_trainer=vae_trainer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_logdir', type=str, help='base subdirectory of log_dir')
    parser.add_argument('--mode', choices=['train_vae', 'train_policy'], help='Choose whether to generate vae dataset, train vae, or both.')
    parser.add_argument('--env_id', default='SawyerDoorHookResetFreeEnv-v1', type=str, help='')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int)
    parser.add_argument('--snapshot_mode', type=str, choices=['none', 'gap', 'last', 'gap_and_last'])
    parser.add_argument('--snapshot_gap', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--spec', type=str)
    args = parser.parse_args()

    if args.mode == 'train_vae':
        train_vae(args)
    elif args.mode == 'train_policy':
        train_policy(args)