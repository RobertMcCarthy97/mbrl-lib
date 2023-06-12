# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import omegaconf
import torch

import mbrl.algorithms.mbpo_custom as mbpo_custom
import mbrl.env

from llm_curriculum_algo.env_wrappers import make_env as make_env_inner
import wandb

class CustomRewardFn:
    def __init__(self, env):
        self.env = env

    def custom_reward_fn(self, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        assert len(next_obs.shape) == len(act.shape) == 2

        rewards = torch.zeros(act.shape[0], 1)
        for i in range(next_obs.shape[0]):
            reward = self.env.agent_conductor.get_active_task_reward(next_obs[i].numpy(force=True))
            rewards[i] = reward

        return torch.tensor(rewards)

def setup_wandb(cfg):
    wandb.init(
        entity='robertmccarthy11',
        project='llm-curriculum',
        group='temp',
        name='temp',
        job_type='training',
        # config=vargs,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
        )

def temp_env_test(env, reward_fn):
    obs, _ = env.reset()
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        reward_fn_val = reward_fn(torch.tensor(action).unsqueeze(0), torch.tensor(obs).unsqueeze(0))
        print(f"reward: {reward}, reward_fn: {reward_fn_val}")
        print(f"i={i}: terminated: {terminated}, truncated: {truncated}")
        input()
    obs, _ = env.reset()
    for i in range(50):
        obs, reward, terminated, truncated, info = env.step(env.get_oracle_action(obs))
        reward_fn_val = reward_fn(torch.tensor(action).unsqueeze(0), torch.tensor(obs).unsqueeze(0))
        print(f"reward: {reward}, reward_fn: {reward_fn_val}")
        print(f"i={i}: terminated: {terminated}, truncated: {truncated}")
        input()
    import pdb; pdb.set_trace()

def make_env(cfg):
    # TODO: get hparams from cfg?
    env = make_env_inner(
            manual_decompose_p=1,
            dense_rew_lowest=False,
            dense_rew_tasks=['move_gripper_to_cube', 'move_cube_towards_target_grasp'], #
            use_language_goals=False,
            render_mode='rgb_array',
            max_ep_len=50,
            single_task_names=['move_gripper_to_cube'], #
            high_level_task_names=['move_cube_to_target'],
            contained_sequence=False,
            state_obs_only=True,
            curriculum_manager_cls=None,
            old_gym=False,
            ) # TODO: add args to config
    
    assert len(env.agent_conductor.single_task_names) == 1, "Only setup for 1 task currently"
    term_fn = mbrl.env.termination_fns.no_termination
    if cfg.algorithm.learned_rewards:
        reward_fn = None
    else:
        reward_fn = CustomRewardFn(env).custom_reward_fn
    # temp_env_test(env, reward_fn)
    return env, term_fn, reward_fn

@hydra.main(config_path="conf", config_name="main_custom")
def run(cfg: omegaconf.DictConfig):
    if cfg.use_wandb:
        setup_wandb(cfg)
    env, term_fn, reward_fn = make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    test_env, *_ = make_env(cfg)
    return mbpo_custom.train(env, test_env, term_fn, cfg, reward_fn=reward_fn)


if __name__ == "__main__":
    run()