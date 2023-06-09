'''
Based on SB3 implementation of TD3
'''


import os

import torch
import torch.nn.functional as F
from torch.optim import Adam

from mbrl.third_party.pytorch_sac_pranz24.td3_model import (
    DeterministicPolicy,
    QNetwork,
)
from mbrl.third_party.pytorch_sac_pranz24.utils import hard_update, soft_update

from mbrl.util.custom_utils import get_grad_norm

class TD3(object):
    def __init__(self, num_inputs, action_space, args):
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval

        self.device = args.device

        # critic
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(
            device=self.device
        )
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(
            num_inputs, action_space.shape[0], args.hidden_size
        ).to(self.device)
        hard_update(self.critic_target, self.critic)

        # policy
        self.policy = DeterministicPolicy(
            num_inputs, action_space.shape[0], args.hidden_size, action_space
        ).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, batched=False, evaluate=False):
        state = torch.FloatTensor(state)
        if not batched:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        if batched:
            return action.detach().cpu().numpy()
        return action.detach().cpu().numpy()[0]

    def update_parameters(
        self, memory, batch_size, updates, logger=None, reverse_mask=False
    ):
        # Sample a batch from memory and ignore truncateds
        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            mask_batch,
            _,
        ) = memory.sample(batch_size).astuple()

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        if reverse_mask:
            mask_batch = mask_batch.logical_not()

        with torch.no_grad():
            next_state_action, _, _ = self.policy.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, _, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = - min_qf_pi.mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if logger is not None:
            logger.log("train/batch_reward", reward_batch.mean(), updates)
            logger.log("train_critic/loss", qf_loss, updates)
            logger.log("train_actor/loss", policy_loss, updates)
            # custom
            logger.log('train/updates', updates, updates)
            logger.log('train/critic_grad_norm', get_grad_norm(self.critic), updates)
            logger.log('train/policy_grad_norm', get_grad_norm(self.policy), updates)
            logger.log('train_critic/qf1', qf1.mean(), updates)
            logger.log('train_critic/qf2', qf2.mean(), updates)
            logger.log('train_critic/qf1_next_target', qf1_next_target.mean(), updates)
            logger.log('train_critic/qf2_next_target', qf2_next_target.mean(), updates)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
        )

    # Save model parameters
    def save_checkpoint(self, env_name=None, suffix="", ckpt_path=None):
        if ckpt_path is None:
            assert env_name is not None
            if not os.path.exists("checkpoints/"):
                os.makedirs("checkpoints/")
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print("Saving models to {}".format(ckpt_path))
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            },
            ckpt_path,
        )

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
