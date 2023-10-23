import copy
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm


class DDPG(object):
    def __init__(self, state_dim, action_dim, actor, critic, discount, tau, actor_args={}, critic_args={},
                 actor_lr=3e-3, critic_lr=2e-2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = actor(state_dim, action_dim, **actor_args).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = critic(state_dim, action_dim, **critic_args).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.discount = discount
        self.tau = tau


    def select_action(self, state):
        """
        takes the current state as input and returns an action to take in that state.
        It uses the actor network to map the state to an action.
        """
        state = torch.FloatTensor(state.reshape(1, *state.shape)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size):
        # For each Sample in replay buffer batch
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss as the negative mean Q value using the critic network and the actor network
        actor_loss = -self.critic(state, self.actor(state)).mean()


        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def fit(self, env, epoch, max_time_steps, start_update, batch_size, noise, replay_buffer, action_valid_fn):
        weight_file = os.path.realpath(__file__). \
            replace(os.path.join('agents', 'ddpg.py'), os.path.join('agents', 'weights'))
        total_rewards = []
        for _ in tqdm(range(epoch)):
            prev_state, info = env.reset()
            total_reward = 0
            for time_step in range(max_time_steps):
                action = action_valid_fn(self.select_action(prev_state) + noise())
                # observation, reward, terminated, truncated, info
                state, reward, done, truncated, info = env.step(action)
                replay_buffer.add(prev_state, action, state, reward, done)
                # Train agent after collecting sufficient data
                if time_step >= start_update:
                    self.update(replay_buffer, batch_size)
                if time_step % 5 == 0:
                    self.save(weight_file)
                prev_state = state
                total_reward += reward
                if done or truncated:
                    break
            total_rewards.append(total_reward)
        return total_rewards

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "ddpg_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "ddpg_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "ddpg_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "ddpg_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "ddpg_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "ddpg_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "ddpg_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "ddpg_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)