import numpy as np
import torch

class ContinuousMemory(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, *state_dim))
        self.action = np.zeros((max_size, *action_dim))
        self.next_state = np.zeros((max_size, *state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self):

        return (
            torch.tensor(self.state[:self.ptr]).to(self.device),
            torch.tensor(self.action[:self.ptr]).to(self.device),
            torch.tensor(self.next_state[:self.ptr]).to(self.device),
            torch.tensor(self.reward[:self.ptr]).to(self.device),
            torch.tensor(self.not_done[:self.ptr]).to(self.device)
        )

    def reset(self):
        self.ptr = 0

    def __len__(self):
        return self.ptr

if __name__ == '__main__':
    import gymnasium as gym
    env = gym.make('MountainCarContinuous-v0')
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    print(np.zeros((10,3)))
    '''
    num_episodes = int(3e3)
    max_timestamp = int(1e4)
    memory = Memory(state_dim, action_dim, num_episodes * max_timestamp)
    total_time_steps = 0
    is_done = False
    for _ in range(num_episodes):
        prev_state, info = env.reset()
        total_reward = 0
        for time_step in range(max_timestamp):
            action = env.action_space.sample()
            # observation, reward, terminated, truncated, info
            state, reward, done, truncated, info = env.step(action)
            memory.add(prev_state, action, state, reward, done)
            total_time_steps += 1
            # Train agent after collecting sufficient data
            prev_state = state
            total_reward += reward
            if done or truncated:
                is_done = True if done or is_done else is_done
                break

    states, actions, next_states, rewards, masks = memory.sample()
    print(len(states), total_time_steps, is_done)
    '''



