import gymnasium as gym
import numpy as np
from metric.finance_indicator import calculate_transaction_remainder_factor

class TradingEnv(gym.Env):

    def __init__(self, global_data, commission_rate=0.25, episode_length=128, window_size=50):
        assert len(global_data) >= episode_length + window_size, 'The size of historical data must be at least ' \
                                                                 'episode_length + window_size '
        # a tensor of size (feature, coin, time)
        self.global_data = global_data.values # include cash
        self.relative_price = self.global_data[1:, :, 0] / self.global_data[:-1, :, 0]
        # total number of periods in one episode
        self.episode_length = episode_length
        self.window_size = window_size
        self.commission_rate = commission_rate
        self.t0 = 0
        self.t = 0
        self.portfolio_value = 1
        self.PVM = []
        # openai gym attributes
        # action will be the portfolio weights from 0 to 1 for each asset
        self.action_space = gym.spaces.Box(
            0, 1, shape=(len(global_data['coin']),), dtype=np.float32)  # include cash

        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(window_size, len(global_data['coin']), len(global_data['feature']) + 1),
                                                dtype=np.float32)
        self.init_time_rng = np.random.default_rng(42)

    def get_last_weight(self):
        return self.PVM[-1]

    def get_portfolio_value(self):
        return self.portfolio_value

    def get_timestamp(self):
        return self.t

    def get_state(self, prices):
        state = prices / prices[-1, :, :]
        weights = np.zeros((*state.shape[:-1], 1))
        weights[:, 0, :] = 1
        i = max(-len(self.PVM), -state.shape[0])
        weights[i:, :, :] = np.array(self.PVM[i:]).reshape(-1, state.shape[1], 1)
        return np.concatenate([state, weights], axis=-1)

    def get_percentage_return(self, action):
        relative_price = self.relative_price[self.t+self.window_size-2]
        last_weight = self.PVM[-1]
        total_weighted_relative_price = np.dot(relative_price, last_weight)
        rebalanced_weight = relative_price * last_weight / total_weighted_relative_price
        # defined in page 5: mu
        transaction_remainder_factor = calculate_transaction_remainder_factor(action, rebalanced_weight, self.commission_rate)
        # immediate percentage return of equation 3
        return transaction_remainder_factor * total_weighted_relative_price

    def is_done(self):
        return self.portfolio_value <= 0 or self.t - self.t0 >= self.episode_length

    def reset(self):
        # t: 0, 1, 2, 3, 4, 5
        init_weight = np.array([1] + [0]*(self.global_data.shape[1]-1))
        self.PVM = []
        self.PVM.append(init_weight)
        self.portfolio_value = 1
        # randomly select a start point in the dataset
        self.t = self.t0 = self.init_time_rng.choice(len(self.global_data)-self.episode_length-self.window_size+1)
        # at the beginning of t=1, the prices are the close prices of 0
        prices = self.get_submatrix(self.t)
        # observation, terminated, truncated, info
        return self.get_state(prices), {'portfolio_value': self.portfolio_value}

    def step(self, action):
        self.t += 1
        # at the beginning of t, we know all close prices of 0, ... t-1
        prices = self.get_submatrix(self.t)
        percentage_return = self.get_percentage_return(action)
        # equation 11: calculate the portfolio value at current time.
        self.portfolio_value *= percentage_return
        self.PVM.append(action)
        state = self.get_state(prices)
        # observation, reward, terminated, truncated, info
        return state, np.log(percentage_return), self.is_done(), False, {'portfolio_value': self.portfolio_value}

    def get_submatrix(self, time):
        return self.global_data[time:time + self.window_size, :, :]
