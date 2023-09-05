import torch
torch.manual_seed(0)
import torch.nn as nn
import os
import pickle
import sys
DIR_PATH = "./"
python_folders = ['environment', 'actor_critic_model', 'marketAPIWrapper', 'metric', 'utils', 'backtest']
for folder in python_folders:
    sys.path.append(os.path.join(DIR_PATH, folder))
from environment.globaldatamatrix import get_history_data_by_coins
from backtest.backtest import backtest
from scipy.special import softmax
from utils.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from utils.replay_buffer import ReplayBuffer

class StateModel(nn.Module):

    def __init__(self, states_dim, action_dim, kernel_size=3, hidden_channel1=2, hidden_channel2=20):
        super(StateModel, self).__init__()
        self.eiie_nets = nn.Sequential(
                nn.Conv2d(states_dim[2]-1, hidden_channel1, kernel_size=(kernel_size, 1), stride=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channel1, hidden_channel2, kernel_size=(states_dim[0]-kernel_size + 1, 1), stride=1),
                nn.ReLU())
        self.hidden_layer = nn.Sequential(
            nn.Conv2d(hidden_channel2 + 1, 1, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            )

    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        last_weights, prices = state[:, -1:, -1:, :], state[:, :-1, :, :]
        out = self.eiie_nets(prices)
        out = torch.cat([out, last_weights], dim=1)
        return self.hidden_layer(out).squeeze()

class Actor(nn.Module):
    """
    The Actor model takes in a state observation as input and
    outputs an action, which is a continuous value.

    It consists of four fully coonected linear layers with ReLU activation functions and
    a final output layer selects one single optimized action for the state
    """
    def __init__(self, states_dim, action_dim, kernel_size=3, hidden_channel1=2, hidden_channel2=20, shared=None):
        super(Actor, self).__init__()

        if shared:
            state_model = shared
        else:
            state_model = StateModel(states_dim, action_dim, kernel_size, hidden_channel1, hidden_channel2)
        self.nets = nn.Sequential(
            state_model,
            nn.Softmax(dim=-1)
            )

    def forward(self, state):
        return self.nets(state)

class Critic(nn.Module):
    """
    The Critic model takes in both a state observation and an action as input and
    outputs a Q-value, which estimates the expected total reward for the current state-action pair.

    It consists of four linear layers with ReLU activation functions,
    State and action inputs are concatenated before being fed into the first linear layer.

    The output layer has a single output, representing the Q-value
    """
    def __init__(self, states_dim, action_dim, kernel_size=3, hidden_channel1=2, hidden_channel2=20, hidden_layer=64, shared=None):
        super(Critic, self).__init__()
        if shared:
            state_model = shared
        else:
            state_model = StateModel(states_dim, action_dim, kernel_size, hidden_channel1, hidden_channel2)
        self.state_net = nn.Sequential(
            state_model,
            nn.Linear(action_dim[0], hidden_layer),
            nn.ReLU()
        )
        self.action_net = nn.Sequential(
            nn.Linear(action_dim[0], hidden_layer),
            nn.ReLU()
            )
        self.final_net = nn.Linear(hidden_layer, 1)

    def forward(self, state, action):
        out = self.state_net(state) + self.action_net(action)
        return self.final_net(out)

def main(spec_file):
    if os.path.exists('data.pkl'):
        with open('data.pkl', 'rb') as f:
            history_data = pickle.load(f)
    else:
        history_data = get_history_data_by_coins(**spec_file['env'])
        with open('data.pkl', 'wb') as f:
            pickle.dump(history_data, f)
    backtest(history_data, agent_params=spec_file['agent'], **spec_file['train'], **spec_file['evaluation'])

if __name__ == '__main__':
    spec_file = {
        'env': {
            'start': '2018-06-01',
            'end': '2022-12-31',
            'period': '30m',
            'coins': ['BTC', 'ETH', 'XRP', 'BNB', 'ADA'],
            'online': False,
            'features': ['close', 'high', 'low'],
            'baseAsset': 'USDT'
        },
        'agent': {
            'actor': Actor,
            'critic': Critic,
            'discount': 0.99,
            'tau': 0.002,
            'actor_lr': 1e-3,
            'critic_lr': 1e-3,
            'actor_args': {'kernel_size': 3, 'hidden_channel1': 2, 'hidden_channel2': 20},
            'critic_args': {'kernel_size': 3, 'hidden_channel1': 2, 'hidden_channel2': 20, 'hidden_layer': 64}
        },
        'train': {
            'epoch': 2,
            'episode_length': 512,
            'max_time_steps': 5000,
            'start_update': 100,
            'batch_size': 64,
            'noise': OrnsteinUhlenbeckActionNoise,
            'replay_buffer': ReplayBuffer,
            'action_valid_fn': lambda action: softmax(action),
            'noise_args': {'mu': 0, 'sigma': 1},
            'replay_buffer_args': {}
        },
        'evaluation': {
            'ratios': [0.8, 0.1, 0.1],
            'eval_times': 2,
            'metrics': ['total_rewards', 'portfolio_values']
        }
    }
    main(spec_file)