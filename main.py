import numpy as np
import torch

from agents import BAC
from metrics import plot_total_reward, plot_confidence_intervals, plot_boxplot
from utils import ContinuousMemory, RunningStat
from models import CNNActor, Value, CNNFeatureExtractor
from environment import TradingEnvWrapper, get_history_data_by_coins

torch.manual_seed(0)
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
def evaluation(env, agent, max_time_steps, eval_times, episode_length):
    performance = RunningStat(episode_length)
    total_rewards = []
    for _ in range(eval_times):
        prev_state, info = env.reset()
        portfolio_value = []
        total_reward = 0
        for time_step in range(max_time_steps):
            action = agent.select_action(prev_state)
            # observation, reward, terminated, truncated, info
            state, reward, done, truncated, info = env.step(action)
            portfolio_value.append(info['portfolio_value'])
            prev_state = state
            total_reward += reward
            if done or truncated:
                break
        performance.record(np.array(portfolio_value))
        total_rewards.append(total_reward)
    return {'performance': performance,
            'total_rewards': total_rewards}

def backtest(test_envs, agent, eval_times, metrics):
    episode_length = test_envs[0].episode_length
    # test
    performance_data = []
    boxplot_data = []
    boxplot_labels = []
    for i, env in enumerate(test_envs):
        performance_sta = evaluation(env, agent, max_time_steps, eval_times, episode_length)
        performance_data.append({'sample_mean': performance_sta['performance'].mean(),
                                 'half_window': performance_sta['performance'].half_window(0.95),
                                 'label': 'data_' + str(i)})
        boxplot_data.append(performance_sta['total_rewards'])
        boxplot_labels.append('data_' + str(i))

    if 'portfolio_values' in metrics:
        plot_confidence_intervals({'x': {'xlabel': 'time', 'data': np.arange(episode_length)},
                                   'y': {'ylabel': 'portfolio value', 'data': performance_data}})

    if 'total_rewards' in metrics:
        plot_boxplot({'x': {'xlabel': 'Environment', 'data': boxplot_labels},
                      'y':{'ylabel': 'total rewards', 'data': boxplot_data}})

if __name__ == '__main__':
    env_spec = {
        'episode_length': 128,
        'commission_rate': 0,
        'period': '30m',
        'coins': ['BTC', 'ETH', 'XRP', 'BNB', 'ADA'],
        'online': False,
        'features': ['close', 'high', 'low'],
        'baseAsset': 'USDT'
    }
    times = ['2018-06-01', '2020-06-01', '2021-06-01', '2022-12-31']
    start, end = 0, 1
    envs = []
    for end in range(1, len(times)):
        env_spec['start'], env_spec['end'] = times[end-1], times[end]
        e = TradingEnvWrapper(**env_spec)
        envs.append(e)
    env = envs[0]
    state_dim, action_dim = env.observation_space.shape, env.action_space.shape
    feature_extractor = CNNFeatureExtractor(state_dim)
    agent_spec = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'actor': CNNActor,
        'critic': Value,
        'discount': 0.995,
        'tau': 0.97,
        'advantage_flag': False,
        'actor_args': {'feature_extractor': feature_extractor},
        'critic_args': {'fisher_num_inputs': 50, 'feature_extractor': feature_extractor},
        'actor_lr': 3e-3,
        'critic_lr': 2e-2,
        'likelihood_noise_level': 1e-4
    }
    agent = BAC(**agent_spec)
    batch_size = 15
    max_time_steps = 1000
    memory = ContinuousMemory(state_dim, action_dim, batch_size * max_time_steps)
    train_spec = {
        'env': env,
        'epoch_num': 15000,
        'max_time_steps': max_time_steps,
        'batch_size': batch_size,
        'replay_memory': memory,
        'svd_low_rank': agent_spec['critic_args']['fisher_num_inputs'],
        'state_coefficient': 1,
        'fisher_coefficient': 5e-5
    }
    total_rewards = agent.fit(**train_spec)
    plot_total_reward(total_rewards)

    backtest_spec = {
        'test_envs': envs,
        'agent': agent,
        'eval_times': 200,
        'metrics': ['total_rewards', 'portfolio_values']
    }
    backtest(**backtest_spec)
