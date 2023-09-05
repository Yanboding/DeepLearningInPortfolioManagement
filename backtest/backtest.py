import numpy as np
from environment.tradingEnvironment import TradingEnv
from actor_critic_model.ddpg import DDPG
from actor_critic_model.train import train
from metrics.visulization import plot_total_reward, plot_confidence_intervals, plot_boxplot
from metrics.stat import Stat


def train_validation_test(data, ratios=[0.8, 0.1, 0.1]):
    res = []
    data_size = len(data)
    prev = 0
    cum_ratio = 0
    for ratio in ratios:
        cum_ratio += ratio
        end = round(data_size * cum_ratio)
        res.append(data[prev:end])
        prev = end
    return res


def evaluation(env, agent, max_time_steps, eval_times, episode_length, action_valid_fn):
    performance = Stat(episode_length)
    total_rewards = []
    for _ in range(eval_times):
        prev_state, info = env.reset()
        portfolio_value = []
        total_reward = 0
        for time_step in range(max_time_steps):
            action = action_valid_fn(agent.select_action(prev_state))
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

def backtest(history_data, agent_params, ratios, episode_length, epoch, eval_times, max_time_steps, start_update, batch_size,
             noise, replay_buffer, action_valid_fn, noise_args, replay_buffer_args, metrics):
    dataset = train_validation_test(history_data, ratios=ratios)
    train_env = TradingEnv(dataset[0], episode_length=episode_length)
    state_dim, action_dim = train_env.observation_space.shape, train_env.action_space.shape
    # 1. how log return changes after n episodes
    agent = DDPG(state_dim, action_dim, **agent_params)
    total_rewards = train(train_env, agent, epoch, max_time_steps, start_update, batch_size,
                          noise(action_dim, **noise_args), replay_buffer(state_dim, action_dim, **replay_buffer_args),
                          action_valid_fn)
    plot_total_reward(total_rewards)

    # test
    performance_data = []
    boxplot_data = []
    boxplot_labels = []
    for i, data in enumerate(dataset):
        env = TradingEnv(data, episode_length=episode_length)
        performance_sta = evaluation(env, agent, max_time_steps, eval_times, episode_length, action_valid_fn)
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

