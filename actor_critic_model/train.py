from tqdm import tqdm
from os import path

weight_file = path.realpath(__file__).\
    replace('actor_critic_model/train.py','actor_critic_model/weights/')

def train(env, agent, epoch, max_time_steps, start_update, batch_size, noise, replay_buffer, action_valid_fn):
    total_rewards = []

    for _ in tqdm(range(epoch)):
        prev_state, info = env.reset()
        total_reward = 0
        for time_step in range(max_time_steps):
            action = action_valid_fn(agent.select_action(prev_state) + noise())
            # observation, reward, terminated, truncated, info
            state, reward, done, truncated, info = env.step(action)
            replay_buffer.add(prev_state, action, state, reward, done)
            prev_state = state
            total_reward += reward
            # Train agent after collecting sufficient data
            if time_step >= start_update:
                agent.update(replay_buffer, batch_size)
            if time_step % 5 == 0:
                agent.save(weight_file)
            if done or truncated:
                break
        total_rewards.append(total_reward)
    return total_rewards
