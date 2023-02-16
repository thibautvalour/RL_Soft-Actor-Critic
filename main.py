import numpy as np
import gym
import pybullet_envs
import time

from sac_agent import Agent
from utils import plot_learning_curve

if __name__ == '__main__':

    # Game settings
    game = 'HopperBulletEnv-v0'
    n_steps = 20_000
    start_learning = 500
    save_result_data = True
    figure_file = 'results/plot/' + game
    video_file = 'results/video/' + game
    result_data_file = 'results/data/' + game
    save_data_and_plot_interval = 50
    video_start = 20
    video_interval = 10

    # Hyperparameters
    lr_actor, lr_critic_value = 3e-4, 3e-4
    max_size_buffer = 1_000_000
    gamma = 0.99
    tau = 0.005
    alpha = 2
    fc1_dims, fc2_dims = 256, 256
    batch_size = 256

    env = gym.make(game)

    # Video record
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, video_file,
            episode_trigger=lambda episode_id: \
                (episode_id>=video_start) and (episode_id%video_interval==0))
    
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0], max_size=max_size_buffer, 
            lr_actor=lr_actor, lr_critic_value=lr_critic_value, 
            gamma=gamma, tau=tau, alpha=alpha,
            fc1_dims=fc1_dims, fc2_dims=fc2_dims,
            batch_size=batch_size)


    best_score = env.reward_range[0]
    result_dict = {'score_history': [], 'step_history': []}

    n_game, n_step = 0, 0
    while n_step < n_steps:
        start = time.time()
        n_game += 1
        observation = env.reset()
        done = False
        score = 0
        while not done:
            train = (n_step > start_learning)
            action = agent.choose_action(observation, train=train)
            observation_, reward, done, info = env.step(action)
            n_step += 1
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if train:
                agent.learn()
            observation = observation_
        result_dict['score_history'].append(score)
        result_dict['step_history'].append(n_step)
        avg_score = np.mean(result_dict['score_history'][-100:])

        if avg_score > (best_score+abs(best_score)*1.25):
            best_score = avg_score

        end = time.time()
        print('episode ', n_game, 'score %.1f' % score, 'avg_score %.1f' % avg_score, 'n_step', n_step, 'time %.2f' % (end-start))

        if  save_result_data and (n_game%save_data_and_plot_interval == 0):
            print('Saving result data and plot...')
            plot_learning_curve(result_dict, figure_file) # Plot score
            np.save(result_data_file, result_dict) # Save result data            
