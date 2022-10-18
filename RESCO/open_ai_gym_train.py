import os
import pickle
import torch
import numpy as np
import pandas as pd
import gym

from .module import Agent
from .PPO import IPPO
from .agent_config import agent_configs

# 学習させる関数
def train_agent_gym(
    env_name, model_save_path=None, episode_per_learn=10, episodes=1400,  max_steps=200, num_layers=1, 
    num_hidden_units=128, lr=3e-5, decay_rate=0.01, temperature=1.0, noise=0.0, encoder_type="fc", 
    lstm_len=5, embedding_type="random", embedding_num=5, embedding_decay=0.99, eps=1e-5, beta=0.25, 
    embedding_no_train=False, embedding_start_train=None, log_dir="./", learn_curve_csv=None, 
    save_actions=False, device="cpu", gui=False):
    
    env = gym.make(env_name)
    num_states = 1
    for i in range(len(env.observation_space.shape)):
        num_states *= env.observation_space.shape[i]
    num_actions = [env.action_space.n]
    
    agent = Agent(
        num_states=num_states, num_traffic_lights=1, num_actions=num_actions, 
        num_layers=num_layers, num_hidden_units=num_hidden_units, temperature=temperature, noise=noise, 
        encoder_type=encoder_type, lr=lr, decay_rate=decay_rate, embedding_type=embedding_type, 
        embedding_num=embedding_num, embedding_decay=embedding_decay, eps=eps, beta=beta, 
        embedding_no_train=embedding_no_train, embedding_start_train=embedding_start_train, 
        is_train=True, device=device)

    best_reward_sum = float("-inf")
    steps_list = list()
    loss_list = list()
    current_reward = list()

    if save_actions:
        actions_data = list()

    for i in range(episodes):
        if save_actions:
            actions_data_episode = list()
        
        if encoder_type == "lstm":
            obs_seq = list()
        
        obs = env.reset()

        steps = 0
        for j in range(max_steps):
            if encoder_type == "lstm":
                if len(obs_seq) == lstm_len:
                    chosen_actions = agent.act(obs_seq)
                    action = chosen_actions[0]
                else:
                    action = np.random.randint(num_actions[0])
            else:
                chosen_actions = agent.act(obs)
                action = chosen_actions[0]

            if save_actions:
                actions_data_episode.extend(chosen_actions)
            
            obs, reward, done, info = env.step(action)
            if gui:
                env.render()
            current_reward.append(reward)
            agent.set_rewards([reward])

            if encoder_type == "lstm":
                obs_seq.append(obs)
                if len(obs_seq) > lstm_len:
                    obs_seq.pop(0)
            steps += 1
            if done:
                steps_list.append(steps)

        if (i+1) % episode_per_learn == 0:
            loss = agent.train(return_loss=True)
            loss_list.append(loss)
            agent.reset_batch()

            current_reward_sum = np.sum(current_reward)
            if current_reward_sum > best_reward_sum:
                best_reward_sum = current_reward_sum
                current_reward = list()
                agent.save_model("best_" + model_save_path)
        
        if save_actions:
            actions_data.append(actions_data_episode)
        
        print(env_name + ": episodes " + str(i + 1) + " ended", str(steps) + "steps")

    env.close()
    agent.reset_batch()

    if learn_curve_csv is not None:
        learn_num = -(-episodes // episode_per_learn)
        mean_steps = list()
        for i in range(learn_num):
            mean_steps.append(np.mean(steps_list[episode_per_learn * i:episode_per_learn * (i + 1)]))

        analysis_data = {"mean_steps": mean_steps, "loss": loss_list}
        analysis_dataframe = pd.DataFrame(analysis_data)
        analysis_dataframe.to_csv(learn_curve_csv, index=False)

    agent.save_model(model_save_path)
    if save_actions:
        path = os.path.join(log_dir, "actions_data.pkl")
        with open(path, "wb") as f:
            pickle.dump(actions_data, f)

# PPOの学習
def train_PPO_gym(
    env_name, episode_per_learn=10, episodes=1400,  max_steps=200, num_layers=1, 
    num_hidden_units=128, lr=3e-5, decay_rate=0.01, temperature=1.0, noise=0.0, encoder_type="fc", 
    lstm_len=5, embedding_type="random", embedding_num=5, embedding_decay=0.99, eps=1e-5, beta=0.25, 
    update_interval=1024, minibatch_size=256, epochs=4, embedding_no_train=False, embedding_start_train=None, 
    log_dir="./", learn_curve_csv=None, 
    model_type="original", save_actions=False, device="cpu", gui=False
    ):

    env = gym.make(env_name)
    num_states = 1
    for i in range(len(env.observation_space.shape)):
        num_states *= env.observation_space.shape[i]
    num_actions = env.action_space.n

    agt_config = agent_configs["IPPO"]

    agt_config['log_dir'] = log_dir
    
    obs_act = {"main": [(num_states, ), num_actions]}

    map_name = "main"
    trial = 1
    
    if model_type == "default":
        model_param = {
            "temperature": temperature, "noise": noise, "encoder_type": encoder_type, 
            "embedding_type": embedding_type, "embedding_no_train": embedding_no_train, 
            "embedding_num": embedding_num, "embedding_decay": embedding_decay, 
            "beta": beta, "eps": eps, "device": device
        }

        agent = IPPO(agt_config, obs_act, map_name, trial, model_type, model_param)
    elif model_type == "original":
        model_param = {
            "num_layers": num_layers, "num_hidden_units": num_hidden_units, "temperature": temperature,
            "noise": noise, "encoder_type": encoder_type, "embedding_type": embedding_type, 
            "embedding_no_train": embedding_no_train, "embedding_num": embedding_num, 
            "embedding_decay": embedding_decay, "beta": beta, "eps": eps, "device": device
        }
        
        agent = IPPO(agt_config, obs_act, map_name, trial, model_type, model_param, update_interval, minibatch_size, epochs, lr, decay_rate)
    
    best_reward_sum = float("-inf")
    steps_list = list()
    current_reward = list()

    if save_actions:
        actions_data = list()

    for i in range(episodes):
        if save_actions:
            actions_data_episode = list()
        
        if encoder_type == "lstm":
            obs_seq = list()
        
        obs = env.reset()

        steps = 0
        for j in range(max_steps):
            if encoder_type == "lstm":
                if len(obs_seq) == lstm_len:
                    chosen_actions = agent.act({"main":obs_seq})
                    action = chosen_actions["main"]
                else:
                    action = np.random.randint(num_actions[0])
            else:
                chosen_actions = agent.act({"main":obs})
                action = chosen_actions["main"]

            if save_actions:
                actions_data_episode.extend([chosen_actions["main"]])
            
            obs, reward, done, info = env.step(action)
            if gui:
                env.render()
            current_reward.append(reward)
            agent.observe({"main":obs}, {"main":reward}, done, info)

            if encoder_type == "lstm":
                obs_seq.append(obs)
                if len(obs_seq) > lstm_len:
                    obs_seq.pop(0)
            steps += 1
            if done:
                steps_list.append(steps)
        
        if save_actions:
            actions_data.append(actions_data_episode)
        
        print(env_name + ": episodes " + str(i + 1) + " ended", str(steps) + "steps")

    env.close()

    if learn_curve_csv is not None:
        learn_num = -(-episodes // episode_per_learn)
        mean_steps = list()
        for i in range(learn_num):
            mean_steps.append(np.mean(steps_list[episode_per_learn * i:episode_per_learn * (i + 1)]))

        analysis_data = {"mean_steps": mean_steps}
        analysis_dataframe = pd.DataFrame(analysis_data)
        analysis_dataframe.to_csv(learn_curve_csv, index=False)

    num = 1
    for model in agent.agents.values():
        filename = "agent" + str(num)
        path = os.path.join(log_dir, filename)
        model.save(path)
        num += 1

    if save_actions:
        path = os.path.join(log_dir, "actions_data.pkl")
        with open(path, "wb") as f:
            pickle.dump(actions_data, f)
