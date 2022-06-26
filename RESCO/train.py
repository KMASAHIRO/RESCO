import os
import torch
import numpy as np
import pandas as pd
import traci

from .multi_signal import MultiSignal
from .module import Agent
from .analysis import read_csv, read_xml

# 学習させる関数
def train_agent(
    run_name, map_name, net_file, route_file, state_f, reward_f, model_save_path=None, 
    episode_per_learn=2, episodes=100, step_length=10, yellow_length=4, step_ratio=1, 
    end_time=3600, max_distance=200, lights=(), warmup=0, num_layers=1, num_hidden_units=512, 
    lr=3e-5, decay_rate=0.01, temperature=1.0, noise=0.0, encoder_type="fc", lstm_len=5, 
    embedding_type="random", embedding_num=5, embedding_decay=0.99, eps=1e-5, beta=0.25, 
    embedding_no_train=False, embedding_start_train=None, log_dir="./", env_base="../RESCO/environments/", 
    reward_csv=None, loss_csv=None, device="cpu", port=None, trial=1, libsumo=False):
    
    csv_dir = log_dir + run_name + '-tr' + str(trial) + '-' + map_name + '-' + str(len(lights)) + '-' + state_f.__name__ + '-' + reward_f.__name__ + "/"

    env = MultiSignal(
        run_name=run_name+'-tr'+str(trial), map_name=map_name,
        net=net_file, state_fn=state_f, reward_fn=reward_f,
        route=route_file, step_length=step_length, yellow_length=yellow_length, 
        step_ratio=step_ratio, end_time=end_time, max_distance=max_distance, 
        lights=lights, gui=False, log_dir=log_dir, libsumo=libsumo, 
        warmup=warmup, port=port)
    
    traffic_light_ids = env.all_ts_ids
    num_states = 0
    num_actions = list()
    for id in traffic_light_ids:
        states_dim = 1
        for i in range(len(env.obs_shape[id])):
            states_dim *= env.obs_shape[id][i]
        num_states += states_dim
        num_actions.append(len(env.phases[id]))
    
    agent = Agent(
        num_states=num_states, num_traffic_lights=len(traffic_light_ids), num_actions=num_actions, 
        num_layers=num_layers, num_hidden_units=num_hidden_units, temperature=temperature, noise=noise, 
        encoder_type=encoder_type, lr=lr, decay_rate=decay_rate, embedding_type=embedding_type, 
        embedding_num=embedding_num, embedding_decay=embedding_decay, eps=eps, beta=beta, 
        embedding_no_train=embedding_no_train, embedding_start_train=embedding_start_train, 
        is_train=True, device=device)

    loss_list = list()
    learn_episodes = list()
    best_reward_mean = float("-inf")
    current_reward = list()
    for i in range(episodes):
        if encoder_type == "lstm":
            obs_seq = list()
        
        obs_dict = env.reset()
        obs = np.concatenate(list(obs_dict.values())).flatten()

        while True:
            if encoder_type == "lstm":
                if len(obs_seq) == lstm_len:
                    chosen_actions = agent.act(obs_seq)
                    action = dict()
                    for k in range(len(traffic_light_ids)):
                        action[traffic_light_ids[k]] = chosen_actions[k]
                else:
                    action = dict()
                    for k in range(len(traffic_light_ids)):
                        action[traffic_light_ids[k]] = None
            else:
                chosen_actions = agent.act(obs)
                action = dict()
                for k in range(len(traffic_light_ids)):
                    action[traffic_light_ids[k]] = chosen_actions[k]
            
            state = env.step(action)
            obs = np.concatenate(list(state[0].values())).flatten()
            reward = list(state[1].values())
            current_reward.append(np.sum(reward))
            end = state[2]
            agent.set_rewards(reward)

            if encoder_type == "lstm":
                obs_seq.append(obs)
                if len(obs_seq) > lstm_len:
                    obs_seq.pop(0)

            if end:
                break

        if (i+1) % episode_per_learn == 0:
            if loss_csv is not None:
                loss = agent.train(return_loss=True)
                loss_list.append(loss)
            else:
                agent.train()
            learn_episodes.append(i+1)
            agent.reset_batch()

            current_reward_mean = np.mean(current_reward)
            if current_reward_mean > best_reward_mean:
                best_reward_mean = current_reward_mean
                current_reward = list()
                agent.save_model("best_" + model_save_path)
        
        print(run_name + '-tr' + str(trial) + "-" + map_name + ": episodes " + str(i + 1) + " ended")
    
    env.close()
    agent.reset_batch()
    if loss_csv is not None:
        loss_data = {"episode": learn_episodes, "loss": loss_list}
        loss_df = pd.DataFrame(loss_data)
        loss_df.to_csv(loss_csv)

    if reward_csv is not None:
        episode_num = list()
        mean_reward = list()
        mean_total_stopped = list()
        mean_total_wait_time = list()
        for i in range(episodes):
            load_path = csv_dir + "metrics_" + str(i + 1) + ".csv"
            dataframe = pd.read_csv(load_path, header=None).dropna(axis=0)
            reward_sum = list()
            for j in range(len(dataframe.index)):
                reward_str = ",".join(list(dataframe.iloc[j, 1:1+len(traffic_light_ids)]))
                reward_sum.append(np.sum(list(eval(reward_str).values())))

            mean_reward.append(np.mean(reward_sum))

        learn_num = -(-episodes // episode_per_learn)
        mean_reward_learn = list()
        for i in range(learn_num):
            mean_reward_learn.append(np.mean(mean_reward[episode_per_learn * i:episode_per_learn * (i + 1)]))

        analysis_data = {"episode": learn_episodes, "mean_reward": mean_reward_learn}
        analysis_dataframe = pd.DataFrame(analysis_data)
        analysis_dataframe.to_csv(reward_csv, index=False)

    read_csv(log_dir)
    read_xml(log_dir, env_base)
    agent.save_model(model_save_path)