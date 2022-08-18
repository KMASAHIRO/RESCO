import os
import torch
import numpy as np
import traci

from .multi_signal import MultiSignal
from .module import Agent
from .PPO import IPPO
from .analysis import read_csv, read_xml
from .agent_config import agent_configs

# PPOのテスト
def test_PPO(
    run_name, map_name, net_file, route_file, state_f, reward_f, model_load_path=[],
    episodes=100, step_length=10, yellow_length=4, step_ratio=1, 
    start_time=0, end_time=3600, max_distance=200, lights=(), warmup=0, num_layers=1, 
    num_hidden_units=512, temperature=1.0, noise=0.0, encoder_type="fc", 
    lstm_len=5, embedding_type="random", embedding_num=5, model_type="original", 
    log_dir="./", env_base="../RESCO/environments/", device="cpu", port=None, trial=1, 
    libsumo=False, gui=False
    ):

    csv_dir = log_dir + run_name + '-tr' + str(trial) + '-' + map_name + '-' + str(len(lights)) + '-' + state_f.__name__ + '-' + reward_f.__name__ + "/"

    env = MultiSignal(
        run_name=run_name+'-tr'+str(trial), map_name=map_name,
        net=net_file, state_fn=state_f, reward_fn=reward_f,
        route=route_file, step_length=step_length, yellow_length=yellow_length, 
        step_ratio=step_ratio, end_time=end_time, max_distance=max_distance, 
        lights=lights, gui=gui, log_dir=log_dir, libsumo=libsumo, 
        warmup=warmup, port=port)

    traffic_light_ids = env.all_ts_ids

    agt_config = agent_configs["IPPO"]

    num_steps_eps = int((end_time - start_time) / step_length)

    agt_config['episodes'] = int(episodes * 0.8)    # schedulers decay over 80% of steps
    agt_config['steps'] = agt_config['episodes'] * num_steps_eps
    agt_config['log_dir'] = log_dir + env.connection_name + os.sep
    agt_config['num_lights'] = len(env.all_ts_ids)

    # Get agent id's, observation shapes, and action sizes from env
    obs_act = dict()
    for key in env.obs_shape:
        obs_act[key] = [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
    
    if model_type == "default":
        model_param = {
            "temperature": temperature, "noise": noise, "encoder_type": encoder_type, 
            "embedding_type": embedding_type, "embedding_num": embedding_num, "device": device
        }

        agent = IPPO(agt_config, obs_act, map_name, trial, model_type, model_param)
    elif model_type == "original":
        model_param = {
            "num_layers": num_layers, "num_hidden_units": num_hidden_units, "temperature": temperature,
            "noise": noise, "encoder_type": encoder_type, "embedding_type": embedding_type, 
            "embedding_num": embedding_num, "device": device
        }
        
        agent = IPPO(agt_config, obs_act, map_name, trial, model_type, model_param)
    
    for _ in range(episodes):
        obs = env.reset()
        if model_type == "original":
            for key in obs.keys():
                obs[key] = obs[key].flatten()
        done = False
        while not done:
            act = agent.act(obs)
            obs, rew, done, info = env.step(act)
            if model_type == "original":
                for key in obs.keys():
                    obs[key] = obs[key].flatten()

    env.close()

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