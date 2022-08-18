import configparser
from RESCO.test import test_PPO
from RESCO import states, rewards
from RESCO.agent_config import agent_configs
from RESCO.map_config import map_configs
import subprocess
import random
import numpy as np
import torch
import logging

if __name__=="__main__":
    map_name = "cologne1"
    dir_name = "base"
    model_load_path = ["./agent1.pt"]
    episodes = 1
    config_agent_name = "IPPO"
    temperature = 1.0
    noise = 0.0
    encoder_type = "fc"
    embedding_type = "random"
    embedding_num = 5
    model_type = "default"
    log_dir = "./"
    device = "cpu"
    gui = True
    first_sleep = 5
    sleep_interval = 0.1

    env_config = configparser.ConfigParser()
    env_config.read("env_config.ini", encoding='utf-8')
    map_dir = env_config.get("DEFAULT", "RESCO_path")
    env_base = env_config.get("DEFAULT", "traffic_map_path")
    # port = int(env_config.get("DEFAULT", "port"))
    port = None

    run_name = config_agent_name + "_" + dir_name
    map_config = map_configs[map_name]
    agent_config = agent_configs[config_agent_name]
    net_file = map_dir + map_config["net"]
    if map_config["route"] is None:
        route_file = None
    else:
        route_file = map_dir + map_config["route"]
    state_f = eval(agent_config["state"])
    reward_f = eval(agent_config["reward"])
    step_length = map_config["step_length"]
    yellow_length = map_config["yellow_length"]
    step_ratio = map_config["step_ratio"]
    start_time = map_config["start_time"]
    end_time = map_config["end_time"]
    max_distance = agent_config["max_distance"]
    lights = map_config["lights"]
    warmup = map_config["warmup"]
    trial = 1

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # ファイル出力ハンドラーの設定
    handler = logging.FileHandler("error_message.log")
    handler.setLevel(logging.DEBUG)
    # 出力フォーマットの設定
    formatter = logging.Formatter('%(levelname)s  %(asctime)s  [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    # ハンドラーの追加
    logger.addHandler(handler)
    # 重複出力をなくす
    logger.propagate = False
    try:
        test_PPO(
            run_name=run_name, map_name=map_name, net_file=net_file, route_file=route_file, 
            state_f=state_f, reward_f=reward_f, model_load_path=model_load_path,
            episodes=episodes, step_length=step_length, yellow_length=yellow_length, step_ratio=step_ratio, 
            start_time=start_time, end_time=end_time, max_distance=max_distance, lights=lights, warmup=warmup, 
            num_layers=1, num_hidden_units=512, temperature=temperature, noise=noise, encoder_type=encoder_type, 
            lstm_len=5, embedding_type=embedding_type, embedding_num=embedding_num, model_type=model_type, 
            log_dir=log_dir, env_base=env_base, device=device, port=port, trial=trial, 
            libsumo=False, gui=gui, first_sleep=first_sleep, sleep_interval=sleep_interval
            )
    except Exception as err:
        data_dir = log_dir + run_name + '-tr' + str(trial) + '-' + map_name + '-' + str(len(lights)) + '-' + state_f.__name__ + '-' + reward_f.__name__
        dir_content = subprocess.run(["find", data_dir, "-type", "f"], stdout=subprocess.PIPE, encoding="utf-8")
        file_num = subprocess.run(["wc", "-l"], input=dir_content.stdout, stdout=subprocess.PIPE, encoding="utf-8")
        episode_num = int(file_num.stdout)//2 + 1
        logger.debug("episode: " + str(episode_num))
        logger.exception("The program stopped because of this error.")
    finally:
        log_path = log_dir + run_name + "-tr1-" + map_name + "-" + str(len(lights)) + "-" + state_f.__name__ + "-" + reward_f.__name__
        subprocess.run(["rm", "-r", log_path])