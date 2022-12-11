import argparse
import configparser
import subprocess
from RESCO.agent_config import agent_configs
from RESCO.map_config import map_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map", type=str, required=True)
    args = parser.parse_args()
    map_name = args.map

    env_config = configparser.ConfigParser()
    env_config.read("env_config.ini", encoding='utf-8')
    train_config = configparser.ConfigParser()
    train_config.read("train_config.ini", encoding='utf-8')

    train_path = env_config.get("DEFAULT", "do_train_path")
    map_dir = env_config.get("DEFAULT", "RESCO_path")
    env_base = env_config.get("DEFAULT", "traffic_map_path")
    port = int(env_config.get("DEFAULT", "port"))

    model_type = train_config.get("DEFAULT", "model_type")
    if model_type == "PPO":
        ppo_model_type = train_config.get("DEFAULT", "ppo_model_type")
    else:
        ppo_model_type = "original"
    config_agent_name = train_config.get("param", "agent_name")
    episodes = train_config.get("param", "episodes")
    episode_per_learn = train_config.get("param", "episode_per_learn")
    lr = train_config.get("param", "lr")
    device = train_config.get("DEFAULT", "device")

    learn_options = list()
    if "num_hidden_units" in train_config["param"]:
        learn_options.extend(["--num_hidden_units", train_config.get("param", "num_hidden_units")])
    if "update_interval" in train_config["param"]:
        learn_options.extend(["--update_interval", train_config.get("param", "update_interval")])
    if "minibatch_size" in train_config["param"]:
        learn_options.extend(["--minibatch_size", train_config.get("param", "minibatch_size")])
    if "epochs" in train_config["param"]:
        learn_options.extend(["--epochs", train_config.get("param", "epochs")])
    if "entropy_coef" in train_config["param"]:
        learn_options.extend(["--entropy_coef", train_config.get("param", "entropy_coef")])
    
    map_config = map_configs[map_name]
    agent_config = agent_configs[config_agent_name]

    sections = train_config.sections()
    experiments = dict()
    for sec in sections:
        if sec != "DEFAULT" and sec != "param":
            experiments[sec] = dict(train_config.items(sec))
            if "device" not in experiments[sec].keys():
                experiments[sec]["device"] = device

    for dir_name in experiments.keys():
        subprocess.run(["mkdir", dir_name])
        reward_csv = map_name + "_" + config_agent_name + "_" + dir_name + "_reward.csv"
        loss_csv = map_name + "_" + config_agent_name + "_" + dir_name + "_loss.csv"
        model_save_path = map_name + "_" + config_agent_name + "_" + dir_name + "_policy-function.pth"
        run_name = config_agent_name + "_" + dir_name
        python_cmd = [
            "python", train_path, "--model_type", model_type, "--ppo_model_type", ppo_model_type, 
            "--run_name", run_name, "--map_name", map_name, "--net_file", map_dir + map_config["net"], 
            "--state_f", str(agent_config["state"]), "--reward_f", str(agent_config["reward"]), 
            "--model_save_path", model_save_path, "--episodes", episodes, "--episode_per_learn", episode_per_learn, 
            "--step_length", str(map_config["step_length"]), "--yellow_length", str(map_config["yellow_length"]), 
            "--step_ratio", str(map_config["step_ratio"]), "--start_time", str(map_config["start_time"]), 
            "--end_time", str(map_config["end_time"]), "--max_distance", str(agent_config["max_distance"]), 
            "--lights", str(map_config["lights"]), "--warmup", str(map_config["warmup"]), "--lr", lr, 
            "--log_dir", "./", "--env_base", env_base, "--reward_csv", reward_csv, "--loss_csv", loss_csv, 
            "--save_actions", "--port", str(port)
            ]
        
        python_cmd.extend(learn_options)
        
        if map_config["route"] is not None:
            python_cmd.extend(["--route_file", map_dir + map_config["route"]])
        
        for op,val in experiments[dir_name].items():
            if val == "True":
                op_cmd = ["--" + op]
            else:
                op_cmd = ["--" + op, val]
            python_cmd.extend(op_cmd)
        
        port += 1

        subprocess.Popen(python_cmd, cwd=dir_name)
