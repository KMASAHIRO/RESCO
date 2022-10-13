import configparser
import subprocess

if __name__ == "__main__":
    env_config = configparser.ConfigParser()
    env_config.read("env_config.ini", encoding='utf-8')
    train_config = configparser.ConfigParser()
    train_config.read("train_config.ini", encoding='utf-8')

    train_path = env_config.get("DEFAULT", "do_train_path")

    env_name = train_config.get("DEFAULT", "env_name")
    model_type = train_config.get("DEFAULT", "model_type")
    if model_type == "PPO":
        ppo_model_type = train_config.get("DEFAULT", "ppo_model_type")
    else:
        ppo_model_type = "original"
    episodes = train_config.get("param", "episodes")
    max_steps = train_config.get("param", "max_steps")
    episode_per_learn = train_config.get("param", "episode_per_learn")
    num_hidden_units = train_config.get("param", "num_hidden_units")
    lr = train_config.get("param", "lr")
    device = train_config.get("DEFAULT", "device")
    
    learn_options = list()
    try:
        if train_config.get("param", "update_interval"):
            learn_options.extend(["--update_interval", train_config.get("param", "update_interval")])
        if train_config.get("param", "minibatch_size"):
            learn_options.extend(["--minibatch_size", train_config.get("param", "minibatch_size")])
        if train_config.get("param", "epochs"):
            learn_options.extend(["--epochs", train_config.get("param", "epochs")])
    except:
        pass

    sections = train_config.sections()
    experiments = dict()
    for sec in sections:
        if sec != "DEFAULT" and sec != "param":
            experiments[sec] = dict(train_config.items(sec))
            if "lr" not in experiments[sec].keys():
                experiments[sec]["lr"] = lr
            if "num_hidden_units" not in experiments[sec].keys():
                experiments[sec]["num_hidden_units"] = num_hidden_units

    for dir_name in experiments.keys():
        subprocess.run(["mkdir", dir_name])
        learn_curve_csv = env_name + "_" + dir_name + "_reward.csv"
        model_save_path = env_name + "_" + dir_name + "_policy-function.pth"
        python_cmd = [
            "python", train_path, "--model_type", model_type, "--ppo_model_type", ppo_model_type, 
            "--env_name", env_name, "--model_save_path", model_save_path, "--episodes", episodes, 
            "--max_steps", max_steps, "--episode_per_learn", episode_per_learn, 
            "--log_dir", "./", "--learn_curve_csv", learn_curve_csv, "--save_actions", 
            "--device", device
            ]
        
        #python_cmd.extend(learn_options)
        
        for op,val in experiments[dir_name].items():
            if val == "True":
                op_cmd = ["--" + op]
            else:
                op_cmd = ["--" + op, val]
            python_cmd.extend(op_cmd)
        
        subprocess.Popen(python_cmd, cwd=dir_name)
