from ast import parse
from RESCO.train import train_agent, train_PPO
from RESCO import states, rewards
import subprocess
import argparse
import random
import numpy as np
import torch
import logging

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="original")
    parser.add_argument("--ppo_model_type", type=str, default="original")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--map_name", type=str, required=True)
    parser.add_argument("-n", "--net_file", type=str, required=True)
    parser.add_argument("-r", "--route_file", type=str, default="")
    parser.add_argument("--state_f", type=str, required=True)
    parser.add_argument("--reward_f", type=str, required=True)
    parser.add_argument("--model_save_path", type=str, required=True)
    parser.add_argument("--episode_per_learn", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=1400)
    parser.add_argument("--step_length", type=int, default=10)
    parser.add_argument("--yellow_length", type=int, default=4)
    parser.add_argument("--step_ratio", type=int, default=1)
    parser.add_argument("--start_time", type=int, default=0)
    parser.add_argument("--end_time", type=int, default=3600)
    parser.add_argument("--max_distance", type=int, default=200)
    parser.add_argument("--lights", type=str, default="()")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--num_hidden_units", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--encoder_type", type=str, default="fc")
    parser.add_argument("--embedding_type", type=str, default="random")
    parser.add_argument("--embedding_num", type=int, default=5)
    parser.add_argument("--embedding_decay", type=float, default=0.99)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--update_interval", type=int, default=1024)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--embedding_no_train", action="store_true")
    parser.add_argument("--embedding_start_train", type=int, default=0)
    parser.add_argument("--noisy_layer_num", type=int, default=1)
    parser.add_argument("--bbb_layer_num", type=int, default=1)
    parser.add_argument("--noisy_layer_type", type=str, default="action")
    parser.add_argument("--bbb_layer_type", type=str, default="action")
    parser.add_argument("--bbb_pi", type=float, default=0.5)
    parser.add_argument("--bbb_sigma1", type=float, default=-0)
    parser.add_argument("--bbb_sigma2", type=float, default=-6)
    parser.add_argument("--log_dir", type=str, default="./")
    parser.add_argument("--env_base", type=str, default="../RESCO/environments/")
    parser.add_argument("--reward_csv", type=str, default="")
    parser.add_argument("--loss_csv", type=str, default="")
    parser.add_argument("--save_actions", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--port", type=int, default=-1)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--libsumo", action="store_true")
    parser.add_argument("--sumo_no_random", action="store_true")
    parser.add_argument("--python_no_random", action="store_true")
    parser.add_argument("-e", "--error_output_path", type=str, default="error_message.log")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # ファイル出力ハンドラーの設定
    handler = logging.FileHandler(args.error_output_path)
    handler.setLevel(logging.DEBUG)
    # 出力フォーマットの設定
    formatter = logging.Formatter('%(levelname)s  %(asctime)s  [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    # ハンドラーの追加
    logger.addHandler(handler)
    # 重複出力をなくす
    logger.propagate = False

    if args.route_file == "":
        route_file = None
    else:
        route_file = args.route_file

    if args.reward_csv == "":
        reward_csv = args.run_name + "_" + arg.map_name + "_reward.csv"
    else:
        reward_csv = args.reward_csv
    
    if args.loss_csv == "":
        loss_csv = args.run_name + "_" + arg.map_name + "_loss.csv"
    else:
        loss_csv = args.loss_csv

    if args.embedding_start_train == 0:
        embedding_start_train = None
    else:
        embedding_start_train = args.embedding_start_train
        
    if args.port == -1:
        port = None
    else:
        port = args.port

    state_f = eval(args.state_f)
    reward_f = eval(args.reward_f)
    lights = eval(args.lights)

    try:
        if args.model_type == "original":
            train_agent(
                run_name=args.run_name, map_name=args.map_name, net_file=args.net_file, route_file=route_file, 
                state_f=state_f, reward_f=reward_f, model_save_path=args.model_save_path, 
                episode_per_learn=args.episode_per_learn, episodes=args.episodes, step_length=args.step_length, 
                yellow_length=args.yellow_length, step_ratio=args.step_ratio, end_time=args.end_time, 
                max_distance=args.max_distance, lights=lights, warmup=args.warmup, num_layers=1, 
                num_hidden_units=args.num_hidden_units, lr=args.lr, decay_rate=0.01, 
                temperature=args.temperature, noise=args.noise, 
                encoder_type=args.encoder_type, lstm_len=5, embedding_type=args.embedding_type, 
                embedding_num=args.embedding_num, embedding_decay=args.embedding_decay, eps=1e-5, 
                noisy_layer_num=args.noisy_layer_num, bbb_layer_num=args.bbb_layer_num, bbb_pi=args.bbb_pi, 
                beta=args.beta, 
                embedding_no_train=args.embedding_no_train, embedding_start_train=embedding_start_train, 
                log_dir=args.log_dir, env_base=args.env_base, reward_csv=reward_csv, loss_csv=loss_csv, 
                save_actions=args.save_actions, device=args.device, port=port, trial=args.trial, libsumo=args.libsumo
                )
        elif args.model_type == "PPO":
            train_PPO(
                run_name=args.run_name, map_name=args.map_name, net_file=args.net_file, route_file=route_file, 
                state_f=state_f, reward_f=reward_f, episode_per_learn=args.episode_per_learn, 
                episodes=args.episodes, step_length=args.step_length, yellow_length=args.yellow_length, 
                step_ratio=args.step_ratio, start_time=args.start_time, end_time=args.end_time, 
                max_distance=args.max_distance, lights=lights, warmup=args.warmup, num_layers=1, 
                num_hidden_units=args.num_hidden_units, lr=args.lr, decay_rate=0.01, 
                temperature=args.temperature, noise=args.noise, 
                encoder_type=args.encoder_type, lstm_len=5, embedding_type=args.embedding_type, 
                embedding_num=args.embedding_num, embedding_decay=args.embedding_decay, eps=1e-5, beta=args.beta, 
                update_interval=args.update_interval, minibatch_size=args.minibatch_size, epochs=args.epochs, 
                entropy_coef = args.entropy_coef, 
                embedding_no_train=args.embedding_no_train, embedding_start_train=embedding_start_train, 
                noisy_layer_type=args.noisy_layer_type, bbb_layer_type=args.bbb_layer_type, bbb_pi=args.bbb_pi, 
                bbb_sigma1=args.bbb_sigma1, bbb_sigma2=args.bbb_sigma2, 
                model_type=args.ppo_model_type, log_dir=args.log_dir, env_base=args.env_base, 
                reward_csv=reward_csv, loss_csv=loss_csv, save_actions=args.save_actions, 
                device=args.device, port=port, trial=args.trial, libsumo=args.libsumo, 
                sumo_no_random=args.sumo_no_random, python_no_random=args.python_no_random
                )
    except Exception as err:
        data_dir = args.log_dir + args.run_name + '-tr' + str(args.trial) + '-' + args.map_name + '-' + str(len(lights)) + '-' + state_f.__name__ + '-' + reward_f.__name__
        dir_content = subprocess.run(["find", data_dir, "-type", "f"], stdout=subprocess.PIPE, encoding="utf-8")
        file_num = subprocess.run(["wc", "-l"], input=dir_content.stdout, stdout=subprocess.PIPE, encoding="utf-8")
        episode_num = int(file_num.stdout)//2 + 1
        logger.debug("episode: " + str(episode_num))
        logger.exception("The program stopped because of this error.")
    finally:
        log_path = args.log_dir + args.run_name + "-tr1-" + args.map_name + "-" + str(len(lights)) + "-" + state_f.__name__ + "-" + reward_f.__name__
        subprocess.run(["rm", "-r", log_path])
