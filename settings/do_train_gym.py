from ast import parse
from RESCO.open_ai_gym_train import train_agent_gym, train_PPO_gym
import subprocess
import argparse
import logging

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="original")
    parser.add_argument("--ppo_model_type", type=str, default="original")
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--model_save_path", type=str, required=True)
    parser.add_argument("--episode_per_learn", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=1400)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_hidden_units", type=int, default=128)
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
    parser.add_argument("--noisy_layer_num", type=int, default=4)
    parser.add_argument("--bbb_layer_num", type=int, default=4)
    parser.add_argument("--bbb_pi", type=float, default=0.5)
    parser.add_argument("--log_dir", type=str, default="./")
    parser.add_argument("--learn_curve_csv", type=str, default="")
    parser.add_argument("--save_actions", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
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


    if args.learn_curve_csv == "":
        learn_curve_csv = args.env_name + "_learncurve.csv"
    else:
        learn_curve_csv = args.learn_curve_csv

    if args.embedding_start_train == 0:
        embedding_start_train = None
    else:
        embedding_start_train = args.embedding_start_train

    try:
        if args.model_type == "original":
            train_agent_gym(
                env_name=args.env_name, model_save_path=args.model_save_path, 
                episode_per_learn=args.episode_per_learn, episodes=args.episodes, max_steps=args.max_steps, num_layers=1, 
                num_hidden_units=args.num_hidden_units, lr=args.lr, decay_rate=0.01, temperature=args.temperature, noise=args.noise, 
                encoder_type=args.encoder_type, lstm_len=5, embedding_type=args.embedding_type, 
                embedding_num=args.embedding_num, embedding_decay=args.embedding_decay, eps=1e-5, beta=args.beta, 
                embedding_no_train=args.embedding_no_train, embedding_start_train=embedding_start_train, 
                log_dir=args.log_dir, learn_curve_csv=learn_curve_csv,
                save_actions=args.save_actions, device=args.device
                )
        elif args.model_type == "PPO":
            train_PPO_gym(
                env_name=args.env_name, 
                episode_per_learn=args.episode_per_learn, episodes=args.episodes, max_steps=args.max_steps, num_layers=1, 
                num_hidden_units=args.num_hidden_units, lr=args.lr, decay_rate=0.01, temperature=args.temperature, noise=args.noise, 
                encoder_type=args.encoder_type, lstm_len=5, embedding_type=args.embedding_type, 
                embedding_num=args.embedding_num, embedding_decay=args.embedding_decay, eps=1e-5, beta=args.beta, 
                update_interval=args.update_interval, minibatch_size=args.minibatch_size, epochs=args.epochs, 
                entropy_coef = args.entropy_coef,
                embedding_no_train=args.embedding_no_train, embedding_start_train=embedding_start_train, 
                noisy_layer_num=args.noisy_layer_num, bbb_layer_num=args.bbb_layer_num, bbb_pi=args.bbb_pi, 
                model_type=args.ppo_model_type, log_dir=args.log_dir, learn_curve_csv=learn_curve_csv,
                save_actions=args.save_actions, device=args.device
                )
    except Exception as err:
        logger.exception("The program stopped because of this error.")