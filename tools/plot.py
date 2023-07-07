import os
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np

if __name__ == "__main__":
    fig_dir = "./graph/"
    map_name = "cologne1"
    delay_target_line = 55.07
    episode_per_learn = 2
    agent_name = "IPPO"
    exp_list = ["base", "noise0.01", "noise0.1", "noise0.2", "temp1.5", "temp2", "temp3", "vq3", "vq5", "vq7", "vq_onehot"]
    
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    
    #figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
    fig_noise_reward = plt.figure(figsize=(5,5))
    fig_temp_reward = plt.figure(figsize=(5,5))
    fig_vq_reward = plt.figure(figsize=(5,5))
    fig_noise_delay = plt.figure(figsize=(5,5))
    fig_temp_delay = plt.figure(figsize=(5,5))
    fig_vq_delay = plt.figure(figsize=(5,5))

    #add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
    ax_noise_reward = fig_noise_reward.add_subplot(1, 1, 1)
    ax_temp_reward = fig_temp_reward.add_subplot(1, 1, 1)
    ax_vq_reward = fig_vq_reward.add_subplot(1, 1, 1)
    ax_noise_delay = fig_noise_delay.add_subplot(1, 1, 1)
    ax_temp_delay = fig_temp_delay.add_subplot(1, 1, 1)
    ax_vq_delay = fig_vq_delay.add_subplot(1, 1, 1)
    
    target_len = 0
    for dir_name in exp_list:
        reward_csv = "./" + dir_name + "/" + map_name + "_" + agent_name + "_" + dir_name + "_reward.csv"
        dataframe = pd.read_csv(reward_csv)
        reward = dataframe["mean_reward"].tolist()
        reward_steps = list(range(1, len(reward)+1))

        delay_py = "./" + dir_name + "/avg_timeLoss.py"
        with open(delay_py) as f:
            delay = eval(f.readlines()[0].split(":")[1])[0]
        delay_mean = list()
        for i in range(len(delay)//episode_per_learn):
            delay_mean.append(np.mean(delay[i*episode_per_learn:(i+1)*episode_per_learn]))
        delay_steps = list(range(1, len(delay_mean)+1))
        target_len = len(delay_mean)

        if dir_name == "base":
            ax_noise_reward.plot(reward_steps, reward, label="baseline(0.0)")
            ax_temp_reward.plot(reward_steps, reward, label="baseline(1.0)")
            ax_vq_reward.plot(reward_steps, reward, label="baseline(fc)")
            ax_noise_delay.plot(delay_steps, delay_mean, label="baseline(0.0)")
            ax_temp_delay.plot(delay_steps, delay_mean, label="baseline(1.0)")
            ax_vq_delay.plot(delay_steps, delay_mean, label="baseline(fc)")
        
        if "noise" in dir_name:
            label = dir_name.replace("noise", "")
            ax_noise_reward.plot(reward_steps, reward, label=label)
            ax_noise_delay.plot(delay_steps, delay_mean, label=label)

        if "temp" in dir_name:
            label = dir_name.replace("temp", "")
            ax_temp_reward.plot(reward_steps, reward, label=label)
            ax_temp_delay.plot(delay_steps, delay_mean, label=label)
        
        if "vq" in dir_name:
            label = dir_name.replace("vq", "")
            label = label.replace("_", "")
            ax_vq_reward.plot(reward_steps, reward, label=label)
            ax_vq_delay.plot(delay_steps, delay_mean, label=label)
    
    # 論文の値を記載
    target_values = [delay_target_line for i in range(target_len)]
    delays_steps = list(range(1, target_len+1))
    ax_noise_delay.plot(delay_steps, target_values, label="targetline")
    ax_temp_delay.plot(delay_steps, target_values, label="targetline")
    ax_vq_delay.plot(delay_steps, target_values, label="targetline")

    # 凡例
    ax_noise_reward.legend(loc = 'lower right')
    ax_temp_reward.legend(loc = 'lower right')
    ax_vq_reward.legend(loc = 'lower right')
    ax_noise_delay.legend(loc = 'upper right')
    ax_temp_delay.legend(loc = 'upper right')
    ax_vq_delay.legend(loc = 'upper right')

    # タイトル付け
    ax_noise_reward.set_title("中間層のnoiseによる変化")
    ax_noise_reward.set_xlabel("networkのupdate回数")
    ax_noise_reward.set_ylabel("reward")

    ax_temp_reward.set_title("温度パラメータによる変化")
    ax_temp_reward.set_xlabel("networkのupdate回数")
    ax_temp_reward.set_ylabel("reward")
    
    ax_vq_reward.set_title("VQネットワークにおける離散化ベクトルの数による変化")
    ax_vq_reward.set_xlabel("networkのupdate回数")
    ax_vq_reward.set_ylabel("reward")

    ax_noise_delay.set_title("中間層のnoiseによる変化")
    ax_noise_delay.set_xlabel("networkのupdate回数")
    ax_noise_delay.set_ylabel("delay(s)")

    ax_temp_delay.set_title("温度パラメータによる変化")
    ax_temp_delay.set_xlabel("networkのupdate回数")
    ax_temp_delay.set_ylabel("delay(s)")
    
    ax_vq_delay.set_title("VQネットワークにおける離散化ベクトルの数による変化")
    ax_vq_delay.set_xlabel("networkのupdate回数")
    ax_vq_delay.set_ylabel("delay(s)")

    # レイアウトの設定
    fig_noise_reward.tight_layout()
    fig_temp_reward.tight_layout()
    fig_vq_reward.tight_layout()
    fig_noise_delay.tight_layout()
    fig_temp_delay.tight_layout()
    fig_vq_delay.tight_layout()

    fig_noise_reward.savefig(fig_dir + "noise_reward.png")
    fig_temp_reward.savefig(fig_dir + "temp_reward.png")
    fig_vq_reward.savefig(fig_dir + "vq_reward.png")
    fig_noise_delay.savefig(fig_dir + "noise_delay.png")
    fig_temp_delay.savefig(fig_dir + "temp_delay.png")
    fig_vq_delay.savefig(fig_dir + "vq_delay.png")