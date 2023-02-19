import os
import re
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np

if __name__ == "__main__":
    filename = "min_delay.txt"
    fig_dir = "./graph/"
    map_name = "cologne1"
    base = "base"
    delay_target_line = 55.07
    episode_per_learn = 1
    agent_name = "IPPO"
    exp_num = 5
    comparison = [
        "noise0.01", "noise0.1", "noise0.2", "temp1.5", "temp2.0", "temp3.0", 
        "bbb1", "noisy1"]

    fig_dict = dict()
    for x in comparison:
        name = re.sub(r"[.\d]", "", x)
        if name not in fig_dict.keys():
            fig_dict[name] = [x]
        else:
            fig_dict[name].append(x)

    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    min_delay_txt = ""
    
    base_reward_list = list()
    for i in range(exp_num):
        dir_name = base + "_" + str(i+1)
        reward_csv = "./" + dir_name + "/" + map_name + "_" + agent_name + "_" + dir_name + "_reward.csv"
        dataframe = pd.read_csv(reward_csv)
        reward = dataframe["mean_reward"].tolist()
        base_reward_list.append(reward)
    base_reward_mean = np.mean(base_reward_list, axis=0)
    base_reward_std = np.std(base_reward_list, axis=0)

    base_delay_list = list()
    for i in range(exp_num):
        dir_name = base + "_" + str(i+1)
        delay_py = "./" + dir_name + "/avg_timeLoss.py"
        with open(delay_py) as f:
            delay = eval(f.readlines()[0].split(":")[1])[0]
        delay_mean = list()
        for i in range(len(delay)//episode_per_learn):
            delay_mean.append(np.mean(delay[i*episode_per_learn:(i+1)*episode_per_learn]))
        base_delay_list.append(delay_mean)
    base_delay_mean = np.mean(base_delay_list, axis=0)
    base_delay_std = np.std(base_delay_list, axis=0)

    min_delay_txt += "base min: " + str(np.min(base_delay_mean)) + ", base std: " + str(base_delay_std[np.argmin(base_delay_mean)]) + "\n"
    min_delay_txt += "base min(average of the best performances): " + str(np.mean(np.min(base_delay_list, axis=1))) + ", base std(std of the best performances): " + str(np.std(np.min(base_delay_list, axis=1))) + "\n"
    min_delay_txt += "base min(the best performance): " + str(np.min(base_delay_list)) + "\n"
    
    for name in fig_dict.keys():
        #figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
        fig_reward = plt.figure(figsize=(5,5))
        fig_delay = plt.figure(figsize=(5,5))

        #add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
        ax_reward = fig_reward.add_subplot(1, 1, 1)
        ax_delay = fig_delay.add_subplot(1, 1, 1)
        
        reward_steps = list(range(1, len(base_reward_mean)+1))
        ax_reward.plot(reward_steps, base_reward_mean, label="baseline")
        ax_reward.fill_between(reward_steps, base_reward_mean-base_reward_std, base_reward_mean+base_reward_std, alpha=0.3)

        delay_steps = list(range(1, len(base_delay_mean)+1))
        ax_delay.plot(delay_steps, base_delay_mean, label="baseline")
        ax_delay.fill_between(delay_steps, base_delay_mean-base_delay_std, base_delay_mean+base_delay_std, alpha=0.3)
        
        for x in fig_dict[name]:
            x_reward_list = list()
            for i in range(exp_num):
                dir_name = x + "_" + str(i+1)
                reward_csv = "./" + dir_name + "/" + map_name + "_" + agent_name + "_" + dir_name + "_reward.csv"
                dataframe = pd.read_csv(reward_csv)
                reward = dataframe["mean_reward"].tolist()
                x_reward_list.append(reward)
            x_reward_mean = np.mean(x_reward_list, axis=0)
            x_reward_std = np.std(x_reward_list, axis=0)

            x_delay_list = list()
            for i in range(exp_num):
                dir_name = x + "_" + str(i+1)
                delay_py = "./" + dir_name + "/avg_timeLoss.py"
                with open(delay_py) as f:
                    delay = eval(f.readlines()[0].split(":")[1])[0]
                delay_mean = list()
                for i in range(len(delay)//episode_per_learn):
                    delay_mean.append(np.mean(delay[i*episode_per_learn:(i+1)*episode_per_learn]))
                x_delay_list.append(delay_mean)
            x_delay_mean = np.mean(x_delay_list, axis=0)
            x_delay_std = np.std(x_delay_list, axis=0)

            min_delay_txt += x + " min: " + str(np.min(x_delay_mean)) + ", " + x + " std: " + str(x_delay_std[np.argmin(x_delay_mean)]) + "\n"
            min_delay_txt += x + " min(average of the best performances): " + str(np.mean(np.min(x_delay_list, axis=1))) + ", " + x + " std(std of the best performances): " + str(np.std(np.min(x_delay_list, axis=1))) + "\n"
            min_delay_txt += x + " min(the best performance): " + str(np.min(x_delay_list)) + "\n"

            param = re.sub(r"[^.\d]", "", x)

            reward_steps = list(range(1, len(x_reward_mean)+1))
            ax_reward.plot(reward_steps, x_reward_mean, label=param)
            ax_reward.fill_between(reward_steps, x_reward_mean-x_reward_std, x_reward_mean+x_reward_std, alpha=0.3)

            delay_steps = list(range(1, len(x_delay_mean)+1))
            ax_delay.plot(delay_steps, x_delay_mean, label=param)
            ax_delay.fill_between(delay_steps, x_delay_mean-x_delay_std, x_delay_mean+x_delay_std, alpha=0.3)

        # 論文の値を記載
        if delay_target_line is not None:
            target_values = [delay_target_line for i in range(len(base_delay_mean))]
            delay_steps = list(range(1, len(base_delay_mean)+1))
            ax_delay.plot(delay_steps, target_values, label="targetline")
        
        # 凡例
        ax_reward.legend()
        # タイトル付け
        ax_reward.set_title("1episode内の平均報酬の変化")
        ax_reward.set_xlabel("episode(s)")
        ax_reward.set_ylabel("1episode内の平均報酬")

        # レイアウトの設定
        fig_reward.tight_layout()
        # save
        fig_reward.savefig(fig_dir + name + "_reward.png")

        # 凡例
        ax_delay.legend()
        # タイトル付け
        ax_delay.set_title("1episode内の平均delayの変化")
        ax_delay.set_xlabel("episode(s)")
        ax_delay.set_ylabel("Delay(s)")

        # レイアウトの設定
        fig_delay.tight_layout()
        # save
        fig_delay.savefig(fig_dir + name + "_delay.png")
    
        # グラフを消してメモリ開放
        plt.clf()
        plt.close(fig_reward)
        plt.close(fig_delay)
    
    with open(fig_dir + filename, "w", encoding="utf-8") as f:
        f.write(min_delay_txt)
    
