import os
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA

if __name__ == "__main__":
    env_name = "CartPole-v1"
    steps_per_learn = 2500
    fig_dir = "./graph/"
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    
    exp_list = ["base", "noise0.01", "noise0.1", "noise0.2", "temp1.5", "temp2.0", "temp3.0", "vq3", "vq5", "vq7", "vq_onehot"]
    
    # データ処理
    actions_vectors = list()
    reward_list = list()
    vec_num = 0
    for dir_name in exp_list:
        data_path = "./" + dir_name + "/actions_data.pkl"
        csv_path = "./" + dir_name + "/" + env_name + "_" + dir_name + "_reward.csv"

        with open(data_path, "rb") as f:
            actions_data = pickle.load(f)
        
        actions_data_flatten = np.asarray(actions_data).flatten()
        
        tmp = pd.read_csv(csv_path)
        reward = tmp["mean_steps"].to_list()
        reward_list.append(reward)

        vec_num = len(actions_data_flatten)//(steps_per_learn)
        for i in range(vec_num):
            actions_vector = actions_data_flatten[steps_per_learn*i:steps_per_learn*(i+1)]
            actions_vectors.append(actions_vector)

    pca = PCA(n_components=2)
    pca_vectors_list = pca.fit_transform(actions_vectors)    
    
    # 行動系列可視化
    for i in range(len(exp_list)):
        dir_name = exp_list[i]
        reward_means = np.asarray(reward_list[i])

        if env_name == "MountainCar-v0":
            reward_arg_sort = np.argsort(-reward_means)
        elif env_name == "CartPole-v1":
            reward_arg_sort = np.argsort(reward_means)

        pca_vectors = pca_vectors_list[vec_num*i:vec_num*(i+1)]
        background_vectors = np.concatenate((pca_vectors_list[:vec_num*i], pca_vectors_list[vec_num*(i+1):]))

        time_fig_path = fig_dir + dir_name + "_PCA_time.png"
        reward_fig_path = fig_dir + dir_name + "_PCA_steps.png"

        #figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
        fig_pca_time = plt.figure(figsize=(5,5))
        fig_pca_steps = plt.figure(figsize=(5,5))

        #add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
        ax_pca_time = fig_pca_time.add_subplot(1, 1, 1)
        ax_pca_steps = fig_pca_steps.add_subplot(1, 1, 1)

        # 時系列プロット
        #ax_pca_time.plot(pca_vectors[:100, 0], pca_vectors[:100, 1], c="b", label="0~99")
        #ax_pca_time.plot(pca_vectors[99:200, 0], pca_vectors[99:200, 1], c="g", label="100~199")
        #ax_pca_time.plot(pca_vectors[199:300, 0], pca_vectors[199:300, 1], c="c", label="200~299")
        #ax_pca_time.plot(pca_vectors[299:400, 0], pca_vectors[299:400, 1], c="m", label="300~399")
        #ax_pca_time.plot(pca_vectors[399:, 0], pca_vectors[399:, 1], c="r", label="400~" + str(len(pca_vectors)))

        ax_pca_time.scatter(background_vectors[:, 0], background_vectors[:, 1], s=5, c="#808080", marker=".", label="other actions")
        ax_pca_time.scatter(pca_vectors[:100, 0], pca_vectors[:100, 1], s=5, c="b", marker=".", label="0～99")
        ax_pca_time.scatter(pca_vectors[100:200, 0], pca_vectors[100:200, 1], s=5, c="g", marker=".", label="100～199")
        ax_pca_time.scatter(pca_vectors[200:, 0], pca_vectors[200:, 1], s=5, color="c", marker=".", label="200～" + str(len(pca_vectors)))
        #ax_pca_time.scatter(pca_vectors[300:400, 0], pca_vectors[300:400, 1], s=5, color="m", marker=".", label="300～399")
        #ax_pca_time.scatter(pca_vectors[400:, 0], pca_vectors[400:, 1], s=5, c="r", marker=".", label="400～" + str(len(pca_vectors)))

        # delay順プロット
        pca_vectors_sort = pca_vectors[reward_arg_sort]
        reward_means_sort = reward_means[reward_arg_sort]

        #ax_pca_delay.plot(pca_vectors_sort[:100, 0], pca_vectors_sort[:100, 1], c="b", label=str(delay_means_sort[0]) + "~" + str(delay_means_sort[99]))
        #ax_pca_delay.plot(pca_vectors_sort[99:200, 0], pca_vectors_sort[99:200, 1], c="g", label=str(delay_means_sort[100]) + "~" + str(delay_means_sort[199]))
        #ax_pca_delay.plot(pca_vectors_sort[199:300, 0], pca_vectors_sort[199:300, 1], c="c", label=str(delay_means_sort[200]) + "~" + str(delay_means_sort[299]))
        #ax_pca_delay.plot(pca_vectors_sort[299:400, 0], pca_vectors_sort[299:400, 1], c="m", label=str(delay_means_sort[300]) + "~" + str(delay_means_sort[399]))
        #ax_pca_delay.plot(pca_vectors_sort[399:, 0], pca_vectors_sort[399:, 1], c="r", label=str(delay_means_sort[400]) + "~" + str(delay_means_sort[-1]))

        ax_pca_steps.scatter(background_vectors[:, 0], background_vectors[:, 1], s=5, c="#808080", marker=".", label="other actions")
        ax_pca_steps.scatter(pca_vectors_sort[:100, 0], pca_vectors_sort[:100, 1], s=5, c="b", marker=".", label=str(reward_means_sort[0]) + "～" + str(reward_means_sort[99]))
        ax_pca_steps.scatter(pca_vectors_sort[100:200, 0], pca_vectors_sort[100:200, 1], s=5, c="g", marker=".", label=str(reward_means_sort[100]) + "～" + str(reward_means_sort[199]))
        ax_pca_steps.scatter(pca_vectors_sort[200:, 0], pca_vectors_sort[200:, 1], s=5, color="c", marker=".", label=str(reward_means_sort[200]) + "～" + str(reward_means_sort[-1]))
        #ax_pca_steps.scatter(pca_vectors_sort[300:400, 0], pca_vectors_sort[300:400, 1], s=5, color="m", marker=".", label=str(reward_means_sort[300]) + "～" + str(reward_means_sort[399]))
        #ax_pca_steps.scatter(pca_vectors_sort[400:, 0], pca_vectors_sort[400:, 1], s=5, c="r", marker=".", label=str(reward_means_sort[400]) + "～" + str(reward_means_sort[-1]))

        # タイトル付け
        ax_pca_time.set_title("学習回数による行動系列の変化")
        ax_pca_time.set_xlabel("第1主成分")
        ax_pca_time.set_ylabel("第2主成分")

        ax_pca_steps.set_title("有効step数による行動系列の変化")
        ax_pca_steps.set_xlabel("第1主成分")
        ax_pca_steps.set_ylabel("第2主成分")

        # 凡例
        ax_pca_time.legend()
        ax_pca_steps.legend()

        # 図の調整
        fig_pca_time.tight_layout()
        fig_pca_steps.tight_layout()

        # 保存
        fig_pca_time.savefig(time_fig_path)
        fig_pca_steps.savefig(reward_fig_path)

        # グラフを消してメモリ開放
        plt.clf()
        plt.close(fig_pca_time)
        plt.close(fig_pca_steps)

    # step数のグラフ
    #figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
    fig_noise_reward = plt.figure(figsize=(5,5))
    fig_temp_reward = plt.figure(figsize=(5,5))
    fig_vq_reward = plt.figure(figsize=(5,5))
    
    #add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
    ax_noise_reward = fig_noise_reward.add_subplot(1, 1, 1)
    ax_temp_reward = fig_temp_reward.add_subplot(1, 1, 1)
    ax_vq_reward = fig_vq_reward.add_subplot(1, 1, 1)
    
    target_len = 0
    for i in range(len(exp_list)):
        dir_name = exp_list[i]
        reward = reward_list[i]
        reward_steps = list(range(1, len(reward)+1))

        if dir_name == "base":
            ax_noise_reward.plot(reward_steps, reward, label="baseline(0.0)")
            ax_temp_reward.plot(reward_steps, reward, label="baseline(1.0)")
            ax_vq_reward.plot(reward_steps, reward, label="baseline(fc)")
        
        if "noise" in dir_name:
            label = dir_name.replace("noise", "")
            ax_noise_reward.plot(reward_steps, reward, label=label)

        if "temp" in dir_name:
            label = dir_name.replace("temp", "")
            ax_temp_reward.plot(reward_steps, reward, label=label)
        
        if "vq" in dir_name:
            label = dir_name.replace("vq", "")
            label = label.replace("_", "")
            ax_vq_reward.plot(reward_steps, reward, label=label)

    # 凡例
    ax_noise_reward.legend()
    ax_temp_reward.legend()
    ax_vq_reward.legend()

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

    # レイアウトの設定
    fig_noise_reward.tight_layout()
    fig_temp_reward.tight_layout()
    fig_vq_reward.tight_layout()

    fig_noise_reward.savefig(fig_dir + "noise_steps.png")
    fig_temp_reward.savefig(fig_dir + "temp_steps.png")
    fig_vq_reward.savefig(fig_dir + "vq_steps.png")

    # グラフを消してメモリ開放
    plt.clf()
    plt.close(fig_noise_reward)
    plt.close(fig_temp_reward)
    plt.close(fig_vq_reward)