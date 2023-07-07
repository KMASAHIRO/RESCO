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
    
    exp_list = ["1e-3", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8"]
    
    # データ処理
    actions_vectors = list()
    reward_list = list()
    loss_list = list()
    vec_num = 0
    for dir_name in exp_list:
        data_path = "./" + dir_name + "/actions_data.pkl"
        csv_path = "./" + dir_name + "/" + env_name + "_" + dir_name + "_learncurve.csv"

        with open(data_path, "rb") as f:
            actions_data = pickle.load(f)
        
        actions_data_flatten = np.asarray(actions_data).flatten()
        
        tmp = pd.read_csv(csv_path)
        reward = tmp["mean_steps"].to_list()
        reward_list.append(reward)
        loss = tmp["loss"].to_list()
        loss_list.append(loss)

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

        ax_pca_time.scatter(background_vectors[:, 0], background_vectors[:, 1], s=5, c="#808080", marker=".", label="other actions")
        ax_pca_time.scatter(pca_vectors[:100, 0], pca_vectors[:100, 1], s=5, c="b", marker=".", label="0～99")
        ax_pca_time.scatter(pca_vectors[100:200, 0], pca_vectors[100:200, 1], s=5, c="g", marker=".", label="100～199")
        ax_pca_time.scatter(pca_vectors[200:, 0], pca_vectors[200:, 1], s=5, color="c", marker=".", label="200～" + str(len(pca_vectors)))

        # reward順プロット
        pca_vectors_sort = pca_vectors[reward_arg_sort]
        reward_means_sort = reward_means[reward_arg_sort]

        ax_pca_steps.scatter(background_vectors[:, 0], background_vectors[:, 1], s=5, c="#808080", marker=".", label="other actions")
        ax_pca_steps.scatter(pca_vectors_sort[:100, 0], pca_vectors_sort[:100, 1], s=5, c="b", marker=".", label=str(reward_means_sort[0]) + "～" + str(reward_means_sort[99]))
        ax_pca_steps.scatter(pca_vectors_sort[100:200, 0], pca_vectors_sort[100:200, 1], s=5, c="g", marker=".", label=str(reward_means_sort[100]) + "～" + str(reward_means_sort[199]))
        ax_pca_steps.scatter(pca_vectors_sort[200:, 0], pca_vectors_sort[200:, 1], s=5, color="c", marker=".", label=str(reward_means_sort[200]) + "～" + str(reward_means_sort[-1]))

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
    for i in range(len(exp_list)):
        dir_name = exp_list[i]
        reward = reward_list[i]
        reward_steps = list(range(1, len(reward)+1))

        #figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
        fig_reward = plt.figure(figsize=(5,5))
        #add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
        ax_reward = fig_reward.add_subplot(1, 1, 1)
        # plot
        ax_reward.plot(reward_steps, reward, label=dir_name)
        # 凡例
        ax_reward.legend()
        # タイトル付け
        ax_reward.set_title("学習回数による平均stepsの変化")
        ax_reward.set_xlabel("networkのupdate回数")
        ax_reward.set_ylabel("mean steps")

        # レイアウトの設定
        fig_reward.tight_layout()
        # save
        fig_reward.savefig(fig_dir + dir_name + "_steps.png")
    
        # グラフを消してメモリ開放
        plt.clf()
        plt.close(fig_reward)
    
    # lossのグラフ
    for i in range(len(exp_list)):
        dir_name = exp_list[i]
        loss = loss_list[i]
        loss_steps = list(range(1, len(loss)+1))

        #figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
        fig_loss = plt.figure(figsize=(5,5))
        #add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
        ax_loss = fig_loss.add_subplot(1, 1, 1)
        # plot
        ax_loss.plot(loss_steps, loss, label=dir_name)
        # 凡例
        ax_loss.legend()
        # タイトル付け
        ax_loss.set_title("学習回数によるlossの変化")
        ax_loss.set_xlabel("networkのupdate回数")
        ax_loss.set_ylabel("loss")

        # レイアウトの設定
        fig_loss.tight_layout()
        # save
        fig_loss.savefig(fig_dir + dir_name + "_loss.png")
    
        # グラフを消してメモリ開放
        plt.clf()
        plt.close(fig_loss)