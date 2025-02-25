import os
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import pickle
from sklearn.decomposition import PCA

map2trnum = {"cologne1": 1, "cologne3": 3, "cologne8": 8, "ingolstadt1": 1, "ingolstadt7": 7, "ingolstadt21": 21, "grid4x4": 16, "arterial4x4": 16}

if __name__ == "__main__":
    map_name = "cologne1"
    fig_dir = "./graph/"
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    
    exp_list = ["base", "noise0.01", "noise0.1", "noise0.2", "temp1.5", "temp2.0", "temp3.0", "vq3", "vq5", "vq7", "vq_onehot"]
    traffic_light_num = map2trnum[map_name]
    
    actions_vectors = list()
    delay_means_list = list()
    vec_num = 0
    for dir_name in exp_list:
        data_path = "./" + dir_name + "/actions_data.pkl"
        delay_path = "./" + dir_name + "/avg_timeLoss.py"

        with open(data_path, "rb") as f:
            actions_data = pickle.load(f)
        
        actions_data_flatten = np.asarray(actions_data).flatten()
        
        with open(delay_path) as f:
            delay = eval(f.readlines()[0].split(":")[1])[0]
        
        if map_name == "arterial4x4":
            steps_per_episode = 720
        else:
            steps_per_episode = 360
        
        delay_means = list()
        vec_num = len(actions_data_flatten)//(1024*traffic_light_num)
        for i in range(vec_num):
            actions_vector = actions_data_flatten[1024*traffic_light_num*i:1024*traffic_light_num*(i+1)]
            actions_vectors.append(actions_vector)

            first_episode = (1024*i) // steps_per_episode
            last_episode = (1024*(i+1)) // steps_per_episode
            first_episode_steps = steps_per_episode - ((1024*i) % steps_per_episode)
            last_episode_steps = (1024*(i+1)) % steps_per_episode

            delay_sum = delay[first_episode]*first_episode_steps + delay[last_episode]*last_episode_steps
            for j in range(first_episode+1, last_episode):
                delay_sum += delay[j]*steps_per_episode
            delay_means.append(delay_sum / 1024)
        delay_means_list.append(delay_means)

    pca = PCA(n_components=2)
    pca_vectors_list = pca.fit_transform(actions_vectors)    
    
    for i in range(len(exp_list)):
        dir_name = exp_list[i]
        delay_means = np.asarray(delay_means_list[i])
        delay_arg_sort = np.argsort(-delay_means)

        pca_vectors = pca_vectors_list[vec_num*i:vec_num*(i+1)]
        background_vectors = np.concatenate((pca_vectors_list[:vec_num*i], pca_vectors_list[vec_num*(i+1):]))

        time_fig_path = fig_dir + dir_name + "_PCA_time.png"
        delay_fig_path = fig_dir + dir_name + "_PCA_delay.png"

        #figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
        fig_pca_time = plt.figure(figsize=(5,5))
        fig_pca_delay = plt.figure(figsize=(5,5))

        #add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
        ax_pca_time = fig_pca_time.add_subplot(1, 1, 1)
        ax_pca_delay = fig_pca_delay.add_subplot(1, 1, 1)

        # 時系列プロット
        #ax_pca_time.plot(pca_vectors[:100, 0], pca_vectors[:100, 1], c="b", label="0~99")
        #ax_pca_time.plot(pca_vectors[99:200, 0], pca_vectors[99:200, 1], c="g", label="100~199")
        #ax_pca_time.plot(pca_vectors[199:300, 0], pca_vectors[199:300, 1], c="c", label="200~299")
        #ax_pca_time.plot(pca_vectors[299:400, 0], pca_vectors[299:400, 1], c="m", label="300~399")
        #ax_pca_time.plot(pca_vectors[399:, 0], pca_vectors[399:, 1], c="r", label="400~" + str(len(pca_vectors)))

        ax_pca_time.scatter(background_vectors[:, 0], background_vectors[:, 1], s=5, c="#808080", marker=".", label="other actions")
        ax_pca_time.scatter(pca_vectors[:100, 0], pca_vectors[:100, 1], s=5, c="b", marker=".", label="0～99")
        ax_pca_time.scatter(pca_vectors[100:200, 0], pca_vectors[100:200, 1], s=5, c="g", marker=".", label="100～199")
        ax_pca_time.scatter(pca_vectors[200:300, 0], pca_vectors[200:300, 1], s=5, color="c", marker=".", label="200～299")
        ax_pca_time.scatter(pca_vectors[300:400, 0], pca_vectors[300:400, 1], s=5, color="m", marker=".", label="300～399")
        ax_pca_time.scatter(pca_vectors[400:, 0], pca_vectors[400:, 1], s=5, c="r", marker=".", label="400～" + str(len(pca_vectors)))

        # delay順プロット
        pca_vectors_sort = pca_vectors[delay_arg_sort]
        delay_means_sort = delay_means[delay_arg_sort]

        #ax_pca_delay.plot(pca_vectors_sort[:100, 0], pca_vectors_sort[:100, 1], c="b", label=str(delay_means_sort[0]) + "~" + str(delay_means_sort[99]))
        #ax_pca_delay.plot(pca_vectors_sort[99:200, 0], pca_vectors_sort[99:200, 1], c="g", label=str(delay_means_sort[100]) + "~" + str(delay_means_sort[199]))
        #ax_pca_delay.plot(pca_vectors_sort[199:300, 0], pca_vectors_sort[199:300, 1], c="c", label=str(delay_means_sort[200]) + "~" + str(delay_means_sort[299]))
        #ax_pca_delay.plot(pca_vectors_sort[299:400, 0], pca_vectors_sort[299:400, 1], c="m", label=str(delay_means_sort[300]) + "~" + str(delay_means_sort[399]))
        #ax_pca_delay.plot(pca_vectors_sort[399:, 0], pca_vectors_sort[399:, 1], c="r", label=str(delay_means_sort[400]) + "~" + str(delay_means_sort[-1]))

        ax_pca_delay.scatter(background_vectors[:, 0], background_vectors[:, 1], s=5, c="#808080", marker=".", label="other actions")
        ax_pca_delay.scatter(pca_vectors_sort[:100, 0], pca_vectors_sort[:100, 1], s=5, c="b", marker=".", label=str(delay_means_sort[0]) + "～" + str(delay_means_sort[99]))
        ax_pca_delay.scatter(pca_vectors_sort[100:200, 0], pca_vectors_sort[100:200, 1], s=5, c="g", marker=".", label=str(delay_means_sort[100]) + "～" + str(delay_means_sort[199]))
        ax_pca_delay.scatter(pca_vectors_sort[200:300, 0], pca_vectors_sort[200:300, 1], s=5, color="c", marker=".", label=str(delay_means_sort[200]) + "～" + str(delay_means_sort[299]))
        ax_pca_delay.scatter(pca_vectors_sort[300:400, 0], pca_vectors_sort[300:400, 1], s=5, color="m", marker=".", label=str(delay_means_sort[300]) + "～" + str(delay_means_sort[399]))
        ax_pca_delay.scatter(pca_vectors_sort[400:, 0], pca_vectors_sort[400:, 1], s=5, c="r", marker=".", label=str(delay_means_sort[400]) + "～" + str(delay_means_sort[-1]))

        # タイトル付け
        ax_pca_time.set_title("学習回数による行動系列の変化")
        ax_pca_time.set_xlabel("第1主成分")
        ax_pca_time.set_ylabel("第2主成分")

        ax_pca_delay.set_title("delayによる行動系列の変化")
        ax_pca_delay.set_xlabel("第1主成分")
        ax_pca_delay.set_ylabel("第2主成分")

        # 凡例
        ax_pca_time.legend()
        ax_pca_delay.legend()

        # 図の調整
        fig_pca_time.tight_layout()
        fig_pca_delay.tight_layout()

        # 保存
        fig_pca_time.savefig(time_fig_path)
        fig_pca_delay.savefig(delay_fig_path)

        # グラフを消してメモリ開放
        plt.clf()
        plt.close(fig_pca_time)
        plt.close(fig_pca_delay)
