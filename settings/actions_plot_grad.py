import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import japanize_matplotlib
import numpy as np
import pickle
from sklearn.decomposition import PCA

map2trnum = {"cologne1": 1, "cologne3": 3, "cologne8": 8, "ingolstadt1": 1, "ingolstadt7": 7, "ingolstadt21": 21, "grid4x4": 16, "arterial4x4": 16}
clip_range = {
    "cologne1": [20,250], 
    "cologne3": [20, 200], 
    "cologne8": [20, 100], 
    "ingolstadt1": [15, 40], 
    "ingolstadt7": [30, 50], 
    "ingolstadt21": [50, 200], 
    "grid4x4": [40, 100], 
    "arterial4x4": [500, 1000]
    }

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
    
    # logスケールでdelay順でプロット
    delay_means = np.concatenate(delay_means_list)
    delay_log_norm = np.log(delay_means)/np.max(np.log(delay_means))
    delay_fig_path = "PCA_delay_log.png"
    fig_pca_delay = plt.figure(figsize=(5,5))
    ax_pca_delay = fig_pca_delay.add_subplot(1, 1, 1)
    ax_pca_delay.scatter(pca_vectors_list[:, 0], pca_vectors_list[:, 1], s=5, c=cm.jet_r(delay_log_norm), marker=".", label="delay(log)")
    ax_pca_delay.set_title("delayによる行動系列の変化(logスケール)")
    ax_pca_delay.set_xlabel("第1主成分")
    ax_pca_delay.set_ylabel("第2主成分")
    ax_pca_delay.legend()
    fig_pca_delay.tight_layout()
    fig_pca_delay.savefig(delay_fig_path)
    # グラフを消してメモリ開放
    plt.clf()
    plt.close(fig_pca_delay)

    # clippingでdelay順でプロット
    delay_means = np.concatenate(delay_means_list)
    delay_means_clipped = (delay_means - clip_range[map_name][0])/clip_range[map_name][1]
    delay_fig_path = "PCA_delay_clipped.png"
    fig_pca_delay = plt.figure(figsize=(5,5))
    ax_pca_delay = fig_pca_delay.add_subplot(1, 1, 1)
    ax_pca_delay.scatter(pca_vectors_list[delay_means_clipped>1.0, 0], pca_vectors_list[delay_means_clipped>1.0, 1], s=5, c="#808080", marker=".", label="other actions")
    ax_pca_delay.scatter(pca_vectors_list[delay_means_clipped<=1.0, 0], pca_vectors_list[delay_means_clipped<=1.0, 1], s=5, c=cm.jet_r(delay_means_clipped[delay_means_clipped<=1.0]), marker=".", label="delay(clipped)")
    ax_pca_delay.set_title("delayによる行動系列の変化(clipping)")
    ax_pca_delay.set_xlabel("第1主成分")
    ax_pca_delay.set_ylabel("第2主成分")
    ax_pca_delay.legend()
    fig_pca_delay.tight_layout()
    fig_pca_delay.savefig(delay_fig_path)
    # グラフを消してメモリ開放
    plt.clf()
    plt.close(fig_pca_delay)
    
    for i in range(len(exp_list)):
        dir_name = exp_list[i]

        pca_vectors = pca_vectors_list[vec_num*i:vec_num*(i+1)]
        background_vectors = np.concatenate((pca_vectors_list[:vec_num*i], pca_vectors_list[vec_num*(i+1):]))

        time_fig_path = fig_dir + dir_name + "_PCA_time_grad.png"

        #figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
        fig_pca_time = plt.figure(figsize=(5,5))

        #add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
        ax_pca_time = fig_pca_time.add_subplot(1, 1, 1)

        # 時系列プロット
        ax_pca_time.scatter(background_vectors[:, 0], background_vectors[:, 1], s=5, c="#808080", marker=".", label="other actions")
        ax_pca_time.scatter(pca_vectors[:, 0], pca_vectors[:, 1], s=5, c=list(range(len(pca_vectors))), cmap="jet", marker=".", label="chronological")
        ax_pca_time.scatter(pca_vectors[:, 0], pca_vectors[:, 1], s=5, c=cm.jet_r(np.arange(0, len(pca_vectors), 1)/(len(pca_vectors)-1)), marker=".", label="chronological")

        # タイトル付け
        ax_pca_time.set_title("学習回数による行動系列の変化")
        ax_pca_time.set_xlabel("第1主成分")
        ax_pca_time.set_ylabel("第2主成分")

        # 凡例
        ax_pca_time.legend()

        # 図の調整
        fig_pca_time.tight_layout()

        # 保存
        fig_pca_time.savefig(time_fig_path)

        # グラフを消してメモリ開放
        plt.clf()
        plt.close(fig_pca_time)
