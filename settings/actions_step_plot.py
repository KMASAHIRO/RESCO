import os
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import pickle

map2trnum = {"cologne1": 1, "cologne3": 3, "cologne8": 8, "ingolstadt1": 1, "ingolstadt7": 7, "ingolstadt21": 21, "grid4x4": 16, "arterial4x4": 16}


if __name__ == "__main__":
    map_name = "cologne1"
    fig_dir = "./graph/"
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    
    exp_list = ["base", "noise0.01", "noise0.1", "noise0.2", "temp1.5", "temp2.0", "temp3.0", "vq3", "vq5", "vq7", "vq_onehot"]
    traffic_light_num = map2trnum[map_name]

    for dir_name in exp_list:
        data_path = "./" + dir_name + "/actions_data.pkl"
        delay_path = "./" + dir_name + "/avg_timeLoss.py"

        with open(data_path, "rb") as f:
            actions_data = pickle.load(f)
        
        actions_data = np.asarray(actions_data)
        
        with open(delay_path) as f:
            delay = eval(f.readlines()[0].split(":")[1])[0]
        
        if map_name == "arterial4x4":
            steps_per_episode = 720
        else:
            steps_per_episode = 360
        

        x = np.arange(-6,7,1)
        y = 1/(1+np.exp(x))

        fig, ax = plt.subplots(nrows=5, ncols=3,sharex=True,sharey=True) 
        ax = ax.ravel()

        # 最初の5件
        for i in range(5):
            for j in range(traffic_light_num):
                y = [actions_data[i][k*traffic_light_num + j] for k in range(steps_per_episode)]
                ax[i*3].step(list(range(1, steps_per_episode+1)), y, "C" + str(j+1) + "-")
            ax[i*3].set_title("delay="+str(delay[i]))
        
        # delay最小5件
        delay_arg = np.argsort(delay)[:5]
        for i in range(5):
            for j in range(traffic_light_num):
                y = [actions_data[delay_arg[i]][k*traffic_light_num + j] for k in range(steps_per_episode)]
                ax[i*3+1].step(list(range(1, steps_per_episode+1)), y, "C" + str(j+1) + "-")
            ax[i*3+1].set_title("delay="+str(delay[delay_arg[i]]))
        
        # 最後の5件
        for i in range(5):
            for j in range(traffic_light_num):
                y = [actions_data[-5+i][k*traffic_light_num + j] for k in range(steps_per_episode)]
                ax[i*3+2].step(list(range(1, steps_per_episode+1)), y, "C" + str(j+1) + "-")
            ax[i*3+2].set_title("delay="+str(delay[-5+i]))

        plt.suptitle('episode内のaction')
        plt.tight_layout()
        plt.savefig("actions_steps_" + dir_name + ".png")