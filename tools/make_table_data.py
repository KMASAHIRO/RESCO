import os
import numpy as np
import pickle

if __name__ == "__main__":
    output_dir = "./data/"
    filename = "table_data.pkl"
    exp_num = 5
    exp_list = [
        "base", "noise0.01", "noise0.1", "noise0.2", "temp1.5", "temp2.0", "temp3.0", 
        "bbb1", "noisy1", 
        "base_no_entropy", "noise0.01_no_entropy", "noise0.1_no_entropy", "noise0.2_no_entropy", "temp1.5_no_entropy", "temp2.0_no_entropy", "temp3.0_no_entropy", 
        "bbb1_no_entropy", "noisy1_no_entropy"]
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    output_data = {}
    for name in exp_list:
        delay_list = []
        for i in range(exp_num):
            path = "./" + name + "_" + str(i+1) + "/avg_timeLoss.py"
            with open(path) as f:
                delay = eval(f.readlines()[0].split(":")[1])[0]
            delay_list.append(delay)
        data = {}
        data["平均曲線の最低値"] = np.min(np.mean(delay_list, axis=0))
        data["最低値の平均"] = np.mean(np.min(delay_list, axis=1))
        data["最低値の標準偏差"] = np.std(np.min(delay_list, axis=1))
        data["最低値"] = np.min(delay_list)
        output_data[name] = data
    
    with open(output_dir + filename, mode="wb") as f:
        pickle.dump(output_data, f)