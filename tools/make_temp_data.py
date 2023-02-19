import os
import numpy as np
import pickle

if __name__ == "__main__":
    output_dir = "./data/"
    filename = "temp_data.pkl"
    exp_list = [
        "temp0.0", "temp0.2", "temp0.4", "temp0.6", 
        "temp0.8", "temp1.2", "temp1.4", "temp1.6", 
        "temp1.8", "temp2.2", "temp2.4", "temp2.6", 
        "temp2.8", "temp3.2", "temp3.4", "temp3.6", 
        "temp3.8", "temp4.0", "temp4.2", "temp4.4", 
        "temp4.6", "temp4.8", "temp5.0", 
        "temp0.0_no_entropy", "temp0.2_no_entropy", "temp0.4_no_entropy", "temp0.6_no_entropy", 
        "temp0.8_no_entropy", "temp1.2_no_entropy", "temp1.4_no_entropy", "temp1.6_no_entropy", 
        "temp1.8_no_entropy", "temp2.2_no_entropy", "temp2.4_no_entropy", "temp2.6_no_entropy", 
        "temp2.8_no_entropy", "temp3.2_no_entropy", "temp3.4_no_entropy", "temp3.6_no_entropy", 
        "temp3.8_no_entropy", "temp4.0_no_entropy", "temp4.2_no_entropy", "temp4.4_no_entropy", 
        "temp4.6_no_entropy", "temp4.8_no_entropy", "temp5.0_no_entropy"]
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    output_data = {}
    for name in exp_list:
        path = "./" + name + "/avg_timeLoss.py"
        with open(path) as f:
            delay = eval(f.readlines()[0].split(":")[1])[0]
        output_data[name] = np.min(delay)
    
    with open(output_dir + filename, mode="wb") as f:
        pickle.dump(output_data, f)