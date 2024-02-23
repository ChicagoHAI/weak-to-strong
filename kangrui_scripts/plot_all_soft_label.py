import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from IPython.display import display
from collections import defaultdict
import os
import glob
import json
import pickle
import fire
import multiprocessing
from datasets import load_from_disk

def plot_figure(datas,model_name):
    # Extracting the data
    x_true, y_true = [], []
    x_false, y_false = [], []

    for data in datas:
        x, y = data['soft_label']
        if data['acc']:
            x_true.append(x)
            y_true.append(y)
        else:
            x_false.append(x)
            y_false.append(y)

    max_values_true = [max(x, y) for x, y in zip(x_true, y_true)]
    max_values_false = [max(x, y) for x, y in zip(x_false, y_false)]

    # Plotting the density of the max values
    plt.figure(figsize=(8, 6))

    # Density plot for True
    plt.hist(max_values_true, bins=30, alpha=0.5, color='blue', label='True', density=False)

    # Density plot for False
    plt.hist(max_values_false, bins=30, alpha=0.5, color='red', label='False', density=False)

    plt.title(f'{model_name}: Maximum Dimension Values by Acc')
    plt.xlabel('Max Dimension Value')
    plt.ylabel('Count')
    plt.legend()
    return plt

def plot_dir(rst_path,
              models_to_plot=["openai-gpt","gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl","Qwen-7B","Llama-2-7b-chat-hf"],
              ):
    name_to_path = dict()
    for result_filename in glob.glob(os.path.join(rst_path,"**/dataset_info.json"), recursive=True):
        dataset_name="/".join(result_filename.split("/")[:-1])
        for name in models_to_plot:
            if (name+'-nd').lower() in dataset_name.lower():
                name_to_path[name] = dataset_name
                break
    
    for model_name,path in name_to_path.items():
        datas=load_from_disk(path)
        plt=plot_figure(datas,model_name)
        plt.savefig(f"{rst_path}/{model_name}_distribution.png",bbox_inches='tight')
        print(f"{rst_path}/{model_name}_distribution.png finished")
        
def main(rst_dir="./results/", 
         sub_folders=["boolq","cosmos_qa","recidivism","sciq","boolq_only_topk","cosmos_qa_only_topk","recidivism_only_topk","sciq_only_topk"],):
    # find all directories in rst_dir

    processes = []

    
    for d in os.listdir(rst_dir):
        if os.path.isdir(os.path.join(rst_dir, d)):
            # run multiprocessing
            rst_path=os.path.join(rst_dir, d)
            sub_folder=rst_path.split("/")[-1]
            if sub_folders and sub_folder not in sub_folders:
                continue
            print(sub_folder)
            process = multiprocessing.Process(target=plot_dir, args=(rst_path,))
            processes.append(process)
            process.start()

     # Wait for all processes to finish
    for process in processes:
        process.join()

    print("All processes have completed.")

if __name__ == "__main__":
    fire.Fire(main)