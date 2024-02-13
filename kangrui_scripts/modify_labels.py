import numpy as np
import os
import glob
from datasets import load_from_disk
from datasets import load_from_disk,Dataset
import fire
import shutil


def convert_to_false_only(dataset):
    update=[]
    for dp in dataset:
        if not dp['acc']:
            update.append(dp)
    dataset = Dataset.from_list(update)
    return dataset

def convert_to_false_only_flipped(dataset):
    update=[]
    for dp in dataset:
        if dp['acc']:
            continue
        else:
            ndp=dp.copy()
            ndp['soft_label']=[dp['soft_label'][1], dp['soft_label'][0]]
            ndp['hard_label']=1-ndp['gt_label']
            update.append(ndp)
    dataset = Dataset.from_list(update)
    return dataset

def convert_to_true_only(dataset):
    update=[]
    for dp in dataset:
        if dp['acc']:
            update.append(dp)
    dataset = Dataset.from_list(update)
    return dataset

def convert_to_true_gt(dataset):
    update=[]
    for dp in dataset:
        if dp['acc']:
            update.append(dp)
        else:
            ndp=dp.copy()
            ndp['soft_label']=[1 - float(ndp['gt_label']), float(ndp['gt_label'])]
            ndp['hard_label']=ndp['gt_label']
            update.append(ndp)
    dataset = Dataset.from_list(update)
    return dataset

def convert_to_false_gt(dataset):
    update=[]
    for dp in dataset:
        if not dp['acc']:
            update.append(dp)
        else:
            ndp=dp.copy()
            ndp['soft_label']=[1 - float(ndp['gt_label']), float(ndp['gt_label'])]
            ndp['hard_label']=ndp['gt_label']
            update.append(ndp)
    dataset = Dataset.from_list(update)
    return dataset

def convert_to_all_false(dataset):
    update=[]
    for dp in dataset:
        if not dp['acc']:
            update.append(dp)
        else:
            ndp=dp.copy()
            ndp['soft_label']=[float(ndp['gt_label']), 1-float(ndp['gt_label'])]
            ndp['hard_label']=1-ndp['gt_label']
            update.append(ndp)
    dataset = Dataset.from_list(update)
    return dataset


Func={
    "convert_to_false_only":convert_to_false_only,
    "convert_to_false_only_flipped":convert_to_false_only_flipped,
    "convert_to_true_only":convert_to_true_only,
    "convert_to_true_gt":convert_to_true_gt,
    "convert_to_all_false":convert_to_all_false,
    "convert_to_false_gt":convert_to_false_gt
}
def main(
        result_path,
        save_path,
        mode,
):
    print(f"result_path: {result_path}")
    print(f"save_path: {save_path}")
    print(f"mode: {mode}")
    for result_filename in glob.glob(os.path.join(result_path,"**/dataset_info.json"), recursive=True):
        dataset_name="/".join(result_filename.split("/")[:-1])
        datas=load_from_disk(dataset_name)
        updated_datas=Func[mode](datas)
        updated_datas.save_to_disk(dataset_name.replace(result_path,save_path))
    for result_filename in glob.glob(os.path.join(result_path,"**/dataset_info.json"), recursive=True):
        
        folder_name="/".join(result_filename.split("/")[:-2])
        files = glob.glob(os.path.join(folder_name, '*'))
        for file in files:
            if file.endswith("safetensors"):
                continue
            if file.endswith("weak_labels"):
                continue
            shutil.copy(file, file.replace(result_path,save_path))
    
if __name__ == "__main__":
    fire.Fire(main)