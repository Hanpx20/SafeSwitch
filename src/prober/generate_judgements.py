import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import argparse
import json
import os
from datetime import datetime
from tqdm import tqdm
from ..utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--judgement_file', type=str, default="/home/ph16/toxic_detection/sorry-bench/data/sorry_bench/model_judgment/ft-mistral-7b-instruct-v0.2/Qwen1.5-7B-Chat.jsonl")
    parser.add_argument('--question_file', type=str, default="/home/ph16/toxic_detection/sorry-bench/data/sorry_bench/mutations/question_v3.jsonl")
    parser.add_argument('--base_save_dir', type=str, default='/shared/nas2/ph16/toxic/outputs/states', help='Path to save the judgements')
    parser.add_argument('--job_name', type=str, default='llama3-1B')
    return parser.parse_args()

def transform_value(level, index):
    '''
    Used for differente granularities in sorry-bench
    Here we only do binary classification
    '''
    index = (int)(index)
    if index == 0:
        return 0
    
    if level == 2:
        return 1
    elif level == 5:
        cat = (index // 10) % 45 + 1
        if 1 <= cat <= 5:
            return 1
        elif 6 <= cat <= 25:
            return 2
        elif 26 <= cat <= 40:
            return 3
        else:
            return 4
    elif level == 46:
        return (index // 10) % 45 + 1


if __name__ == "__main__":
    '''
    Save judgements and safety labels as tensors
    '''
    args = parse_args()

    output_name = args.job_name
    mod = "".join(args.judgement_file.split('/')[-1].split('.')[:-1])
    if mod.find('_') != -1:
        output_name += '_' + mod.split('_')[1]
    
    save_dir = os.path.join(args.base_save_dir, output_name)
    os.system(f"mkdir -p {save_dir}")
    
    if os.path.isfile(os.path.join(save_dir, "judgements.pt")) and os.path.isfile(os.path.join(save_dir, "safety_labels.pt")):
        print("Judgements already generated.")
        exit(0)
    
    '''extract judgements'''
    judgements = []
    labels = []

    with open(args.judgement_file, "r") as file, open(args.question_file, "r") as file2:
        for line in file:
            labels.append((transform_value(2, json.loads(file2.readline())["category"])))
            judgements.append((int)(json.loads(line)["judgment"]))
    judgements = torch.tensor(judgements)
    labels = torch.tensor(labels)
    

    rate = torch.sum(judgements) / judgements.shape[0]
    print(f"Fulfillment rate is: {rate}")
    
    
    torch.save(judgements, os.path.join(save_dir, "judgements.pt"))
    torch.save(labels, os.path.join(save_dir, "safety_labels.pt"))
    