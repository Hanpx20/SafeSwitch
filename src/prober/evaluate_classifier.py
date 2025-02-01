import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch.optim as optim
import argparse
import json
import os
from tqdm import tqdm
import wandb
from datetime import datetime
from ..utils import *
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from transformers import AdamW, get_scheduler

def parse_args():
    parser = argparse.ArgumentParser(description='Eval Hidden Classifier')
    parser.add_argument('--judges', type=str, nargs="+", required=True, help="Safety prober(s). If passing one argument then use direct prober, otherwise use two-stage prober")
    parser.add_argument('--two_stage', action="store_true")
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and testing')
    parser.add_argument('--base_dir', type=str, default='/shared/nas2/ph16/toxic/outputs/eval', help='Path to save the evaluation results')
    parser.add_argument('--data_dir', type=str, required=True, help='Path of eval data')
    parser.add_argument('--job_name', type=str, default='')
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--hidden_sizes', type=str, nargs='+', default=[], help='Intermediate sizes foe the MLP prober.')
    parser.add_argument('--token_rule', type=str, default="last", choices=['last', 'multi', "post"], help="last=last prefill token, multi=deocding phase, post=decode all answers")
    parser.add_argument('--n_decode', type=int, default=0, help="Number of decoded tokens before extracting internal states. If two-stage prober is used, this only applies for the second stage.")
    parser.add_argument('--layer_id', type=int, default=-1)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('-llm', type=str, default="")
    return parser.parse_args()


def evaluate_model_e2e(args, judge_1, judge_2, test_loader, threshold=0.5):
    label_list = []
    pred_list = []
    
    if judge_2 == None:
        judge_1.eval()
        with torch.no_grad():
            for (inputs, labels) in test_loader:
                outputs = judge_1(inputs)
                labels = labels.tolist()
                predicted = (outputs[:, 1] >= threshold).long().tolist()
                
                label_list.extend(labels)
                pred_list.extend(predicted)
    else:
        judge_1.eval()
        judge_2.eval()
        with torch.no_grad():
            for (inputs1, labels1), (inputs2, labels2) in test_loader:    
                labels = (labels1 & labels2).tolist()
                
                outputs1 = judge_1(inputs1)
                safety_label_predicted = (outputs1[:, 1] >= threshold).long()

                outputs2 = judge_2(inputs2)
                fulfill_label_predicted = (outputs2[:, 1] >= threshold).long()

                predicted = (safety_label_predicted & fulfill_label_predicted).tolist()
                
                label_list.extend(labels)
                pred_list.extend(predicted)

    return get_statistic(pred_list, label_list)

    
if __name__ == "__main__":
    '''initialization'''
    args = parse_args()
    if args.job_name == "N/A":
        args.job_name = ""
    if args.job_name != "":
        args.job_name = "_" + args.job_name

    assert (len(args.judges) == 2) == args.two_stage, "Provide judge_2 when and only when two_stage."

    if args.two_stage:
        id2 = '-'.join(args.judges[1].split('/')[-2:])
        args.job_name = "_" + id2 + args.job_name
    
    id1 = '-'.join(args.judges[0].split('/')[-2:])
    args.job_name = id1 + args.job_name

    if args.gpu != "":
        device = "cuda:" + args.gpu
    else:
        device = "cuda"
    
    print(f"Using device: {device}")

    '''load datasets'''
    if args.two_stage:
        test_dataset_1 = load_dataset(os.path.join(args.data_dir, "eval"), 
                                    device=device, 
                                    layer_id=args.layer_id, 
                                    token_rule=args.token_rule, 
                                    n_decode=0,
                                    label="safety"
                                    )
        test_dataset_2 = load_dataset(os.path.join(args.data_dir, "eval"), 
                                    device=device, 
                                    layer_id=args.layer_id, 
                                    token_rule=args.token_rule, 
                                    n_decode=args.n_decode,
                                    label="response"
                                    )
        test_loader = zip(DataLoader(test_dataset_1, batch_size=args.batch_size, shuffle=False), DataLoader(test_dataset_2, batch_size=args.batch_size, shuffle=False))
    else:
        test_dataset_1 = load_dataset(os.path.join(args.data_dir, "eval"), 
                                    device=device, 
                                    layer_id=args.layer_id, 
                                    token_rule=args.token_rule, 
                                    n_decode=args.n_decode,
                                    label="both"
                                    )
        test_loader = DataLoader(test_dataset_1, batch_size=args.batch_size, shuffle=False)


    '''load classifier model'''
    HIDDEN_STATE_DIM = test_dataset_1[0][0].shape[0]
    CLASSIFIER_DIM = 2
    
    
    judge_1 = load_classifier(args.judges[0], HIDDEN_STATE_DIM, CLASSIFIER_DIM, device=device)
    if args.two_stage:
        judge_2 = load_classifier(args.judges[1], HIDDEN_STATE_DIM, CLASSIFIER_DIM, device=device)
    else:
        judge_2 = None
    
    

    '''evaluation'''
    results = evaluate_model_e2e(args, judge_1, judge_2, test_loader, 
                             threshold=args.threshold)
        
    print(f"Overall Acc: {results['acc']}")
    print(f"Overall F1: {results['F1']}")

    '''save results'''
    output_dir = os.path.join(args.base_dir, args.job_name)
    
    if os.path.exists(output_dir):
        if args.overwrite:
            os.system(f"rm {output_dir}/*")
        else:
            output_dir += datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        os.system(f"mkdir -p {output_dir}")
    
    json.dump({key: results[key] for key in sorted(results)}, open(os.path.join(output_dir, f"result.json"), "w"), indent=4)    
    
    meta_dict = vars(args)
    json.dump(meta_dict, open(os.path.join(output_dir, "args.json"), "w"), indent=4)
    print(f"Result stored at {output_dir}")
    