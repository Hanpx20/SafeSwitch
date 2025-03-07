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
    parser = argparse.ArgumentParser(description='Train Hidden Classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate for the optimizer')
    parser.add_argument('--base_dir', type=str, default='/shared/nas2/ph16/toxic/outputs/classifier', help='Path to save the trained model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path of training data')
    parser.add_argument('--job_name', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help="whether to overwrite the original output directory if exists") 
    parser.add_argument('--hidden_sizes', type=str, nargs='+', default=[], help='')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--token_rule', type=str, default="last", choices=['last', 'multi', "post"])
    parser.add_argument('--label', type=str, default="both", choices=['safety', 'response', "both"], help="both for direct probers, for two-dtage probers, please train two separately")
    parser.add_argument('--n_decode', type=int, default=0, help="Number of decoded tokens before extracting internal states. If two-stage prober is used, this only applies for the second stage.")
    parser.add_argument('--layer_id', type=int, default=-1)
    parser.add_argument('--neg_weight', type=float, default=1.0, help="Weight for negative samples")
    parser.add_argument('--llm', type=str, required=True)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(args, model, train_loader, criterion, optimizer, num_epochs=10, wandb_name=None):
    print("Training begins!")
    if wandb_name:
        wandb.init(project="DeToxic", name=wandb_name)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = train_loader
        
        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if wandb_name:
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({"epoch": epoch + 1, "loss": loss.item(), "learning_rate": current_lr})
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch: {epoch + 1} Loss: {epoch_loss:.4f}")


def evaluate_model(args, model, test_loader, threshold=0.5):
    '''
    we provide basic evaluation for the prober, but for evaluating two-stages probers, use evaluate_classifier.py
    '''
    model.eval()
    
    label_list = []
    pred_list = []
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            outputs = model(inputs)
            labels = labels.tolist()
            predicted = (outputs[:, 1] >= threshold).long().tolist()
            
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
        
    args.job_name = "_" + args.label + args.job_name
    
    if args.token_rule == "multi":
        args.job_name = "_multi" + args.job_name
        args.job_name = args.llm + args.job_name + "/token" + (str)(args.n_decode)
    else:
        args.job_name = args.llm + args.job_name + "/layer" + (str)(args.layer_id)


        
    if args.gpu != "":
        device = "cuda:" + args.gpu
    else:
        device = "cuda"
    
    print(f"Using device: {device}")
    set_seed(args.random_seed)


    '''load datasets'''
    train_dataset = load_dataset(os.path.join(args.data_dir, "train"), 
                                 device=device, 
                                 layer_id=args.layer_id, 
                                 token_rule=args.token_rule, 
                                 n_decode=args.n_decode,
                                 label=args.label
                                 )
    test_dataset = load_dataset(os.path.join(args.data_dir, "eval"), 
                                 device=device, 
                                 layer_id=args.layer_id, 
                                 token_rule=args.token_rule, 
                                 n_decode=args.n_decode,
                                 label=args.label
                                 )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    '''build classifier model'''
    HIDDEN_STATE_DIM = test_dataset[0][0].shape[0]
    CLASSIFIER_DIM = 2
    dtype=test_dataset[0][0].dtype
    
    hidden_sizes = [(int)(x) for x in args.hidden_sizes]
    hidden_sizes.insert(0, HIDDEN_STATE_DIM)
    hidden_sizes.append(CLASSIFIER_DIM)
        
    if args.ckpt != "":
        model = load_classifier(args.ckpt, HIDDEN_STATE_DIM, CLASSIFIER_DIM, device=device)
    else:
        model = LinearProber(hidden_sizes=hidden_sizes).to(device)
    # model.to(dtype=dtype)
    
    
    '''training'''
    if args.neg_weight >= 1:
        weight = torch.tensor([args.neg_weight, 1.0]).to(device)
    else:
        weight = torch.tensor([1.0, 1.0 / args.neg_weight]).to(device)

    criterion = nn.CrossEntropyLoss(weight)
    total_steps = len(train_loader) * args.epochs
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)


    train_model(args, model, train_loader, criterion, optimizer, 
                num_epochs=args.epochs, 
                wandb_name=args.job_name if args.wandb else None)



    '''evaluation'''

    results = evaluate_model(args, model, test_loader, 
                            threshold=0.5)


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
    
    torch.save(model.state_dict(), os.path.join(output_dir, "model_weights.pth"))
