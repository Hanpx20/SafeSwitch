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

HELPFUL_PROMPT = '''You are a helpful assistant. You should always respond to the user's query to the best of your ability.'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--question_file', type=str, required=True)
    parser.add_argument('--answer_file', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--base_save_dir', type=str, default='/shared/nas2/ph16/toxic/outputs/states', help='Path to save the states')
    parser.add_argument('--job_name', type=str, default='')
    parser.add_argument('--token_rule', type=str, default="last")
    parser.add_argument('--layer_id', type=str, default="all")
    parser.add_argument('--do_decoding', action="store_true")
    parser.add_argument('--decoding_all', action="store_true")
    parser.add_argument('--raw', action="store_true", help="Not using system prompt")
    
    return parser.parse_args()

def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions

def batch_read(lst, bsz):
    for i in tqdm(range(0, len(lst), bsz)):
        yield lst[i:i+bsz]

if __name__ == "__main__":
    '''
    Generate Hidden States for given inputs
    '''
    args = parse_args()
    output_name = args.job_name

    save_dir = os.path.join(args.base_save_dir, output_name)
    os.system(f"mkdir -p {save_dir}")
    
    save_file_name = f"tensor_{args.layer_id}"
    if args.token_rule != "last":
        save_file_name += "_" + args.token_rule
    if args.do_decoding:
        save_file_name += "_multi"
    elif args.decoding_all:
        save_file_name += "_post"
    if args.raw:
        save_file_name += "_raw"
        
    if os.path.isfile(os.path.join(save_dir, save_file_name + ".pt")):
        print("Hidden states already generated.")
        exit(0)
    
    '''generate hidden states'''
    print(f"Loading from: {args.model_name_or_path}")
    print(f"you have {torch.cuda.device_count()} gpus")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto', torch_dtype=torch.bfloat16)
    hidden_states = []
    questions = load_questions(args.question_file)
    
    if args.do_decoding:
        assert args.answer_file != "", "must provide an answer file"
        answers = load_questions(args.answer_file)
        for batch in batch_read(list(zip(questions, answers)), args.batch_size):
            batch_q = [x[0] for x in batch]
            batch_a = [x[1] for x in batch]
            state = get_multi_hidden_state(tokenizer, 
                                           model, 
                                           [format_prompt(tokenizer, question["turns"][0], sys_prompt=HELPFUL_PROMPT, raw=args.raw) for question in batch_q],
                                           [format_prompt(tokenizer, answer["choices"][0]["turns"][0], raw=True) for answer in batch_a])
            hidden_states.append(state)
        state_tensor = torch.cat(hidden_states, dim=1)
    elif args.decoding_all:
        assert args.answer_file != "", "must provide an answer file"
        answers = load_questions(args.answer_file)
        for batch in batch_read(list(zip(questions, answers)), args.batch_size):
            state = get_hidden_state(tokenizer, model, [format_prompt(tokenizer, question["turns"][0], sys_prompt=HELPFUL_PROMPT, raw=args.raw) + format_prompt(tokenizer, answer["choices"][0]["turns"][0], raw=True) for (question, answer) in batch])
            hidden_states.append(state)
        
        state_tensor = torch.cat(hidden_states, dim=0)
    else:
        for batch in batch_read(questions, args.batch_size):
            state = get_hidden_state(tokenizer, model, [format_prompt(tokenizer, question["turns"][0], sys_prompt=HELPFUL_PROMPT, raw=args.raw) for question in batch])
            hidden_states.append(state)
            
        state_tensor = torch.cat(hidden_states, dim=0)

    print(state_tensor.shape)
    torch.save(state_tensor, os.path.join(save_dir, save_file_name + ".pt"))