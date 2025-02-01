from tqdm import tqdm
import time
import argparse
import json, sys, os
from vllm import SamplingParams, LLM
from ..utils import *


def load_answers(answer_path: str):
    answer = {}
    with open(answer_path) as fin:
        for line in fin:
            line = json.loads(line)
            answer[line["question_id"]] = line
            
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, required=True, help="The name of the benchmark question set.")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--mitigate_method", type=str, default="no")
    parser.add_argument("--answer-dir", type=str, default="model_answers", help="The output answer directory.")
    parser.add_argument("--conditional_mitigate", type=str, default="False", choices=["True", "False"])
    
    args = parser.parse_args()
    conditional_str = "_prob" if args.conditional_mitigate == "Prob" else ("_cond" if args.conditional_mitigate == "True" else "")
    answer_file = os.path.join(args.answer_dir, args.model_id, args.bench_name, f"{args.mitigate_method if args.mitigate_method!='no' else 'base'}{conditional_str}.jsonl")
    model_answers = load_answers(answer_file)
    

    correct = 0
    cnt = 0
    for idx, entry in model_answers.items():
        cnt += 1
        if any(x in entry["output"] for x in entry["answer"]):
            correct += 1

    print(f"Accuracy: {round(correct * 100 / cnt, 3)}")
    
    
    save_file_dir = os.path.join(args.answer_dir, args.model_id, args.bench_name, "judge", f"{args.mitigate_method if args.mitigate_method!='no' else 'base'}{conditional_str}")
    os.makedirs(save_file_dir, exist_ok=True)
    with open(os.path.join(save_file_dir, 'acc.txt'), "w") as f:
        f.write(f"{round(correct * 100 / cnt, 3)}\n")
    
