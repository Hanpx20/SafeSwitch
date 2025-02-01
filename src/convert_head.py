import torch
from transformers import AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str, help="Path to the Hugging Face model")
parser.add_argument("head_path", type=str, help="Path to the refusal head")
parser.add_argument("save_path", type=str, help="Path to save the modified model")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_path)
tensor = torch.load(args.head_path)

model.lm_head.weight.data = tensor

model.save_pretrained(args.save_path)