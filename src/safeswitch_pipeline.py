import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import safeswitch_generation, load_classifier, format_prompt
from transformers.utils import logging
import warnings
from termcolor import colored, cprint

warnings.filterwarnings("ignore")
logging.set_verbosity(logging.CRITICAL)

def main():
    parser = argparse.ArgumentParser(description="Run a language model with safety classifiers and refusal head.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model.")
    parser.add_argument("--n_token", type=int, default=256, help="Maximum number of new tokens to generate.")
    parser.add_argument("--n_decode", type=int, default=3, help="Decoding parameter.")
    parser.add_argument("--llm_dir", type=str, required=True, help="Directory containing LLM models.")
    parser.add_argument("--classifier_dir", type=str, required=True, help="Directory containing classifier models.")
    parser.add_argument("--refusal_head_dir", type=str, required=True, help="Directory containing refusal head models.")
    
    args = parser.parse_args()
    
    model_path = f'{args.llm_dir}/{args.model_name}'
    llm = AutoModelForCausalLM.from_pretrained(model_path, device_map=args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    judge_1 = load_classifier(f"{args.classifier_dir}/{args.model_name}_multi_safety/token0", llm.config.hidden_size, 2, device=args.device)
    judge_2 = load_classifier(f"{args.classifier_dir}/{args.model_name}_multi_response/token{args.n_decode}", llm.config.hidden_size, 2, device=args.device)
    new_head = torch.load(f"{args.refusal_head_dir}/{args.model_name}/lm_head.pth", map_location=args.device)


    while True:
        cprint("Input your prompt ending with a line break. Type 'exit' to end the script.", "blue")
        line = input()
        if line.strip().lower() == 'exit':
            break
        line = format_prompt(tokenizer, line)
        safeswitch_generation(
            llm, tokenizer, new_head, judge_1, judge_2, line,
            n_decode=args.n_decode, max_new_tokens=args.n_token,
            device=args.device
        )

if __name__ == "__main__":
    main()
