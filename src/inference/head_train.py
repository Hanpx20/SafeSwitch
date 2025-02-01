import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import wandb
import os
import torch

def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a model using Huggingface Trainer.")
    
    # Model and dataset parameters
    parser.add_argument("--model_id", type=str, default="/shared/nas2/shared/llms/Qwen1.5-7B-Chat", help="Pretrained model ID or path.")
    parser.add_argument("--train_file", type=str, default="/shared/nas2/ph16/toxic/data/refusal/finetune_data.jsonl", help="Path to the training dataset (JSONL format).")
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, default="/shared/nas2/ph16/toxic/finetuned_LM/Qwen1.5-7B-Chat", help="Output directory to save the model.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay.")
    
    # Miscellaneous
    parser.add_argument("--logging_steps", type=int, default=40, help="Logging frequency in steps.")
    parser.add_argument("--save_strategy", type=str, default="no", choices=["no", "epoch", "steps"], help="Model saving strategy.")
    parser.add_argument("--wandb_project", type=str, default="full-finetune", help="Wandb project name.")

    args = parser.parse_args()
    args.wandb_run_name = args.model_id

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load and configure model
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto", torch_dtype=torch.float16)
    
    for name, param in model.named_parameters(): # if you want to do full para finetuning remove this
        if "lm_head" not in name:
            param.requires_grad = False
        else:
            param.data = param.data.float()
    
    # Load dataset
    dataset = load_dataset("json", data_files={"train": args.train_file})

    def preprocess(example):
        messages = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(preprocess)
    
    def tokenize_function(example):
        encoded = tokenizer(example["text"])
        return encoded


    tokenized_datasets = dataset.map(tokenize_function)
    train_dataset = tokenized_datasets["train"]

    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False,
        return_tensors="pt"
    )

    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        save_strategy=args.save_strategy,
        logging_dir="./logs",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        report_to="wandb",
        fp16=True,
        run_name=args.wandb_run_name
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # Save the final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    lm_head_path = os.path.join(args.output_dir, "lm_head.pth")  # Save the head separately as a .pth file (torch tensor)
    torch.save(model.lm_head.weight, lm_head_path)


if __name__ == "__main__":
    main()
