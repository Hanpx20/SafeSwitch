import datasets
import json


''''''
alpaca_data_file = "datasets/alpaca-eval.jsonl"
''''''

eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

with open("datasets/alpaca-eval.jsonl", "w") as file:
    for example in eval_set:
        example["turns"] = [example["instruction"]]
        del example["output"]
        del example["generator"]
        del example["instruction"]
        file.write(json.dumps(example) + "\n")