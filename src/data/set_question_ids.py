import json
import random


''''''
file_name = "datasets/trustllm-misuse.jsonl"
dataset = "misuse"
''''''

X = []
with open(file_name, "r") as file:
    for id, line in enumerate(file):
        orig_dic = json.loads(line)
        dic = {
            "question_id": id + 1,
            "dataset": dataset,
            "turns": orig_dic["turns"]
        }
        X.append(dic)
        

with open(file_name, "w") as file:
    for dic in X:
        file.write(json.dumps(dic) + '\n')
