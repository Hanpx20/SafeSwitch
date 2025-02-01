from datasets import load_dataset
import json



ds = load_dataset("mandarjoshi/trivia_qa", "rc")["train"]


with open("datasets/trivia_qa.jsonl", "w") as file:
    for i in range(2000):
        Q = {
            "question_id": i,
            "turns": [ds[i]["question"]],
            "dataset": "trivia_qa",
            "answer": ds[i]["answer"]["aliases"]
        }
        file.write(json.dumps(Q) + "\n")