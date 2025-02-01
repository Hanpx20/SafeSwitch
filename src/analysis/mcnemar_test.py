from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
import json


def calc(both_correct, A_correct, B_correct, both_wrong):
    table = np.array([[both_correct, A_correct], [B_correct, both_wrong]])
    result = mcnemar(table, exact=True, correction=True)
    return result.pvalue

if __name__ == "__main__":
    prompt_judge = []
    switch_judge = []
    for model in ["Qwen2.5-7B-Instruct", "Yi-1.5-9B-Chat", "Llama-3.1-8B-Instruct"]:
        for bench in ["trustllm"]:
            for subtype in ["jailbreak", "misuse"]:
                with open(f"model_answers/{model}/{bench}-{subtype}_prompt_strong/judgements.jsonl", "r") as f:
                    prompt_judge.extend([json.loads(x) for x in f.readlines()])
                with open(f"model_answers/{model}/{bench}-{subtype}_head_cond/judgements.jsonl", "r") as f:
                    switch_judge.extend([json.loads(x) for x in f.readlines()])
                    
                    
    a, b, c, d = 0, 0, 0, 0
    for p, s in zip(prompt_judge, switch_judge):
        if p["judgment"] == "0" and s["judgment"] == "0":
            a += 1
        elif p["judgment"] == "0" and s["judgment"] == "1": # prompt better
            b += 1
        elif p["judgment"] == "1" and s["judgment"] == "0": # switch better
            c += 1
        elif p["judgment"] == "1" and s["judgment"] == "1":
            d += 1
    # print(f"Results for {model} on {bench}:")
    if b > c:
        print("Prompt is better")
    else:
        print("Switch is better")
        b, c = c, b
    print(f"{b} {c} p value: {calc(a, b, c, d)}")