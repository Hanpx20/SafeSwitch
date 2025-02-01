import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
from collections import Counter
import re


false_pos, false_neg = [], []
for model in ["Qwen2.5-7B-Instruct", "Yi-1.5-9B-Chat", "Llama-3.1-8B-Instruct", "Ministral-8B"]:
    false_pos, false_neg = [], []
    for bench in ["trustllm-jailbreak"]:
        with open(f"model_answers/{model}/{bench}/judgements.jsonl", "r") as f:
            judgements = [json.loads(x) for x in f.readlines()]
        with open(f"model_answers/{model}/{bench}_head_cond.jsonl", "r") as f:
            model_answers = [json.loads(x) for x in f.readlines()]
        with open(f"datasets/{bench}.jsonl", "r") as f:
            orig_questions = [json.loads(x) for x in f.readlines()]
            
        for q, a, j in zip(orig_questions, model_answers, judgements):
            if a["mitigate"] == True and j["judgment"] == "0":
                false_pos.append(q["turns"][0])
            elif a["mitigate"] == False and j["judgment"] == "1":
                false_neg.append(q["turns"][0])
                
    print(len(false_pos), len(false_neg))
        
count = Counter(false_pos)
result = [element for element, freq in count.items() if freq == 3] # queries that are falsely predicted by all 3 models
for sent in result:
    print(sent)
    print('-'*50)

# def preprocess(sentences):
#     words = []
#     for sentence in sentences:
#         sentence = re.sub(r'[^\w\s]', '', sentence).lower()
#         words.extend(sentence.split())
#     return words

# def word_frequency_analysis(sentences):
#     words = preprocess(sentences)
#     word_counts = Counter(words)
#     return word_counts


# word_counts = word_frequency_analysis(false_pos)

# print(len(false_pos), len(false_neg))
# print("Word Frequency Analysis:")
# for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:200]:
#         print(f"{word}: {count}")