import json
import matplotlib.pyplot as plt
import numpy as np

line_colors = {}
def style():
    global line_colors
    
    line_colors['red'] = "#e3716e"
    
    line_colors["light_grey"] = "#afb0b2"
    line_colors["grey"] = "#656565"
    
    line_colors["green"] = "#c0db82"
    line_colors["yellow_green"] = "#54beaa"
    
    line_colors["pink"] = "#efc0d2"
    
    line_colors["light_purple"] = "#eee5f8"
    line_colors["purple"] = "#af8fd0"
    
    line_colors["blue"] = "#6d8bc3"
    line_colors["cyan"] = "#2983b1"
    
    line_colors["yellow"] = "#f9d580"
    
    line_colors["orange"] = "#eca680"

    plt.rcParams['font.family'] = 'Cambria'
    plt.rcParams['font.size'] = 16
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 10
    
    
style()



correct = {}
all = {}
comply = {}

for i in range(1, 46):
    correct[i] = 0
    all[i] = 0
    comply[i] = 0

for model in ["Qwen2.5-7B-Instruct", "Yi-1.5-9B-Chat", "Llama-3.1-8B-Instruct"]:
    question_file = "datasets/sorry-bench-eval.jsonl"
    judgement = f"model_answers_3/{model}/sorry-bench-eval/judgements.jsonl"
    prediction = f"model_answers_3/{model}/sorry-bench-eval_head_cond.jsonl"
    with open(question_file, "r") as f1, open(judgement, "r") as f2, open(prediction, "r") as f3:
        questions = [json.loads(x) for x in f1.readlines()]
        judgements = [json.loads(x) for x in f2.readlines()]
        predictions = [json.loads(x) for x in f3.readlines()]
        
        for q, j, p in zip(questions, judgements, predictions):
            category = (int(q["question_id"] - 1) % 90) // 2 + 1
            all[category] += 1
            if j["judgment"] == "1":
                comply[category] += 1
            
            if (j["judgment"] == "1") == (p["mitigate"] == True):
                correct[category] += 1



name = ["Hate Speech Generation", "Assistance with Crimes or Torts", "Potentially Inappropriate Topics", "Potentially Unqualified Advice"]
def convert(x):
    if 1 <= x <= 5:
        return 0
    if x <= 25:
        return 1
    if x <= 40:
        return 2
    return 3

correct_grp = {}
comply_grp = {}
all_grp = {}
for i in range(4):
    correct_grp[i] = 0
    comply_grp[i] = 0
    all_grp[i] = 0

for i in range(1, 46):
    correct_grp[convert(i)] += correct[i]
    comply_grp[convert(i)] += comply[i]
    all_grp[convert(i)] += all[i]
    correct[i] = correct[i] / all[i]
    comply[i] = comply[i] / all[i]

    if correct[i] < 0.8:
        print("bad", i, correct[i], comply[i])
    if correct[i] > 0.99:
        print("good", i, correct[i], comply[i])




colors = [line_colors["blue"], line_colors["red"], line_colors["purple"], line_colors["green"]]
fig, ax = plt.subplots(figsize=(8, 6))
appeared = set()
for i in range(1, 46):
    plt.scatter(list(comply.values())[i-1], list(correct.values())[i-1], color=colors[convert(i)], s=81, marker='o', alpha=1, label=None if convert(i) in appeared else name[convert(i)])
    appeared.add(convert(i))

ax.set_ylim(None, 1.02)
ax.set_xlim(None, 1.02)
xticks = ax.get_xticks()
yticks = ax.get_yticks()


for tick in xticks:
    ax.axvline(x=tick, color=line_colors["light_grey"], linestyle='--', linewidth=0.5, zorder=0)
for tick in yticks:
    ax.axhline(y=tick, color=line_colors["light_grey"], linestyle='--', linewidth=0.5, zorder=0)

# for i in range(4):
#     plt.scatter(comply_grp[i] / all_grp[i], correct_grp[i] / all_grp[i], color=colors[i], s=400, marker="o", alpha=0.4)
    
ax.set_xlabel("Comply Rate")
ax.set_ylabel("Prediction Accuracy")
ax.legend()
plt.savefig("figs/comply_vs_correct.png", dpi=200)