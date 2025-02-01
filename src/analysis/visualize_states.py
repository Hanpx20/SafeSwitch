import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.lines as mlines
import torch
import json
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.gridspec as gridspec
from sklearn.svm import SVC

def brighten_color(hex_color, brightness_factor=0.2):
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) for i in range(0, 6, 2)]
    
    r = min(int(r * (1 + brightness_factor)), 255)
    g = min(int(g * (1 + brightness_factor)), 255)
    b = min(int(b * (1 + brightness_factor)), 255)

    return f"#{r:02x}{g:02x}{b:02x}"


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
    plt.rcParams['font.size'] = 14
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 12



# The only hparams
model_names = ["Qwen2.5-7B-Instruct"]
model2layer = {"Qwen2.5-7B-Instruct": 28, "Yi-1.5-9B-Chat": 48, "Llama-3.1-8B-Instruct": 32, "Ministral-8B-Instruct-2410": 36}
random_seed = 42
random.seed(random_seed)



style() # set the color & settings
n_points = 100

for model_name in model_names:
    # 创建一个Figure对象
    fig = plt.figure(figsize=(12, 4))  # 设置整个图的大小
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1])
    
    titles = [f"Layer {model2layer[model_name] // 4}", f"Layer {model2layer[model_name] // 2}", f"Layer {model2layer[model_name] * 3 // 4}", f"Layer {model2layer[model_name]} (Last)"]

    # 加载内部状态
    internal_states = torch.load(f"/shared/nas2/ph16/toxic/outputs/states/{model_name}/tensor_all.pt")
    safe_list, refused_list, complied_list = [], [], []

    # 获取真实标签
    with open(f"model_answers_3/{model_name}/sorry-bench-plus/judgements.jsonl", "r") as f:
        llama_base = [json.loads(x) for x in f.readlines()]

    for judgement in llama_base:
        if judgement["question_id"] > 4500:
            safe_list.append(judgement["question_id"])
        elif judgement["judgment"] == "1":
            complied_list.append(judgement["question_id"])
        else:
            refused_list.append(judgement["question_id"])

    # 随机采样
    safe_list = random.sample(safe_list, n_points)
    complied_list = random.sample(complied_list, n_points)
    refused_list = random.sample(refused_list, n_points)

    # 绘制子图
    for i in range(4):
        ax = fig.add_subplot(gs[i])
        ax.set_aspect('equal')

        # 获取三种类型点的隐藏状态
        harmless_points = internal_states[safe_list, model2layer[model_name] // 4 * (i + 1), :]
        harmful_rejected_points = internal_states[refused_list, model2layer[model_name] // 4 * (i + 1), :]
        harmful_complied_points = internal_states[complied_list, model2layer[model_name] // 4 * (i + 1), :]

        hidden_states = torch.cat([harmless_points, harmful_rejected_points, harmful_complied_points], dim=0).to(torch.float16)

        labels = torch.tensor([0] * n_points + [1] * n_points + [2] * n_points)

        hidden_states_np = hidden_states.numpy()

        pca = PCA(n_components=2)
        hidden_states_2d = pca.fit_transform(hidden_states_np)
        hidden_states_2d_torch = torch.tensor(hidden_states_2d, dtype=torch.float32)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        hidden_states_2d_normalized = scaler.fit_transform(hidden_states_2d)
        hidden_states_2d_torch = torch.tensor(hidden_states_2d_normalized, dtype=torch.float32)
        
        ax.scatter(hidden_states_2d_torch[:n_points, 0], hidden_states_2d_torch[:n_points, 1], c='#b0d992', label="Safe Input", alpha=0.7, s=36)
        ax.scatter(hidden_states_2d_torch[n_points:2*n_points, 0], hidden_states_2d_torch[n_points:2*n_points, 1], c='#54beaa', label="Refused Harmful Input", alpha=0.7, s=36)
        ax.scatter(hidden_states_2d_torch[2*n_points:3*n_points, 0], hidden_states_2d_torch[2*n_points:3*n_points, 1], c='#e3716e', label="Complied Harmful Input", alpha=0.7, s=36)
        
        for j, k, col, label in [(0, 2, "grey", "Safe-Unsafe Border"), (1, 2, "purple", "Refuse-Comply Border")]:
            # 准备两类点的数据
            X = np.vstack((hidden_states_2d_normalized[j*n_points:(j+1)*n_points], 
                        hidden_states_2d_normalized[k*n_points:(k+1)*n_points]))
            y = np.array([j] * n_points + [k] * n_points)
            
            # 训练线性SVM
            svm = SVC(kernel='linear', C=1.0)
            svm.fit(X, y)
            
            # 获取分隔线的系数
            w = svm.coef_[0]
            b = svm.intercept_[0]
            
            # 计算分隔线的两个端点
            x_values = np.linspace(-1, 1, 1000)
            y_values = -(w[0] / w[1]) * x_values - (b / w[1])
            filtered_points = [(x, y) for x, y in zip(x_values, y_values) if -1 <= x <= 1 and -1 <= y <= 1]
            x_values, y_values = zip(*filtered_points)
            # 绘制分隔线
            ax.plot(x_values, y_values, '--', linewidth=3, color=col, label=label)
        
        
        ax.tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_title(titles[i])
        if i == 0:
            fig.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.86), fancybox=True)


    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f"figs/visualize_states_{model_name}.png", dpi=400)
    plt.savefig(f"figs/visualize_states_{model_name}.pdf")