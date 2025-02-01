import os
import torch
import sys

'''
Split internal states into train and eval set
Used for prober training
'''



dir = sys.argv[1]
assert dir != "", "Please provide a directory path as an argument"
ratio = 8  # ratio of training data, out of 10
train_dir = f"{dir}/train"
eval_dir = f"{dir}/eval"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

for file in os.listdir(dir):
    if file.endswith(".pt"):
        base_name = file.split('/')[-1].split('.')[0]
        file_path = os.path.join(dir, file)
        X = torch.load(file_path).to("cpu")
        
        print(X.dtype)
        if X.dtype == torch.bfloat16:
            X = X.to(torch.float32)
        print(X.dtype)
        
        dim_with_N = None
        for dim, size in enumerate(X.shape):
            if size == 11000:
                dim_with_N = dim
                break
        
        if dim_with_N is None:
            print(f"No dimension of size 11000 found in {file_path}")
            continue
        
        train_indices = [i for i in range(11000) if i % 10 < ratio]  # 80%
        eval_indices = [i for i in range(11000) if i % 10 >= ratio]  # 20%

        train_data = X.index_select(dim_with_N, torch.tensor(train_indices))
        eval_data = X.index_select(dim_with_N, torch.tensor(eval_indices))

        torch.save(train_data, f"{train_dir}/{base_name}.pt")
        torch.save(eval_data, f"{eval_dir}/{base_name}.pt")

        print(f"Saved {base_name}.pt and {base_name}.pt")
