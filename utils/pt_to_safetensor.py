from safetensors import safe_open
import torch
import numpy as np
import os
from pathlib import Path
from safetensors.torch import save_file

import sys


def load_reference_weights(file_path):
    with safe_open(file_path, framework="pt", device="cpu") as f:
        reference_weights = {k: f.get_tensor(k) for k in f.keys()}
    return reference_weights

def load_processed_tensor(file_path):
    processed_tensor = torch.load(file_path,map_location=torch.device('cpu'))
    processed_tensor = processed_tensor.flatten(0)
    processed_tensor = processed_tensor/10
    new_tensor = processed_tensor.detach()
    print(processed_tensor.shape)
    return new_tensor

def restore_weights(reference_weights, processed_tensor):
    restored_weights = {}
    current_index = 0
    num = 0
    for key, value in reference_weights.items():
        if value.numel() == 1 and value.item()-4.0 <= 0.00001:
            restored_weights[key] = torch.tensor(4.0).to(torch.float32)
            num += restored_weights[key].numel()
            # num +=1
        else:
            if value.shape[-1] == 4:
                size = value.numel() / 4
                reshaped_tensor = processed_tensor[current_index:current_index+value.numel()].reshape(4, -1).t().numpy()
                # num +=value.numel()
                # print(reshaped_tensor)
                # exit()
            elif value.shape[0] == 4:
                size = value.numel() / 4
                reshaped_tensor = processed_tensor[current_index:current_index+value.numel()].reshape(4, -1).numpy()
                # num += value.numel()
                # print(reshaped_tensor)
                # exit()
            else:
                raise TypeError("wrong")
            restored_weights[key] = torch.from_numpy(reshaped_tensor).contiguous().to(torch.float32)
            current_index += value.numel()
            num += restored_weights[key].numel()
    assert isinstance(restored_weights, dict), "restore_weights() should return a dictionary"
    print(num)
    # exit()
    return restored_weights


def save_safetensors(weights, file_path):
    save_file(weights, file_path)

def create_directory_if_not_exists(directory_path):
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

# 文件夹路径,需要根据实际情况调整
reference_file_path = "/root/shiym_proj/DiffLook/utils/zandaye.safetensors"
processed_folder_path = "/root/shiym_proj/DiffLook/z_outputs"
output_folder_path = "/root/shiym_proj/DiffLook/z_outputs"

# 检查命令行参数的数量
if len(sys.argv) > 3:
    processed_folder_path = sys.argv[1]
    output_folder_path = sys.argv[2]
    reference_file_path = sys.argv[3]
else:
    processed_folder_path = processed_folder_path
    output_folder_path = output_folder_path
    reference_file_path = reference_file_path

# 加载参考权重
reference_weights = load_reference_weights(reference_file_path)

# 遍历处理后的 .pt 文件
for file_name in os.listdir(processed_folder_path):
    if file_name.endswith(".pt"):
        processed_file_path = os.path.join(processed_folder_path, file_name)
        
        # 加载处理后的张量
        processed_tensor = load_processed_tensor(processed_file_path)
        
        # 恢复权重
        restored_weights = restore_weights(reference_weights, processed_tensor)
        
        # 保存为 .safetensors 文件
        output_file_path = os.path.join(output_folder_path, file_name.replace(".pt", ".safetensors"))
        create_directory_if_not_exists(output_folder_path)
        save_safetensors(restored_weights, output_file_path)
