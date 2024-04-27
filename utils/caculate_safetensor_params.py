from safetensors import safe_open

def load_model_weights(file_path):
    # 加载模型权重
    with safe_open(file_path, framework="pt", device="cpu") as f:
        model_weights = {k: f.get_tensor(k) for k in f.keys()}
    
    # 统计总参数量
    total_params = sum(p.numel() for p in model_weights.values())
    
    return model_weights, total_params

# 指定权重文件路径
weights_file = "/root/shiym_proj/DiffLook/z_outputs/zandaye.safetensors"

# 加载权重并统计参数量
model_weights, total_params = load_model_weights(weights_file)


# 打印权重的键和形状
print("Weight Keys and Shapes:")
# Open a new text file in write mode
with open('playground_vae.txt', 'w') as file:
    # Loop through the items in the dictionary
    for key, value in model_weights.items():
        # Write each key-value pair to the file
        # if value.numel() == 1:
        #     print("")
        # elif value.shape[0] == 4:
        #     print(value)
        #     exit()
        
        file.write(f"{key}: {value.shape}\n")


    
# 打印总参数量
print(f"Total Parameters: {total_params}")