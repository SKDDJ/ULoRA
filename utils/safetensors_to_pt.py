import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

desired_length = 374544


def load_model_weights(file_path):
    with safe_open(file_path, framework="pt", device="cpu") as f:
        model_weights = {k: f.get_tensor(k) for k in f.keys()}
    return model_weights


def process_weights(model_weights):
    scalars = []
    vectors = []
    number = 0
    for key, value in model_weights.items():
        if value.numel() == 1 and value.item() - 4.0 <= 0.00001:  # Check if the tensor is a scalar
            number += 1
        else:
            if value.shape[-1] == 4:  # Check if the shape is (x, 4)
                reshaped = (
                    value.t()
                    .numpy()
                    .reshape(
                        -1,
                    )
                )  # Transpose and then reshape to 1D
            elif value.shape[0] == 4:
                reshaped = value.numpy().reshape(
                    -1,
                )  # Reshape tensor to 1D
            else:
                raise TypeError("wrong")
            vectors.append(reshaped)
    # Concatenate all vectors and then add the scalars at the end
    final_vector = np.concatenate(vectors + scalars)
    return final_vector


def create_directory_if_not_exists(directory_path):
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def convert_one_path(folder_path, output_path):
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        print(output_path+'/'+file_name.replace(".safetensors", ".pt"))
        if os.path.exists(output_path+'/'+file_name.replace(".safetensors", ".pt")):
            print("exist")
            continue
        if file_name.endswith(".safetensors"):
            file_path = os.path.join(folder_path, file_name)

            # 加载模型权重
            model_weights = load_model_weights(file_path)

            # 处理权重
            final_vector = process_weights(model_weights)

            # 添加填充以匹配所需长度
            # if final_vector.size < desired_length:
            #     padding = np.zeros(desired_length - final_vector.size)
            #     final_vector = np.concatenate([final_vector, padding])

            # # 保存最终向量
            final_tensor = torch.tensor(final_vector)

            # from labml.logger import inspect
            # inspect(final_tensor)
            # return

            # layers = [final_tensor[i * 612:(i + 1) * 612] for i in range(612)]
            # final_tensor = torch.stack(layers).to(torch.float32)

            # 保存为.pt文件
            # print( final_tensor.shape)
            # exit()
            new_file_name = file_name.replace(".safetensors", ".pt")
            create_directory_if_not_exists(output_path)
            torch.save(final_tensor, os.path.join(output_path, new_file_name))


if __name__ == "__main__":
    # 文件夹路径，需要根据实际情况调整
    folder_path = "/root/part2/"
    output_path = "/root/data/"

    for filename in os.listdir(folder_path):
        new_folder_path = folder_path + filename
        new_output_path = output_path + filename
        convert_one_path(new_folder_path, new_output_path)


# # 路径需要根据你的文件位置调整
# weights_file = "/root/shiym_proj/kohya_ss/outputs/amadams_xl_1_standard_merger_35_63_03_07/amadams_xl_1_standard_merger_35_63_03_07-step00000438.safetensors"

# model_weights = load_model_weights(weights_file)
# final_vector = process_weights(model_weights)

# if final_vector.size < desired_length:
#     padding = np.zeros(desired_length - final_vector.size)
#     final_vector = np.concatenate([final_vector, padding])

# # 保存最终向量
# final_tensor = torch.tensor(final_vector)
# layers = [final_tensor[i * 612:(i + 1) * 612] for i in range(612)]
# final_tensor = torch.stack(layers).to(torch.float32)

# # 保存为.pt文件
# torch.save(final_tensor, 'final_tensor.pt')
