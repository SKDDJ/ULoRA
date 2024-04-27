# from safetensors import safe_open
# import pysnooper
import torch

lora_path = (
    "/root/shiym_proj/DiffLook/data/aulicravalho_xl_1_standard_wo_cap-000059-step00000435.pt"
)
tensors = torch.load(lora_path)
# tensors = {}
# with pysnooper.snoop():
# with safe_open(lora_path, framework="pt", device=0) as f:
#     for k in f.keys():
#         tensors[k] = f.get_tensor(k)

total_params = 0

# print("=="*20)

# ## calculate the paramters of the model
for key, tensor in tensors.item():
    num_elements = tensor.numel()  # For PyTorch tensors
    total_params += num_elements

print(f"Total parameters in the model: {total_params}")


# import torch

# flattened_tensors = []
# for tensor in tensors.values():
#     flattened_tensor = tensor.view(-1)
#     flattened_tensors.append(flattened_tensor)
# # flattened_tensors torch.Size([128]) 1210 个形状为 torch.Size([128]) 的张量

# # 这将把每个张量展平成一维，并将它们存储在 flattened_tensors 列表中

# concatenated_tensor = torch.cat(flattened_tensors, dim=0)
# # 这将把所有展平的张量连接成一个大张量。
# # concatenated_tensor torch.Size([154880])


# ### 现在，我们需要将这个大张量转换为 PyTorch Lightning 可以接受的数据格式。我们可以创建一个自定义的 Dataset 类，并在__getitem__方法中返回相应的张量切片。
# from torch.utils.data import Dataset

# class LoRADataset(Dataset):
#     def __init__(self, tensor):
#         self.tensor = tensor

#     def __len__(self):
#         return len(self.tensor)

#     def __getitem__(self, idx):
#         return self.tensor[idx]

# dataset = LoRADataset(concatenated_tensor)

# ### 最后，我们可以从这个自定义的 Dataset 创建一个 DataLoader, 并将其传递给 trainer.fit 方法:

# from torch.utils.data import DataLoader

# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # 在PyTorch Lightning模型中
# # trainer.fit(model, dataloader)
