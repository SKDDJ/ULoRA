import hydra
import pysnooper
import rootutils
import torch
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.lora_module import LoRALitModule


def reverse_Tensor2model(lora, outputs):
    param = outputs  # torch.Size([374544])
    target_num = 0
    for _, module in lora.named_parameters():
        target_num += torch.numel(module)

    params_num = torch.squeeze(param).shape[0]  # + 30720
    assert target_num == params_num
    param = torch.squeeze(param)
    # 参数重塑并更新模型：通过函数partial_reverse_tomodel，方法把压缩的参数向量转回模型的形式，并应用到模型的指定层上。这个步骤用于在测试之前更新模型的权重
    model = partial_reverse_tomodel(param, lora).to(param.device)

    model.eval()
    torch.save(model, "reversed_model.pth")
    return


def partial_reverse_tomodel(flattened, model):
    layer_idx = 0
    for _, pa in model.named_parameters():
        pa_shape = pa.shape
        pa_length = pa.view(-1).shape[0]
        pa.data = flattened[layer_idx : layer_idx + pa_length].reshape(pa_shape)
        pa.data.to(flattened.device)
        layer_idx += pa_length
    return model


def load_and_flatten(file_path):
    # 加载.pt文件
    loaded_data = torch.load(file_path)
    # 若加载的数据是单个张量
    # print(loaded_data.shape) # torch.Size([612, 612])
    if torch.is_tensor(loaded_data):
        # data = loaded_data.flatten()
        # print(loaded_data.shape) # torch.Size([612, 612])
        data = loaded_data.view(-1)
        # print(type(data)) # <class 'torch.Tensor'>
        # print(data.shape) # torch.Size([374544])
        # print("=="*20)

        final = torch.unsqueeze(data, 0)

        # print(final.size()) # torch.Size([1, 374544])
        # print("final:",type(final)) # final: <class 'torch.Tensor'>
        return final
    # 若加载的数据是包含多个张量的字典
    tensors = []
    for k, tensor in loaded_data.items():
        print("this lora has two more tensors")
        flattened_tensor = tensor.flatten()
        tensors.append(flattened_tensor)
    # 将处理后的多个张量串联为一个张量
    return torch.cat(tensors)


def fix_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("net._orig_mod.", "net.")
        new_state_dict[new_key] = value
    return new_state_dict


@pysnooper.snoop()
def main():
    # 1. original lora weights
    lora_path = "/root/shiym_proj/DiffLook/tests/test_lora.pt"
    lora = torch.load(lora_path)
    print(lora)
    print("==" * 20)
    print(lora.shape)

    # 2. get flattened lora weights
    lora_input = load_and_flatten(lora_path)
    # print(lora_input.shape) # torch.Size([1, 374544])
    # print(type(lora_input))
    # print(lora_input.shape) # torch.Size([374544])

    # 3. load autoencoder model using Lightning's pl.load_from_checkpoint
    ae_path = "/root/shiym_proj/DiffLook/logs/train/multiruns/2024-03-27_02-30-20/0/checkpoints/epoch_479.ckpt"

    import tempfile

    # Load the checkpoint
    checkpoint = torch.load(ae_path)

    # Fix the state dict in the checkpoint
    checkpoint["state_dict"] = fix_state_dict(checkpoint["state_dict"])

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    # Save the checkpoint to the temporary file
    torch.save(checkpoint, temp_file.name)

    # Load the model from the checkpoint
    autoencoder = LoRALitModule.load_from_checkpoint(temp_file.name)
    # Load the model from the checkpoint
    # autoencoder = LoRALitModule.load_from_checkpoint(checkpoint=checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device) # cuda
    autoencoder.to(device)
    lora_input = lora_input.to(device)

    # 4. get the ae output(predicted lora tensors)
    outputs = autoencoder(lora_input)

    outputs = outputs.to(device)

    reverse_Tensor2model(lora, outputs)


if __name__ == "__main__":
    main()


# def partial_reverse_tomodel(flattened, model, train_layer):
#     layer_idx = 0
#     for name, pa in model.named_parameters():
#         if name in train_layer:
#             pa_shape = pa.shape
#             pa_length = pa.view(-1).shape[0]
#             pa.data = flattened[layer_idx:layer_idx + pa_length].reshape(pa_shape)
#             pa.data.to(flattened.device)
#             layer_idx += pa_length
#     return model
