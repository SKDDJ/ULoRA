
import torch
import sys


import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


compiled_model_path="/root/shiym_proj/DiffLook/logs/train/runs/2024-04-08_20-08-07/checkpoints/epoch_089.ckpt"

# 检查命令行参数的数量
if len(sys.argv) > 0:
    compiled_model_path = sys.argv[1]
else:
    compiled_model_path = compiled_model_path

# NOTE
# if use torch.compile then use this function
def fix_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("net._orig_mod.", "net.")
        new_state_dict[new_key] = value
    return new_state_dict

def on_load_checkpoint(path:str):
        checkpoint = torch.load(path)
        # Fix the state dict in the checkpoint
        checkpoint["state_dict"] = fix_state_dict(checkpoint["state_dict"])
        # keys_list = list(checkpoint['state_dict'].keys())
        # for key in keys_list:
        #     if 'orig_mod.' in key:
        #         deal_key = key.replace('net._orig_mod.', 'net.')
        #         checkpoint['state_dict'][deal_key] = checkpoint['state_dict'][key]
        torch.save(checkpoint, path)
        print(f"Checkpoint has been fixed and saved back to {compiled_model_path}")
# 已经修正了 checkpoint 的 "state_dict" 属性，现在保存回原始路径
                # del checkpoint['state_dict'][key]
                



          
if __name__ == "__main__":
        on_load_checkpoint(compiled_model_path)