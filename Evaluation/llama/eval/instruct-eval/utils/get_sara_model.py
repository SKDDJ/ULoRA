import os
import sys
from typing import List

import fire
import torch
import transformers
from transformers import TrainerCallback, TrainerState, TrainerControl
from datasets import load_dataset

from safetensors import safe_open
from safetensors.torch import load_model, save_model

import wandb
import torch.nn as nn
import bitsandbytes as bnb
# from peft import (
    # LoraConfig,
    # get_peft_model,
    # get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    # set_peft_model_state_dict,
# )

from labml.logger import inspect
from labml import monit

from functools import partial

import sys
sys.path.append('/home/yimingshi/shiym_proj/Sara')

from utils.SaRA.minsara import SaRAParametrization,add_sara, apply_to_sara, disable_sara, enable_sara, get_sara_params, merge_sara, name_is_sara, remove_sara,get_sara_state_dict,add_sara_by_name,sara_state_dict


from transformers import LlamaForCausalLM, LlamaTokenizer, set_seed





# Specify the path to your .safetensors file
file_path = "/home/yimingshi/shiym_proj/Sarallama-lora/r-128-4-alpha-512-qv-bs-128-lr-3e-2-len-256-epochs-3-seed-42/model/model.safetensors"

# Read the .safetensors file
state_dict_to_save = {}
with safe_open(file_path, framework="pt", device=0) as f:  # Adjust 'framework' and 'device' as necessary
    for k in f.keys():
        state_dict_to_save[k] = f.get_tensor(k)

# Print the keys and shapes of the tensors
print("Keys and shapes of the tensors in the safetensors file:")
for key, tensor in state_dict_to_save.items():
    print(f"{key}: shape {tensor.shape}")
    
    

base_model = "Shiym/llama2-7B"
device_map = "auto"

lora_r = 128
lora_alpha = 512
lora_dropout = 0.05
lora_target_modules: List[str] = [
    "q_proj",
    "v_proj",
]



model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map=device_map,
    use_safetensors=True,
)



sara_config = {
    nn.Linear: {
        "weight": partial(SaRAParametrization.from_linear, rank=lora_r, lora_dropout_p=lora_dropout, lora_alpha=lora_alpha)
    },
}        


target_modules=lora_target_modules
with monit.section("Merge_SaRA"):
    for name, param in model.named_parameters():
        if 'layers.0' in name:
            if 'q_proj' in name:
                print("I'm original llama2 model")
                print(f"{name}: {param.size()} , {param}")
    add_sara_by_name(model, target_module_names=target_modules,sara_config=sara_config)
    
    for name, param in model.named_parameters():
            if 'layers.0' in name:
                if 'q_proj' in name:
                    print("I'm after apply sara")
                    print(f"{name}: {param.size()} , {param}")
    _ = model.load_state_dict(state_dict_to_save, strict=False)
    
    merge_sara(model) 
    
    save_model(model, "model.safetensors")
    
    for name, param in model.named_parameters():
        if 'layers.0' in name:
            if 'q_proj' in name:
                print("I'm new merged sara params")
                print(f"{name}: {param.size()} , {param}")