from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel, PeftConfig
import argparse
import torch

import sys
from typing import List

from safetensors import safe_open
from safetensors.torch import load_model, save_model


import torch.nn as nn
import bitsandbytes as bnb
from labml import monit

from functools import partial


import sys
sys.path.append('/root/shiym_proj/Sara/utils/loldu')

# print(sys.path)
# exit()
from minsara import SaRAParametrization,add_sara, apply_to_sara, disable_sara, enable_sara, get_sara_params, merge_sara, name_is_sara, remove_sara,get_sara_state_dict,add_sara_by_name,sara_state_dict
# import sys
# sys.path.append('/root/shiym_proj/Sara/')

# from utils.SaRA.minsara import SaRAParametrization,add_sara, apply_to_sara, disable_sara, enable_sara, get_sara_params, merge_sara, name_is_sara, remove_sara,get_sara_state_dict,add_sara_by_name,sara_state_dict

parser = argparse.ArgumentParser(description='Merge Adapter to Base Model')
parser.add_argument('--base_mode', type=str)
parser.add_argument('--use_float', type=str)
parser.add_argument('--adapter', type=str)
parser.add_argument('--r', type=int)
parser.add_argument('--alpha', type=int)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()

if args.use_float == 'fp16':
    float_type = torch.float16
elif args.use_float == 'bf16':
    float_type = torch.bfloat16
elif args.use_float == 'fp32':
    float_type = torch.float32
else:
    raise ValueError('Invalid float type')
    
model = AutoModelForCausalLM.from_pretrained(args.base_mode, torch_dtype=float_type, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(args.base_mode, device_map='auto')

# here need to do the load sara ckpt and load_stat_dict into and merge it.


# Read the .safetensors file
state_dict_to_save = {}
with safe_open(args.adapter, framework="pt", device=model.device) as f:  # Adjust 'framework' and 'device' as necessary
    for k in f.keys():
        state_dict_to_save[k] = f.get_tensor(k)

lora_r = args.r
lora_alpha = args.alpha
lora_dropout = 0.
lora_target_modules: List[str] = [
    "q_proj",
    "v_proj",
]

sara_config = {
    nn.Linear: {
        "weight": partial(SaRAParametrization.from_linear, rank=lora_r, lora_dropout_p=lora_dropout, lora_alpha=lora_alpha)
    },
}        

target_modules=lora_target_modules
with monit.section("Merge_SaRA"):
    add_sara_by_name(model, target_module_names=target_modules,sara_config=sara_config)
    _ = model.load_state_dict(state_dict_to_save, strict=False)
    merge_sara(model) 



model.save_pretrained(args.output_path)
tokenizer.save_pretrained(args.output_path)