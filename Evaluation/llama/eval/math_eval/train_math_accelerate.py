#    Copyright 2024 Fanxu Meng, Zhaohui Wang, Muhan Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Literal

import torch
import transformers
from transformers import Trainer
from datasets import load_dataset
# from peft import LoraConfig, get_peft_model, PeftModel
import sys
from typing import List

from safetensors import safe_open
from safetensors.torch import load_model, save_model


import torch.nn as nn
import bitsandbytes as bnb
from labml import monit

from functools import partial

# import sys
# sys.path.append('/root/shiym_proj/Sara/')

# from utils.SaRA.minsara import SaRAParametrization,add_sara, apply_to_sara, disable_sara, enable_sara, get_sara_params, merge_sara, name_is_sara, remove_sara,get_sara_state_dict,add_sara_by_name,sara_state_dict

import sys
sys.path.append('/root/shiym_proj/Sara/utils/loldu')

# print(sys.path)
# exit()
from minsara import SaRAParametrization,add_sara, apply_to_sara, disable_sara, enable_sara, get_sara_params, merge_sara, name_is_sara, remove_sara,get_sara_state_dict,add_sara_by_name,sara_state_dict

import wandb

from accelerate import Accelerator





IGNORE_INDEX = -100
PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )


# torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)

def print_vector_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    classifier_params = 0
    
    trainable_params = 0
    vector_params = 0
    all_param = 0
    for n, param in model.named_parameters():
        num_params = param.numel() 
        all_param += num_params
        if 'original' in n:
            # print(f"{n} : {num_params:,d}\n")
            continue
        if param.requires_grad:
            if 'classifier' in n:
                classifier_params += num_params
            trainable_params += num_params
            if "vector_z" in n:
                vector_params += num_params
        # print(f"{n} : {num_params:,d}\n")
    print(
        f"vector params: {vector_params:,d} || trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    print(
        f"vector params: {vector_params:,d} || trainable params(wo classifier): {trainable_params-classifier_params:,d} || all params: {all_param-classifier_params:,d} || trainable%: {100 * (trainable_params-classifier_params) / (all_param-classifier_params)}"
    )
    return vector_params

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_float: Optional[str] = field(default="fp32", metadata={"help": "The floating point type to use: fp16, fp32, or bf16."})
    # max_steps: Optional[int] = field(default=10),  # 或者设置最大步数，取较小值
    optim: str = field(default="adamw_torch", metadata={"help": "The optimizer to use: adamw_torch, adamw_8bit, adamw_bnb_8bit, adamw_apex_fused, or adafactor"})
    use_gradient_checkpointing: bool = field(default=False, metadata={"help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."})
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    adapter_name_or_path: Optional[str] = field(default=None)
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_split: str = field(
        default="train[:100000]", metadata={"help": "(`['train', 'test', 'eval']`):"}
    )
    dataset_field: List[str] = field(
        default=None, metadata={"help": "Fields of dataset input and output."}
    )
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    lora_r: int = field(default=None, metadata={"help": "The rank of the adapter. When passing `None` and `adapter_name_or_path` is also `None`, full fine-tuning is used."})
    init_lora_weights: Literal[True, "pissa"] = field(
        default=True,
        metadata={
            "help": (
                "Passing True (default) results in the LoRA initialization." 
                "Passing `pissa` results in PiSSA initialization."
            ),
        },
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

class CustomTrainer(Trainer):
    def __init__(self, *args, accelerator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.accelerator = accelerator
    def create_optimizer(self):
        if self.accelerator.is_local_main_process:
            print(f"Creating optimizer with args.optim = {self.args.optim}")
        if self.args.optim == "adamw_bnb_8bit":
            if self.accelerator.is_local_main_process:
                print("=================Using 8-bit AdamW=================")
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            return super().create_optimizer()
        
        optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        if self.accelerator.is_local_main_process:
            print(f"Created optimizer: {optimizer}")
        return optimizer



def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    # 初始化 Accelerator
    accelerator = Accelerator()
    # print(script_args)   
    if script_args.use_float == "fp16": 
        if accelerator.is_local_main_process:
            print("=================Using fp16=================")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            # device_map="auto",
            torch_dtype=torch.float16,  # 使用float16类型
            use_safetensors=True, # todo: test use_safetensors
        )
    elif script_args.use_float == "fp32":
        if accelerator.is_local_main_process:
            print("=================Using fp32=================")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path, 
            # device_map="auto"
            )
    elif script_args.use_float == "bf16":
        if accelerator.is_local_main_process:
            print("=================Using bf16=================")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            # device_map="auto",
            torch_dtype=torch.bfloat16  # 使用bfloat16类型
        )
    else:
        raise ValueError("Invalid `use_float` argument.")
    # model.push_to_hub("Shiym/llama2-7B")
    # exit()


    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    
    tokenizer, model = accelerator.prepare(tokenizer, model)
    
    
    if script_args.adapter_name_or_path is not None:
        if accelerator.is_local_main_process:
            print(f"Load {script_args.init_lora_weights} from {script_args.adapter_name_or_path}: ",)
        # model = PeftModel.from_pretrained(model, script_args.model_name_or_path, subfolder=script_args.adapter_name_or_path, is_trainable=True)
    elif script_args.lora_r is not None:

        lora_r = script_args.lora_r
        lora_alpha = script_args.lora_r
        if accelerator.is_local_main_process:
            print(f"Using sara, sara_r :{lora_r}")
            print(f"Using sara, sara_alpha :{lora_alpha}")
        
        if accelerator.is_local_main_process:
            # Initialize wandb and name with the lora_r value
            wandb.init(project="LLAMA_MATH_ACC", name=f"float_{script_args.use_float}_rank_{lora_r}_accelerate")
            
        lora_dropout = 0.
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ]
        # lora_target_modules: List[str] = [
        #     "q_proj",  "k_proj", "v_proj",
        # ]
        # lora_target_modules: List[str] = [
        #     "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj",
        # ]

        sara_config = {
            nn.Linear: {
                "weight": partial(SaRAParametrization.from_linear, rank=lora_r, lora_dropout_p=lora_dropout, lora_alpha=lora_alpha)
            },
        }        

        target_modules=lora_target_modules
        with monit.section("Applying_SaRA"):
            add_sara_by_name(model, target_module_names=target_modules,sara_config=sara_config)
            
    else:
        print("Full Parameter Fine-Tuning")

    
    for param in model.parameters():
        param.requires_grad = False
    for param in get_sara_params(model):
        param.requires_grad = True

    if accelerator.is_local_main_process:    
        print("<=======params.requires_grad=======>")    
        print_vector_parameters(model)
        
    # def check_model_parameters(model):
    #     trainable_params = 0
    #     all_param = 0
    #     for name, param in model.named_parameters():
    #         all_param += param.numel()
    #         if param.requires_grad:
    #             trainable_params += param.numel()
    #             print(f"{name} requires gradient, shape: {param.shape}")
    #         else:
    #             print(f"{name} does not require gradient, shape: {param.shape}")
    #     print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    # # 在应用SaRA后调用这个函数
    # check_model_parameters(model)
    
    # if script_args.use_gradient_checkpointing:
        # note: gradient checkpointing is not supported with LoLDU
        # print("=================Using gradient checkpointing=================")
        # model.gradient_checkpointing_enable()
    
    model = model.module if hasattr(model, "module") else model
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    with accelerator.main_process_first():
        raw_train_datasets = load_dataset(script_args.data_path, split=script_args.dataset_split)
        train_dataset = raw_train_datasets.map(
            train_tokenize_function,
            batched=True,
            batch_size=3000,
            num_proc=32,
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
            fn_kwargs={"tokenizer": tokenizer, "query": script_args.dataset_field[0], "response": script_args.dataset_field[1]}
        )

        
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)
    
    # model.state_dict = sara_state_dict.__get__(model, type(model))
    
    # 使用自定义的Trainer
    trainer = CustomTrainer(accelerator=accelerator, model=model, tokenizer=tokenizer, args=script_args, **data_module)
    
    
    # 使用 Accelerator 准备 Trainer
    trainer = accelerator.prepare(trainer)
    
    model.config.use_cache = False
    trainer.train()
    
    with monit.section("Merge_SaRA"):
        merge_sara(model) 
    
    # model.save_pretrained(os.path.join(script_args.output_dir, "final_model_test_3"))
    # tokenizer.save_pretrained(os.path.join(script_args.output_dir, "final_model_test_2"))
    
    # 保存模型
    # trainer.save_model("final_model")
    accelerator.wait_for_everyone()
    # save tokenizier and model together
    trainer.save_model(os.path.join(script_args.output_dir, "Trained_llama"))

if __name__ == "__main__":
    train()
