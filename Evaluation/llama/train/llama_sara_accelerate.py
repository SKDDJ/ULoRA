import os
import sys
from typing import List

import fire
import torch
import transformers
from transformers import TrainerCallback, TrainerState, TrainerControl
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import wandb
import torch.nn as nn
import bitsandbytes as bnb

from labml.logger import inspect
from labml import monit

from functools import partial




import sys
sys.path.append('/root/shiym_proj/Sara/utils/loldu')
from minsara import SaRAParametrization,add_sara, apply_to_sara, disable_sara, enable_sara, get_sara_params, merge_sara, name_is_sara, remove_sara,get_sara_state_dict,add_sara_by_name,sara_state_dict

sys.path.append('/root/shiym_proj/Sara/')
from eval.llama_utils.prompter import Prompter
# sys.path.append('/root/shiym_proj/Sara/')
# from utils.SaRA.minsara import SaRAParametrization,add_sara, apply_to_sara, disable_sara, enable_sara, get_sara_params, merge_sara, name_is_sara, remove_sara,get_sara_state_dict,add_sara_by_name,sara_state_dict


from transformers import LlamaForCausalLM, LlamaTokenizer, set_seed

# from utils.prompter import Prompter

# from root.shiym_proj.Sara.eval.llama_utils.prompter import Prompter


torch.backends.cuda.matmul.allow_tf32 = True


from accelerate import Accelerator


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

# class FactorWeightLogCallback(TrainerCallback):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.last_step = -1

#     def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
#         if self.last_step >= state.global_step or state.global_step % 10 != 0:
#             return

#         # Initialize a dictionary to collect all the logging information
#         log_data_q = {}
#         log_data_v = {}

#         # Iterate over all parameters in the model that contain the word 'factor'
#         for name, param in self.model.named_parameters():
#             if 'q_proj' in name:
#                 if 'factor' in name:
#                     param_values = param.detach().flatten().cpu().numpy()

#                     # Populate the log_data dictionary with parameter statistics
#                     log_data_q.update({
#                         f"{name}_hist": wandb.Histogram(param_values),  # Histogram of parameter values
#                         f"{name}_min": param_values.min(),               # Minimum of parameter values
#                         f"{name}_max": param_values.max(),               # Maximum of parameter values
#                         f"{name}_mean": param_values.mean(),             # Mean of parameter values
#                         f"{name}_std": param_values.std()                # Standard deviation of parameter values
#                     })
#             elif 'v_proj' in name:
#                 if 'factor' in name:
#                     param_values = param.detach().flatten().cpu().numpy()

#                     # Populate the log_data dictionary with parameter statistics
#                     log_data_v.update({
#                         f"{name}_hist": wandb.Histogram(param_values),  # Histogram of parameter values
#                         f"{name}_min": param_values.min(),               # Minimum of parameter values
#                         f"{name}_max": param_values.max(),               # Maximum of parameter values
#                         f"{name}_mean": param_values.mean(),             # Mean of parameter values
#                         f"{name}_std": param_values.std()                # Standard deviation of parameter values
#                     })

#         # Log all collected data at once. This helps in synchronizing the data display in WandB dashboard
#         wandb.log(log_data_q, step=state.global_step)
#         wandb.log(log_data_v, step=state.global_step)
        
#         self.last_step = state.global_step
        

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    seed: int = 42,
    # lora hyperparams
    mode: str = "sara",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"seed: {seed}\n"
            f"mode: {mode}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"


    gradient_accumulation_steps = batch_size // micro_batch_size

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb",
        project_dir=output_dir,
    )
    # accelerator = Accelerator()
    
    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    set_seed(seed)
    with monit.section("loading llama..."):
        # model = LlamaForCausalLM.from_pretrained(
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            # load_in_8bit=True,
            torch_dtype=torch.float16,
            # device_map=device_map,
            use_safetensors=True,
        )
    with monit.section("loading tokenizer..."):
        # tokenizer = LlamaTokenizer.from_pretrained(base_model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
                base_model,
                model_max_length=512,
            )

        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"  # Allow batched inference

    tokenizer, model = accelerator.prepare(tokenizer, model)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt


    if accelerator.is_local_main_process:
        print(f"Using sara, sara_r :{lora_r}")
        print(f"Using sara, sara_alpha :{lora_alpha}")
        print(f"target modules :{lora_target_modules}")
        
    if accelerator.is_main_process:
        wandb.init()
        
    sara_config = {
        nn.Linear: {
            "weight": partial(SaRAParametrization.from_linear, rank=lora_r, lora_dropout_p=lora_dropout, lora_alpha=lora_alpha)
        },
    }        

    # model, tokenizer = accelerator.prepare(model, tokenizer)
    target_modules=lora_target_modules
    with monit.section("Apply_SaRA"):
        add_sara_by_name(model, target_module_names=target_modules,sara_config=sara_config)

        
    for param in model.parameters():
        param.requires_grad = False
    for param in get_sara_params(model):
        param.requires_grad = True


    if accelerator.is_local_main_process:
        print_vector_parameters(model)

    model = model.module if hasattr(model, "module") else model
    
    with accelerator.main_process_first():
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            data = load_dataset("json", data_files=data_path)
        else:
            data = load_dataset(data_path)

        if resume_from_checkpoint:
            # Check the available weights and load them
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "pytorch_model.bin"
            )  # Full checkpoint
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    resume_from_checkpoint, "adapter_model.bin"
                )  # only LoRA model - LoRA config above has to fit
                resume_from_checkpoint = (
                    False  # So the trainer won't try loading its state
                )


        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
            val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            logging_steps=10,
            fp16=True,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            save_safetensors=True,
            eval_steps=2 if val_set_size > 0 else None,
            save_steps=2,
            max_steps=2,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    # model.state_dict = sara_state_dict.__get__(model, type(model))
    
    # print(model.state_dict())  # 这应该输出经过 sara 处理的状态字典
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # note: we log the scaling factor using wandb
    # 添加这个callback到你的trainer
    # trainer.add_callback(FactorWeightLogCallback(model))
    
    # 使用 Accelerator 准备 Trainer
    trainer = accelerator.prepare(trainer)
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    with monit.section("Merging Model..."):
        merge_sara(model) 
    accelerator.wait_for_everyone()
    # model.save_pretrained(output_dir)
    trainer.save_model(output_dir)

    # print(
    #     "\n If there's a warning about missing keys above, please disregard :)"
    # )


if __name__ == "__main__":
    fire.Fire(train)