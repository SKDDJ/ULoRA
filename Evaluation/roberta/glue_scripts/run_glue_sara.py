#!/usr/bin/env python
# coding=utf-8
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset, concatenate_datasets
import time
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from accelerate import Accelerator
import wandb

from labml.logger import inspect
from labml import monit

from functools import partial

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

import sys
sys.path.append('/root/shiym_proj/Sara/utils/loldu')

# print(sys.path)
# exit()
from minsara import SaRAParametrization,add_sara, apply_to_sara, disable_sara, enable_sara, get_sara_params, merge_sara, name_is_sara, remove_sara,get_sara_state_dict,add_sara_by_name,sara_state_dict

# from utils.SaRA.minsara import SaRAParametrization,add_sara, apply_to_sara, disable_sara, enable_sara, get_sara_params, merge_sara, name_is_sara, remove_sara,get_sara_state_dict,add_sara_by_name,sara_state_dict

# from utils.SaRA.minsara import SaRAParametrization,add_sara, apply_to_sara, disable_sara, enable_sara, get_sara_params, merge_sara, name_is_sara, remove_sara,get_sara_state_dict,add_sara_by_name

check_min_version("4.29.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    wandb_project: Optional[str] = field(
        default='',
        metadata={"help": "The name of the wandb project" },
    )
    wandb_run_name: Optional[str] = field(
        default='',
        metadata={"help": "The name of the wandb run" },
    )
    wandb_watch: Optional[str] = field(
        default='',
        metadata={"help": "options: false | gradients | all"},
    )
    wandb_log_model: Optional[str] = field(
        default='',
        metadata={"help": "options: false | true"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    init_method: str = field(
        default="lu", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lora_path: str = field(
        default=None, metadata={"help": "Path to peft model or model identifier from huggingface.co/models"}
    )
    l_num: int = field(default=None, metadata={"help": "How many Lora Weights for a pretrained weight"})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    rank: int = field(
        default=8, metadata={"help": "rank of lora"}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "alpha of lora"}
    )
    target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "Target modules of lora"}
    )
    train_classifier: bool = field(
        default=True, metadata={"help": "Whether to train classifier"}
    )
    use_sara: bool = field(
        default=False, metadata={"help": "Whether to use sara"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "Target modules of lora"}
    )
    lora_bias: str = field(
        default="none", metadata={"help": "bias option of lora"}
    )
    lora_task_type: str = field(
        default="SEQ_CLS", metadata={"help": "task type of lora model"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

# @dataclass
# class TrainingArguments:
#     """
#     Arguments pertaining to what data we are going to input our model for training and eval.

#     Using `HfArgumentParser` we can turn this class
#     into argparse arguments to be able to specify them on
#     the command line.
#     """
#     # output_dir=output_dir,
#     # evaluation_strategy="epoch",
#     save_strategy="no",  # 禁用保存策略
#     logging_steps=1,
#     # per_device_train_batch_size=batch_size,
#     # per_device_eval_batch_size=batch_size,
#     # max_steps=-1,
#     save_total_limit=0,  # 不保存检查点

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

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # we don't use a json file
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # here we got the model_args, data_args, training_args
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if len(data_args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = data_args.wandb_project
    if len(data_args.wandb_run_name) > 0:
        os.environ["WANDB_NAME"] = data_args.wandb_run_name
    if len(data_args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = data_args.wandb_watch
    if len(data_args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = data_args.wandb_log_model

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    last_checkpoint = None
    set_seed(training_args.seed)

    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    accelerator = Accelerator()
    # if accelerator.is_main_proces
    if accelerator.is_local_main_process:
        wandb.init()
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    model, tokenizer = accelerator.prepare(model, tokenizer)
    
    
    sara_config = {
    nn.Linear: {
        "weight": partial(SaRAParametrization.from_linear, rank=model_args.rank, lora_dropout_p=model_args.lora_dropout, lora_alpha=model_args.lora_alpha, init_method=model_args.init_method)
    },
}
    target_modules=model_args.target_modules
    train_classifier=model_args.train_classifier
    # if accelerator.is_local_main_process:
    if model_args.use_sara:
        with monit.section("Apply_SaRA"):
            # print(model.device)
            add_sara_by_name(model, target_module_names=target_modules,sara_config=sara_config)

    
    for param in model.parameters():
        param.requires_grad = False
    # vector_Z启用梯度
    for param in get_sara_params(model):
        param.requires_grad = True
    classifer_params = []
    # 假设我们想要冻结模型的所有参数，除了 vector_z
    if train_classifier:
        for name, param in model.named_parameters():
            if "classifier" in name:
                classifer_params.append(param)
                param.requires_grad = True
    # whether train classifier 
    if train_classifier:
        parameters = [
            {"params": list(get_sara_params(model))},
            {"params": classifer_params, "lr": training_args.learning_rate} if train_classifier else None,
        ]
    else:
        parameters = [
            {"params": list(get_sara_params(model))},
        ]
    
    if accelerator.is_local_main_process:
        print_vector_parameters(model)
    
    optimizer = torch.optim.AdamW(params=parameters, lr=training_args.learning_rate)
    optimizer = accelerator.prepare_optimizer(optimizer)

    prodigy = False
    if prodigy == True:
        # del optimizer
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            # params_to_optimize[1]["lr"] = args.learning_rate
            # params_to_optimize[2]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            parameters,
            lr=training_args.learning_rate,
            betas=(0.9,0.99),
            beta3=None,
            weight_decay=1e-4,
            eps=1e-8,
            decouple=True,
            use_bias_correction=True,
            safeguard_warmup=True,
        )
        optimizer = accelerator.prepare_optimizer(optimizer)
        training_args.lr_scheduler_type = "constant"

    # for name, param in model.state_dict().items():
    #     print(name, param.size())
    # state_dict_to_save = get_sara_state_dict(model)
    # print("*** state_dict_to_save ***\n")
    # print(state_dict_to_save.keys())
    # todo note: distributed
    model = model.module if hasattr(model, "module") else model
    # if the accelerate is the distributed training, the model is the model.module
    # model = accelerator.unwrap_model(model)
    
    if model_args.lora_path is not None and data_args.task_name in ['mrpc', 'rte', 'stsb']:
        print(f"*** Load MNLI weight from {os.path.join(model_args.lora_path,'adapter_model.bin')} ***")
        adapters_weights = torch.load(os.path.join(model_args.lora_path,'adapter_model.bin'), map_location=model.device)
        filtered_dict = {key: value for key, value in adapters_weights.items() if 'classifier' not in key}
        # set_peft_model_state_dict(model, filtered_dict)
        # set_peft_model_state_dict(model, filtered_dict)
        del adapters_weights
    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression 
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    # with training_args.main_process_first(desc="dataset map pre-processing"):
    with accelerator.main_process_first():
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    elif is_regression:
        metric = evaluate.load("mse")
    else:
        metric = evaluate.load("accuracy")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # # 遍历模型的所有参数
    if accelerator.is_local_main_process:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} 是可训练的: {param.requires_grad}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )
    model.config.use_cache = False

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            # resume_from_checkpoint=checkpoint
        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        accelerator.wait_for_everyone()
        # unwrapped_model = accelerator.unwrap_model(model)
        # save
        # accelerator.save(unwrapped_model.state_dict(), filename)
        wandb.log("train", metrics)
        trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        
    # # Evaluation
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     eval_combined = {}
    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #     tasks = [data_args.task_name]
    #     eval_datasets = [eval_dataset]
    #     if data_args.task_name == "mnli":
    #         tasks.append("mnli-mm")
    #         valid_mm_dataset = raw_datasets["validation_mismatched"]
    #         if data_args.local_test:
    #             valid_mm_dataset = valid_mm_dataset.select(range(8000, len(valid_mm_dataset)))
    #         if data_args.max_eval_samples is not None:
    #             max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
    #             valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
    #         eval_datasets.append(valid_mm_dataset)
    #         combined = {}

    #     for eval_dataset, task in zip(eval_datasets, tasks):
    #         metrics = trainer.evaluate(eval_dataset=eval_dataset)

    #         max_eval_samples = (
    #             data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #         )
    #         metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    #         if task == "mnli-mm":
    #             metrics = {k + "_mm": v for k, v in metrics.items()}
    #         if task is not None and "mnli" in task:
    #             combined.update(metrics)

    #         eval_combined = combined if task is not None and "mnli" in task else metrics
    #         trainer.log_metrics("eval", metrics)
    #         wandb.log("eval", metrics)
    #         # todo test log the vector_z
    #         for n,p in model.named_parameters():
    #             # wandb log 'vector_z' in name of model
    #             if 'vector_z' in n:
    #                wandb.log(p.detach().cpu().tolist())
    #                wandb.watch(p)
    #         trainer.save_metrics("eval", eval_combined)
if __name__ == "__main__":
    main()
