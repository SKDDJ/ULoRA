from typing import List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.stat_scores import StatScores
from transformers import AutoConfig, AutoModelForImageClassification
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from src.loss import SoftTargetCrossEntropy
from src.mixup import Mixup

from labml.logger import inspect
from labml import monit

from functools import partial

import sys
sys.path.append('/root/shiym_proj/Sara/')

from utils.SaRA.minsara import SaRAParametrization,add_sara, apply_to_sara, disable_sara, enable_sara, get_sara_params, merge_sara, name_is_sara, remove_sara,get_sara_state_dict,add_sara_by_name


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
            if "vector_z" in n:
                vector_params += num_params
            trainable_params += num_params
        # print(f"{n} : {num_params:,d}\n")
    print(
        f"vector params: {vector_params:,d} || trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    print(
        f"vector params: {vector_params:,d} || trainable params(wo classifier): {trainable_params-classifier_params:,d} || all params: {all_param-classifier_params:,d} || trainable%: {100 * (trainable_params-classifier_params) / (all_param-classifier_params)}"
    )
    return vector_params

def only_train_vector_params(model):
    for param in model.parameters():
        param.requires_grad = False
    # vector_Z启用梯度
    # scaling factor启用梯度 sara_utils.py
    for param in get_sara_params(model):
        param.requires_grad = True
    # 为classifier也启用梯度
    for n, p in model.named_parameters():
        if 'classifier' in n:
            p.requires_grad = True

MODEL_DICT = {
    "vit-b16-224-in21k": "google/vit-base-patch16-224-in21k",
    "vit-b32-224-in21k": "google/vit-base-patch32-224-in21k",
    "vit-l32-224-in21k": "google/vit-large-patch32-224-in21k",
    "vit-l15-224-in21k": "google/vit-large-patch16-224-in21k",
    "vit-h14-224-in21k": "google/vit-huge-patch14-224-in21k",
    "vit-b16-224": "google/vit-base-patch16-224",
    "vit-l16-224": "google/vit-large-patch16-224",
    "vit-b16-384": "google/vit-base-patch16-384",
    "vit-b32-384": "google/vit-base-patch32-384",
    "vit-l16-384": "google/vit-large-patch16-384",
    "vit-l32-384": "google/vit-large-patch32-384",
    "vit-b16-224-dino": "facebook/dino-vitb16",
    "vit-b8-224-dino": "facebook/dino-vitb8",
    "vit-s16-224-dino": "facebook/dino-vits16",
    "vit-s8-224-dino": "facebook/dino-vits8",
    "beit-b16-224-in21k": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "beit-l16-224-in21k": "microsoft/beit-large-patch16-224-pt22k-ft22k",
}


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "vit-b16-224-in21k",
        optimizer: str = "sgd",
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "cosine",
        warmup_steps: int = 0,
        n_classes: int = 10,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        mix_prob: float = 1.0,
        label_smoothing: float = 0.0,
        image_size: int = 224,
        weights: Optional[str] = None,
        training_mode: str = "full",
        lora_r: int = 16,
        lora_alpha: int = 16,
        init_sara_weights: str = "fast_init_8",
        init_method: str = "svd",
        lora_target_modules: List[str] = ["query", "value"],
        lora_dropout: float = 0.0,
        lora_bias: str = "none",
        from_scratch: bool = False,
    ):
        """Classification Model

        Args:
            model_name: Name of model checkpoint. List found in src/model.py
            optimizer: Name of optimizer. One of [adam, adamw, sgd]
            lr: Learning rate
            betas: Adam betas parameters
            momentum: SGD momentum parameter
            weight_decay: Optimizer weight decay
            scheduler: Name of learning rate scheduler. One of [cosine, none]
            warmup_steps: Number of warmup steps
            n_classes: Number of target class
            mixup_alpha: Mixup alpha value
            cutmix_alpha: Cutmix alpha value
            mix_prob: Probability of applying mixup or cutmix (applies when mixup_alpha and/or
                cutmix_alpha are >0)
            label_smoothing: Amount of label smoothing
            image_size: Size of input images
            weights: Path of checkpoint to load weights from (e.g when resuming after linear probing)
            training_mode: Fine-tuning mode. One of ["full", "linear", "lora"]
            lora_r: Dimension of LoRA update matrices
            lora_alpha: LoRA scaling factor
            lora_target_modules: Names of the modules to apply LoRA to
            lora_dropout: Dropout probability for LoRA layers
            lora_bias: Whether to train biases during LoRA. One of ['none', 'all' or 'lora_only']
            from_scratch: Initialize network with random weights instead of a pretrained checkpoint
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr = lr
        self.betas = betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.n_classes = n_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.label_smoothing = label_smoothing
        self.image_size = image_size
        self.weights = weights
        self.training_mode = training_mode
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.init_sara_weights = init_sara_weights
        self.init_method = init_method
        self.lora_target_modules = lora_target_modules
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.from_scratch = from_scratch

        # Initialize network
        try:
            model_path = MODEL_DICT[self.model_name]
        except:
            raise ValueError(
                f"{model_name} is not an available model. Should be one of {[k for k in MODEL_DICT.keys()]}"
            )

        if self.from_scratch:
            # Initialize with random weights
            config = AutoConfig.from_pretrained(model_path)
            config.image_size = self.image_size
            self.net = AutoModelForImageClassification.from_config(config)
            self.net.classifier = torch.nn.Linear(config.hidden_size, self.n_classes)
        else:
            # Initialize with pretrained weights
            self.net = AutoModelForImageClassification.from_pretrained(
                model_path,
                num_labels=self.n_classes,
                ignore_mismatched_sizes=True,
                image_size=self.image_size,
            )

        # Load checkpoint weights
        if self.weights:
            print(f"Loaded weights from {self.weights}")
            ckpt = torch.load(self.weights)["state_dict"]

            # Remove prefix from key names
            new_state_dict = {}
            for k, v in ckpt.items():
                if k.startswith("net"):
                    k = k.replace("net" + ".", "")
                    new_state_dict[k] = v

            self.net.load_state_dict(new_state_dict, strict=True)

        # Prepare model depending on fine-tuning mode
        if self.training_mode == "linear":
            # Freeze transformer layers and keep classifier unfrozen
            for name, param in self.net.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        elif self.training_mode == "sara":
            print(f"Using sara, sara_r :{self.lora_r}")
            print(f"Using sara, sara_alpha :{self.lora_alpha}")
            sara_config = {
                nn.Linear: {
                    "weight": partial(SaRAParametrization.from_linear, rank=self.lora_r, lora_dropout_p=self.lora_dropout, lora_alpha=self.lora_alpha,init_sara_weights=self.init_sara_weights, init_method=self.init_method),
                },
            }        
            # model, tokenizer = accelerator.prepare(model, tokenizer)
            
            target_modules=self.lora_target_modules
            with monit.section("Apply_SaRA"):
                # # 打印模型参数的名称和大小
                # for name, param in self.net.named_parameters():
                #     print(f"{name}: {param.size()}")
                # # 打印模型参数的名称和数据类型
                # for name, param in self.net.named_parameters():
                #     print(f"{name}: {param.dtype}")
                add_sara_by_name(self.net, target_module_names=target_modules,sara_config=sara_config)
                only_train_vector_params(self.net)
                print_vector_parameters(self.net)
        elif self.training_mode == "lora":
            # Wrap in LoRA model
            config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias=self.lora_bias,
                modules_to_save=["classifier"],
            )
            self.net = get_peft_model(self.net, config)
        elif self.training_mode == "full":
            pass  # Keep all layers unfrozen
        else:
            raise ValueError(
                f"{self.training_mode} is not an available fine-tuning mode. Should be one of ['full', 'linear', 'sara']"
            )

        # Define metrics
        self.train_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                "acc_top5": Accuracy(
                    num_classes=self.n_classes,
                    task="multiclass",
                    top_k=min(5, self.n_classes),
                ),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                "acc_top5": Accuracy(
                    num_classes=self.n_classes,
                    task="multiclass",
                    top_k=min(5, self.n_classes),
                ),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                "acc_top5": Accuracy(
                    num_classes=self.n_classes,
                    task="multiclass",
                    top_k=min(5, self.n_classes),
                ),
                "stats": StatScores(
                    task="multiclass", average=None, num_classes=self.n_classes
                ),
            }
        )

        # Define loss
        self.loss_fn = SoftTargetCrossEntropy()

        # Define regularizers
        self.mixup = Mixup(
            mixup_alpha=self.mixup_alpha,
            cutmix_alpha=self.cutmix_alpha,
            prob=self.mix_prob,
            label_smoothing=self.label_smoothing,
            num_classes=self.n_classes,
        )

        self.test_metric_outputs = []

    def forward(self, x):
        return self.net(pixel_values=x).logits

    def shared_step(self, batch, mode="train"):
        x, y = batch

        if mode == "train":
            # Only converts targets to one-hot if no label smoothing, mixup or cutmix is set
            x, y = self.mixup(x, y)
        else:
            y = F.one_hot(y, num_classes=self.n_classes).float()

        # Pass through network
        pred = self(x)
        loss = self.loss_fn(pred, y)

        # Get accuracy
        metrics = getattr(self, f"{mode}_metrics")(pred, y.argmax(1))

        # Log
        self.log(f"{mode}_loss", loss, on_epoch=True)
        for k, v in metrics.items():
            if len(v.size()) == 0:
                self.log(f"{mode}_{k.lower()}", v, on_epoch=True)

        if mode == "test":
            self.test_metric_outputs.append(metrics["stats"])

        return loss

    def training_step(self, batch, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return self.shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self.shared_step(batch, "val")

    def test_step(self, batch, _):
        return self.shared_step(batch, "test")

    def on_test_epoch_end(self):
        """Save per-class accuracies to csv"""
        # Aggregate all batch stats
        combined_stats = torch.sum(
            torch.stack(self.test_metric_outputs, dim=-1), dim=-1
        )

        # Calculate accuracy per class
        per_class_acc = []
        for tp, _, _, _, sup in combined_stats:
            acc = tp / sup
            per_class_acc.append((acc.item(), sup.item()))

        # Save to csv
        df = pd.DataFrame(per_class_acc, columns=["acc", "n"])
        df.to_csv("per-class-acc-test.csv")
        print("Saved per-class results in per-class-acc-test.csv")

    def configure_optimizers(self):
        # Initialize optimizer
        if self.optimizer == "adam":
            optimizer = Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            optimizer = SGD(
                self.net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        # Initialize learning rate scheduler
        if self.scheduler == "cosine":
            # Apply the cosine learning rate scheduler with warmup phase
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
                num_warmup_steps=self.warmup_steps,
            )
        elif self.scheduler == "none":
            # Static learning rate using a lambda that always returns 1 (no change in learnign rate)
            scheduler = LambdaLR(optimizer, lambda _: 1)
        elif self.scheduler == "linear":
            # Apply linear scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
            )
        else:
            # Raise ValueError for unknown scheduler types in setup
            raise ValueError(
                f"{self.scheduler} is not an available optimizer setting. Should be one of ['cosine', 'none', 'linear']"
            )
        # if self.scheduler == "cosine":
        #     scheduler = get_cosine_schedule_with_warmup(
        #         optimizer,
        #         num_training_steps=int(self.trainer.estimated_stepping_batches),
        #         num_warmup_steps=self.warmup_steps,
        #     )
        # elif self.scheduler == "none":
        #     scheduler = LambdaLR(optimizer, lambda _: 1)
        # else:
        #     raise ValueError(
        #         f"{self.scheduler} is not an available optimizer. Should be one of ['cosine', 'none']"
        #     )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
