#!/bin/bash
echo "Starting Evaluations..."
date
# | 缩写     | Class名称        |
# |----------|------------------|
# | cifar10  | CIFAR10          |
# | cifar100 | CIFAR100         |
# | dtd      | DTD              |
# | cars     | StanfordCars     |
# | aircraft | FGVCAircraft     |
# | pets37   | OxfordIIITPet    |
# | flowers102| Flowers102      |
# | eurosat  | EuroSAT          |

# DATASETS=("cifar10" "cifar100" "dtd" "aircraft" )
# DATASETS=("aircraft")
DATASETS=("cifar10" "cifar100" "dtd" "aircraft"  "eurosat" "pets37" "flowers102" "cars")
# DATASETS=("flowers102" )



MODEL_NAME="vit-b16-224-in21k"
NAME="base"


# MODEL_NAME="vit-l32-224-in21k" # Assuming a placeholder name for ViT/Large
# NAME="large"
gpu=7

R=16
ALPHA=32
FULLRANK="16-2"

LR=9e-5


# init_method = 'svd'  # 可选项: 'svd', 'uniform', 'normal', 'constant', 'ones', 'zeros'
# INIT_SARA_WEIGHTS="fast_init_8"
# init_sara_weights=fast_init_8
INIT_METHOD="svd"
init_method=blora
# default cosine but fourier use linear
LR_SCHEDULER="linear" 
LR_SCHEDULER_NAME=linear

SEED=42

for DATASET in "${DATASETS[@]}"; do 
    echo "Evaluating on ${DATASET} with ${MODEL_NAME}..."
    CONFIG="/root/shiym_proj/Sara/vit-finetune/configs/sara/${DATASET}.yaml"
    CUDA_VISIBLE_DEVICES=$gpu python main.py fit --config $CONFIG --seed_everything $SEED --trainer.accelerator gpu --trainer.devices=[0] --trainer.precision 16-mixed \
        --trainer.max_epochs 10 --trainer.logger.name 8LR-$LR-$init_method-$FULLRANK-$NAME-$DATASET-R-$R-LR-$LR-SCHEDULER-$LR_SCHEDULER_NAME --model.warmup_steps 30 --model.lr $LR --model.scheduler $LR_SCHEDULER \
      --data.batch_size 128 --data.dataset $DATASET --data.workers 4 --model.model_name $MODEL_NAME \
      --model.lora_r $R --model.lora_alpha $ALPHA --model.init_method $INIT_METHOD > ./logs/529-${INIT_METHOD}/vit_base_${DATASET}.log 2>&1
    
    echo "ViT/Base model with LoRA fine-tuning on $DATASET completed." >> vit_base_output.log
done
wait

echo "All tasks completed, logs saved in vit_base_output.log"
date
