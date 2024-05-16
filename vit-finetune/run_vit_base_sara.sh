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



# DATASETS=("cifar10" "cifar100" "dtd" "cars" "aircraft" "pets37" "flowers102" "eurosat")
# DATASETS=("aircraft" "eurosat" "pets37")
DATASETS=( "eurosat")
# DATASETS=( "aircraft"  "eurosat")

MODEL_NAME="vit-b16-224-in21k"
NAME="base"

# MODEL_NAME="vit-l32-224-in21k" # Assuming a placeholder name for ViT/Large
# NAME="large"

# SaRA specific parameters
R=768
ALPHA=768
LR=3e-2
# default cosine but fourier use linear
LR_SCHEDULER="linear" 
LR_SCHEDULER_NAME=linear

FULLRANK="768-1"

SEED=42
gpu=3

for DATASET in "${DATASETS[@]}"; do 
    echo "Evaluating on ${DATASET} with ${MODEL_NAME}..."
    CONFIG="/root/shiym_proj/Sara/vit-finetune/configs/sara/${DATASET}.yaml"
    CUDA_VISIBLE_DEVICES=$gpu python main.py fit --config $CONFIG --seed_everything $SEED --trainer.accelerator gpu --trainer.devices=[0] --trainer.precision 16-mixed \
        --trainer.max_steps 5000 --trainer.logger.name $FULLRANK-$NAME-$DATASET-R-$R-LR-$LR-SCHEDULER-$$LR_SCHEDULER_NAME --model.warmup_steps 500 --model.lr $LR --model.scheduler $LR_SCHEDULER \
      --data.batch_size 128 --data.dataset $DATASET --data.workers 4 --model.model_name $MODEL_NAME \
      --model.lora_r $R --model.lora_alpha $ALPHA  > ./logs/515-sc2/vit_base_${DATASET}.log 2>&1
    
    echo "ViT/Base model with LoRA fine-tuning on $DATASET completed." >> vit_base_output.log
done
wait

echo "All tasks completed, logs saved in vit_base_output.log"
date
