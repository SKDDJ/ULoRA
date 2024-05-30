#!/bin/bash
# Redirect all output to output.log
# exec > vit_output.log 2>&1
# Script to fine-tune ViT-Base model with LoRA on CIFAR-100
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
DATASETS=("aircraft")
# SaRA specific parameters
gpu=6
R=768
ALPHA=768
SEED=42

MODEL_NAME="vit-l32-224-in21k" # Assuming a placeholder name for ViT/Large
NAME="large"

for DATASET in "${DATASETS[@]}"; do 
    echo "Evaluating on ${DATASET} with ${MODEL_NAME}..."
    CONFIG="/root/shiym_proj/Sara/vit-finetune/configs/sara/${DATASET}.yaml"
    CUDA_VISIBLE_DEVICES=$gpu python main.py fit --config $CONFIG --seed_everything $SEED --trainer.accelerator gpu  --trainer.precision 16-mixed \
       --trainer.max_epochs 100 --trainer.logger.name $NAME-$DATASET --model.warmup_steps 500 --model.lr 1e-2 \
      --data.batch_size 128 --data.dataset $DATASET --model.model_name $MODEL_NAME \
      --model.lora_r $R --model.lora_alpha $ALPHA > ./logs/514/vit_large_${DATASET}.log 2>&1
    
    echo "ViT/Large model with LoRA fine-tuning on $DATASET completed."  >> vit_large_output.log
done

echo "All tasks completed, logs saved in vit_large_output.log"
date
