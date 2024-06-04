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


MODEL_NAME="vit-l32-224-in21k" # Assuming a placeholder name for ViT/Large
NAME="large"
# init_method = 'svd'  # 可选项: 'svd', 'uniform', 'normal', 'constant', 'ones', 'zeros'
INIT_SARA_WEIGHTS="ldu"
init_sara_weights=ldu
INIT_METHOD="lu"
init_method=lu
# LR_SCHEDULER="none" 
# LR_SCHEDULER_NAME=none
LR_SCHEDULER="linear" 
LR_SCHEDULER_NAME=linear

OPT="prodigy"
DATASETS=("cifar10" "cifar100" "dtd" "aircraft"  "eurosat" "pets37" "flowers102" "cars")
# DATASETS=("flowers102")
gpu=1

R=768
ALPHA=768
FULLRANK="768-1"
LR=1.0

SEED=42

for DATASET in "${DATASETS[@]}"; do 
    echo "Evaluating on ${DATASET} with ${MODEL_NAME}..."
    CONFIG="/root/shiym_proj/Sara/vit-finetune/configs/sara/${DATASET}.yaml"
    CUDA_VISIBLE_DEVICES=$gpu python /root/shiym_proj/Sara/vit-finetune/main.py fit --config $CONFIG --seed_everything $SEED --trainer.accelerator gpu --trainer.devices=[0] --trainer.precision 16-mixed \
        --trainer.max_epochs 10 --trainer.logger.name linear-$NAME-prodigy-$init_method-$FULLRANK-$NAME-$DATASET-R-$R-LR-$LR-SCHEDULER-$LR_SCHEDULER_NAME --model.warmup_steps 30 --model.lr $LR --model.scheduler $LR_SCHEDULER --model.optimizer $OPT \
      --data.batch_size 128 --data.dataset $DATASET --data.workers 4 --model.model_name $MODEL_NAME \
      --model.lora_r $R --model.lora_alpha $ALPHA --model.init_method $INIT_METHOD --model.init_sara_weights $INIT_SARA_WEIGHTS > /root/shiym_proj/Sara/vit-finetune/logs/61-${INIT_METHOD}/vit_large_${DATASET}.log 2>&1
    
    echo "ViT/Base model with LoRA fine-tuning on $DATASET completed." >> vit_base_output.log
done
wait

echo "All tasks completed, logs saved in vit_base_output.log"
date
