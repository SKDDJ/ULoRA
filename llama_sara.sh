#!/bin/bash
# export WANDB_MODE=offline

gpu=0
# Hyper-Parameter MELoRA
# Max sequence length 256
# Batch size 128
# Number of epochs 3
# LoRA dropout 0.05
# LR scheduler Linear
# Learning rate η 3e-4

# Hyperparameter FourierFT
# Optimizer AdamW
# Batch Size 4
# Accumulation Steps 4
# Epochs 1
# LR Schedule Linear
# Learning Rate 3E-3

run(){
  # bs=128
  # micro_bs=8
  precision='bf16'
  bs=16
  micro_bs=4 # per device batch size
  learning_rate='1e-2'
  num_train_epochs=30
  mode=$1
  rank=$2
  seed=42
  bf16=True
  fp16=False

  lora_alpha="1536"
  target_name='qkvout'
  lora_dropout=0.05 # default 0.05
  lora_bias=none
  cutoff_len=256
  wandb_project=528-lr-llama-1536
  wandb_run_name=1536-bcz4no1-lr-${learning_rate}-mbs-${micro_bs}-r-${rank}-1-alpha-${lora_alpha}-${target_name}-bs-${bs}-len-${cutoff_len}-epochs-${num_train_epochs}-seed-${seed}
  echo $wandb_run_name
  exp_dir=./llama-lora/${wandb_run_name}
  mkdir -p $exp_dir
  #  --lora_target_modules='[q_proj,v_proj]' 
    # --lora_target_modules='[q_proj, o_proj, k_proj, v_proj, gate_proj, up_proj, down_proj]' \
  CUDA_VISIBLE_DEVICES=$gpu python llama_sara.py \
    --base_model=/root/shiym_proj/Sara/models/llama2_hf \
    --cutoff_len=$cutoff_len \
    --mode=$mode \
    --fp16=$fp16 \
    --bf16=$bf16 \
    --seed=$seed \
    --group_by_length \
    --lora_r=$rank \
    --lora_alpha=$lora_alpha \
    --lora_dropout=$lora_dropout \
    --lora_target_modules='[q_proj, o_proj, k_proj, v_proj]' \
    --batch_size=$bs \
    --micro_batch_size=$micro_bs \
    --num_epochs=$num_train_epochs \
    --learning_rate=$learning_rate \
    --wandb_project=$wandb_project \
    --wandb_run_name=$wandb_run_name \
    --output_dir=${exp_dir}/model
}

# run SaRA with rank 256
run 'sara' 1536
