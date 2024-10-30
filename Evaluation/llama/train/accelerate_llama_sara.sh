#!/bin/bash
export WANDB_MODE=offline

run(){
  # bs=128
  # micro_bs=8
  precision='fp16'
  bs=128
  micro_bs=8 # per device batch size
  learning_rate='1e-3'
  num_train_epochs=3
  mode=$1
  rank=$2
  lora_alpha=$2
  seed=42

  target_name='qv'
  lora_dropout=0. # default 0.05
  lora_bias=none
  cutoff_len=256
  # cutoff_len=512
  wandb_project=loldu-instruct-following
  wandb_run_name=lr-${learning_rate}-mbs-${micro_bs}-r-${rank}-${target_name}-bs-${bs}-epochs-${num_train_epochs}-test

  echo $wandb_run_name
  exp_dir=./llama-lora/${wandb_run_name}
  mkdir -p $exp_dir
    #  --lora_target_modules='[q_proj,v_proj]' 
    # --lora_target_modules='[q_proj, o_proj, k_proj, v_proj, gate_proj, up_proj, down_proj]' \
    HF_ENDPOINT=https://hf-mirror.com accelerate launch llama_sara_accelerate.py \
    --base_model="Shiym/llama2-7B" \
    --cutoff_len=$cutoff_len \
    --mode=$mode \
    --seed=$seed \
    --group_by_length \
    --lora_r=$rank \
    --lora_alpha=$lora_alpha \
    --lora_dropout=$lora_dropout \
    --lora_target_modules='[q_proj, v_proj]' \
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