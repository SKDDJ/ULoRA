#!/bin/bash
export WANDB_MODE=offline
gpu=0

run(){
  # bs=128
  # micro_bs=8
  bs=32
  micro_bs=4
  learning_rate='3e-4'
  num_train_epochs=6
  mode=$1
  rank=$2
  seed=42
  bf16=False
  fp16=True
  lora_alpha="512"
  target_name='qv'
  lora_dropout=0.05
  lora_bias=none
  cutoff_len=256
  wandb_project=sara_llama_alpaca
  wandb_run_name=llama-sara-bf16-${bf16}-fp16-${fp16}-${target_name}-${mode}-r-${rank}-alpha-${lora_alpha}-seed-${seed}-bs-${bs}-lr-${learning_rate}-len-${cutoff_len}-epochs-${num_train_epochs}
  echo $wandb_run_name
  exp_dir=./llama-lora/${wandb_run_name}
  mkdir -p $exp_dir
  
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
    --lora_target_modules='[q_proj,v_proj]' \
    --batch_size=$bs \
    --micro_batch_size=$micro_bs \
    --num_epochs=$num_train_epochs \
    --learning_rate=$learning_rate \
    --wandb_project=$wandb_project \
    --wandb_run_name=$wandb_run_name \
    --output_dir=${exp_dir}/model
}

# run SaRA with rank 256
run 'sara' 256 
