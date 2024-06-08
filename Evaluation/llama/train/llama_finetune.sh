#!/bin/bash
# export WANDB_MODE=offline
gpu=7
run(){
  # bs=128
  # micro_bs=8
  bs=128
  micro_bs=4 # per device batch size
  learning_rate='1e-4'
  num_train_epochs=3
  mode=$1
  rank=$2
  seed=42
  lora_alpha="16"
  target_name='qv'
  lora_dropout=0.05
  lora_bias=none
  cutoff_len=512
  wandb_project=sara_llama_alpaca
  wandb_run_name=test-full-tune
  echo $wandb_run_name
  exp_dir=./llama-lora/${wandb_run_name}
  mkdir -p $exp_dir

  CUDA_VISIBLE_DEVICES=$gpu python llama_finetune.py \
    --base_model=Shiym/llama2-7B \
    --cutoff_len=$cutoff_len \
    --mode=$mode \
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

run 'base' 8 1
