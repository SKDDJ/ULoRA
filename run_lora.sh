#!/bin/bash

# export WANDB_MODE=offline


run(){
  task_name="cola"
  learning_rate=1e-2
  num_train_epochs=80
  per_device_train_batch_size=64
  rank=8
  l_num=12
  seed=0
  lora_alpha="16"
  target_modules="query value key"
  mode="base"
  lora_dropout=0.
  lora_bias=none
  lora_task_type=SEQ_CLS
  wandb_project=test_sara
  wandb_run_name=seed0-2cards-cola-lora-qkv-classifier
  share=false
  wandb_run_name=roberta-sara
  
  exp_dir=../roberta_glue_reproduce/${wandb_run_name}
#CUDA_VISIBLE_DEVICES=0,1,2,4
  HF_ENDPOINT=https://hf-mirror.com accelerate launch --main_process_port 29509 ./run_glue_lora.py  \
  --model_name_or_path FacebookAI/roberta-base  \
  --task_name ${task_name} \
  --do_train --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size ${per_device_train_batch_size} \
  --load_best_model_at_end True --metric_for_best_model "matthews_correlation" \
  --learning_rate ${learning_rate} \
  --num_train_epochs ${num_train_epochs} \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --weight_decay 0. \
  --warmup_ratio 0.06 \
  --logging_steps 10 \
  --seed ${seed} --wandb_project ${wandb_project} \
  --lora_alpha ${lora_alpha} --lora_dropout ${lora_dropout} --lora_bias ${lora_bias} \
  --lora_task_type ${lora_task_type} --target_modules ${target_modules} --rank ${rank} \
  --l_num ${l_num} --mode "base" \
  --output_dir ${exp_dir}/model \
  --logging_dir ${exp_dir}/log \
  --run_name ${wandb_run_name} \
  --overwrite_output_dir
}
task_base=('cola')
# task_base=('mnli' 'mrpc' 'qnli' 'qqp' 'rte' 'sst2' 'stsb' 'cola')

run $task_base[0]
# for task in "${task_base[@]}"; do
#     # run $task "8" "1" "base"
#     run $task "8" "2" "me"
# done