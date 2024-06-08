#!/bin/bash
# conda create -n mmlu python=3.10 -y
# conda activate mmlu
# pip install -r requirements.txt
# mkdir -p data
# wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
# tar -xf data/mmlu.tar -C data && mv data/data data/mmlu

echo "Starting Evaluations..."
date

gpu=4
MODEL_PATH="Shiym/llama2-7B"
ADAPTER="/home/yimingshi/shiym_proj/Sara/llama-lora"
OUTPUT="output/lora-llama-2-7b-r128"
R=1024

# Function to generate log file name
generate_log_filename() {
    local task=$1
    echo "${task}_${MODEL_PATH}.log"
}

echo "Merging adapter to base model..."
CUDA_VISIBLE_DEVICES=$gpu python merge_adapter_to_base_model.py --base_mode $MODEL_PATH --adapter $ADAPTER/ft/ --output_path $OUTPUT --r $R --alpha $R


echo "Evaluating on MMLU with LLaMA..."
MMLU_LOG=$(generate_log_filename "mmlu")
CUDA_VISIBLE_DEVICES=$gpu python main.py mmlu --model_name llama --model_path $OUTPUT > ./logs/519/$MMLU_LOG 2>&1

# Uncomment if you want to evaluate on 5-shot MMLU
# echo "Evaluating on 5-shot MMLU with LLaMA..."
# MMLU_5SHOT_LOG=$(generate_log_filename "mmlu_5shot")
# CUDA_VISIBLE_DEVICES=$gpu python main.py mmlu --model_name llama --model_path $OUTPUT --n_sample 5 > $MMLU_5SHOT_LOG 2>&1

echo "Evaluating on BBH with LLaMA..."
BBH_LOG=$(generate_log_filename "bbh")
CUDA_VISIBLE_DEVICES=$gpu python main.py bbh --model_name llama --model_path $OUTPUT > ./logs/519/$BBH_LOG 2>&1

echo "Evaluating on DROP with LLaMA..."
DROP_LOG=$(generate_log_filename "drop")
CUDA_VISIBLE_DEVICES=$gpu python main.py drop --model_name llama --model_path $OUTPUT > ./logs/519/$DROP_LOG 2>&1

echo "Evaluating on HumanEval with LLaMA..."
HUMANEVAL_LOG=$(generate_log_filename "humaneval")
CUDA_VISIBLE_DEVICES=$gpu python main.py humaneval --model_name llama --model_path $OUTPUT --n_sample 1 > ./logs/519/$HUMANEVAL_LOG 2>&1

echo "Evaluations Completed."
date


