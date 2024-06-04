#!/bin/bash
gpu=4
### Todo: SETUP INSTRUCTIONS
# conda create -n mmlu python=3.10 -y
# conda activate mmlu
# pip install -r requirements.txt
# mkdir -p data
# wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
# tar -xf data/mmlu.tar -C data && mv data/data data/mmlu

# Redirect all output to output.log
# exec > 518-output-$gpu.log 2>&1
#!/bin/bash

echo "Starting Evaluations..."
date

# /root/shiym_proj/Sara/llama-lora/testLR-lr-5e-2-ac-4-r-1024-1-alpha-1024-qv-bs-16-len-256-epochs-3-seed-42/model/model.safetensors

# /root/shiym_proj/Sara/llama-lora/Dropout-bs128-lr-1e-3-ac-4-r-1024-1-alpha-1024-qv-bs-128-len-256-epochs-3-seed-42/model/model.safetensors

# /root/shiym_proj/Sara/llama-lora/Dropout-lr-3e-3-ac-4-r-1024-1-alpha-1024-qv-bs-16-len-256-epochs-3-seed-42/model/model.safetensors

# /root/shiym_proj/Sara/llama-lora/testLR-lr-1e-1-ac-4-r-1024-1-alpha-1024-qv-bs-16-len-256-epochs-3-seed-42/model/model.safetensors

# /root/shiym_proj/Sara/llama-lora/testLR-lr-3e-1-ac-4-r-1024-1-alpha-1024-qv-bs-16-len-256-epochs-3-seed-42/model/model.safetensors

# /root/shiym_proj/Sara/llama-lora/testLR-lr-3e-3-ac-4-r-1024-1-alpha-1024-qv-bs-16-len-256-epochs-3-seed-42/model/model.safetensors

# /root/shiym_proj/Sara/llama-lora/testLR-lr-3e-4-ac-4-r-1024-1-alpha-1024-qv-bs-16-len-256-epochs-3-seed-42/model/model.safetensors




MODEL_PATH="/root/shiym_proj/Sara/models/llama2_hf"
LORA_PATH="/root/shiym_proj/Sara/llama-lora/RANKHIGH-ac-4-r-1536-1-alpha-1536-lr-3e-3-qv-bs-16-len-256-epochs-3-seed-42/model/checkpoint-8000/model.safetensors"
# LORA_INFO=$(basename $(dirname $(dirname $(dirname $LORA_PATH))))  # 提取独有信息
LORA_INFO=$(basename $(dirname $(dirname $LORA_PATH)))  # 提取 Dropout-bs128-lr-1e-3-ac-4-r-1024-1-alpha-1024-qv-bs-128-len-256-epochs-3-seed-42
# echo $LORA_INFO

# Function to generate log file name
generate_log_filename() {
    local task=$1
    echo "log_${task}_${LORA_INFO}.log"
}

echo "Evaluating on MMLU with LLaMA..."
MMLU_LOG=$(generate_log_filename "mmlu")

echo $MMLU_LOG

CUDA_VISIBLE_DEVICES=$gpu python main.py mmlu --model_name llama --model_path $MODEL_PATH --lora_path $LORA_PATH > ./logs/519/$MMLU_LOG 2>&1

# Uncomment if you want to evaluate on 5-shot MMLU
# echo "Evaluating on 5-shot MMLU with LLaMA..."
# MMLU_5SHOT_LOG=$(generate_log_filename "mmlu_5shot")
# CUDA_VISIBLE_DEVICES=$gpu python main.py mmlu --model_name llama --model_path $MODEL_PATH --n_sample 5 > $MMLU_5SHOT_LOG 2>&1

echo "Evaluating on BBH with LLaMA..."
BBH_LOG=$(generate_log_filename "bbh")
CUDA_VISIBLE_DEVICES=$gpu python main.py bbh --model_name llama --model_path $MODEL_PATH --lora_path $LORA_PATH > ./logs/519/$BBH_LOG 2>&1

echo "Evaluating on DROP with LLaMA..."
DROP_LOG=$(generate_log_filename "drop")
CUDA_VISIBLE_DEVICES=$gpu python main.py drop --model_name llama --model_path $MODEL_PATH --lora_path $LORA_PATH > ./logs/519/$DROP_LOG 2>&1

echo "Evaluating on HumanEval with LLaMA..."
HUMANEVAL_LOG=$(generate_log_filename "humaneval")
CUDA_VISIBLE_DEVICES=$gpu python main.py humaneval --model_name llama --model_path $MODEL_PATH --lora_path $LORA_PATH --n_sample 1 > ./logs/519/$HUMANEVAL_LOG 2>&1

echo "Evaluations Completed."
date


