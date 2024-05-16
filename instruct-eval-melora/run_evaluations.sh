#!/bin/bash

### Todo: SETUP INSTRUCTIONS
# conda create -n instruct-eval python=3.8 -y
# conda activate instruct-eval
# pip install -r requirements.txt
# mkdir -p data
# wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
# tar -xf data/mmlu.tar -C data && mv data/data data/mmlu

# Redirect all output to output.log
exec > output.log 2>&1

echo "Starting Evaluations..."
date

echo "Evaluating on MMLU with LLaMA..."
python main.py mmlu --model_name llama --model_path chavinlo/alpaca-native

echo "Evaluating on MMLU with LLaMA..."
python main.py mmlu --model_name llama --model_path google/flan-t5-xl

echo "Evaluating on BBH with LLaMA..."
python main.py bbh --model_name llama --model_path TheBloke/koala-13B-HF --load_8bit

echo "Evaluating on DROP with LLaMA..."
python main.py drop --model_name llama --model_path google/flan-t5-xl

echo "Evaluating on HumanEval with LLaMA..."
python main.py humaneval  --model_name llama --model_path eachadea/vicuna-13b --n_sample 1 --load_8bit

echo "Evaluations Completed."
date