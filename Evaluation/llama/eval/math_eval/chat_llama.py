from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import logging

logging.basicConfig(level=logging.INFO)

model_id = "/root/shiym_proj/Sara/models/llama2_hf"
# model_id = "/root/shiym_proj/Sara/Evaluation/llama/math_output/1536_fp16/Trained_llama"
# model_id = "/root/shiym_proj/Sara/Evaluation/llama/math_output/1536_fp16/checkpoint-51"


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_id,
    model_max_length=512,
    padding_side="right",
    use_fast=True,
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.float16,
)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Create a text generation pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("<============loading success!============>")

# Generate text
sequences = pipeline(
    'Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers market?',
    do_sample=True,
    max_length=200,  # Adjust as needed
)

print(sequences[0].get("generated_text"))

