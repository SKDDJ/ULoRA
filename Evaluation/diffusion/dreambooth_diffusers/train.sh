export WANDB_MODE=offline
export WANDB_PROJECT=dreambooth-sara
export WANDB_NAME=test
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_DIR="dog"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="/home/yimingshi/shiym_proj/Saradiffusers/examples/dreambooth/outputs"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=500 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --report_to="wandb" \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_steps=50 \
  --seed="0" \
  

# --learning_rate=lora 1e-4 5e-6\

  
