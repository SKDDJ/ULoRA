export WANDB_MODE=offline
export WANDB_PROJECT=dreambooth-sara
export WANDB_NAME=peft-test
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export INSTANCE_DIR="dog"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="/root/shiym_proj/Sara/peft/examples/lora_dreambooth/output"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=3e-2 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --report_to="wandb" \
  --validation_prompt="A photo of sks dog in a bucket" \
  --num_validation_images=4 \
  --validation_steps=50 \
  --seed="42" \
  --use_lora \
  --no_tracemalloc

# --learning_rate=lora 1e-4 5e-6\

  # --class_data_dir=$CLASS_DATA_DIR \
  # --class_prompt="photo of a woman" \
  # --with_prior_preservation --prior_loss_weight=1.0 \
  # --learning_rate_1d=1e-6 \
  # --train_text_encoder \
  # --num_class_images=200 \
  # --max_train_steps=1000 \
  # --use_8bit_adam \
  # --enable_xformers_memory_efficient_attention \
  # --gradient_checkpointing
