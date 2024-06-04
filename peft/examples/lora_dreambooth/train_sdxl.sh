# export WANDB_MODE=offline
export WANDB_PROJECT=xl-dreambooth-sara
export WANDB_NAME=use-snr-gamma-5

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="./dog_file/dog"
export OUTPUT_DIR="output-xl"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

gpu=3

CUDA_VISIBLE_DEVICES=$gpu accelerate launch train_dreambooth_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --checkpointing_steps=700 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1. \
  --report_to="wandb" \
  --wandb_proj=$WANDB_PROJECT \
  --optimizer="prodigy" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of dog in a bucket" \
  --validation_epochs=25 \
  --seed="42" \
  --use_lora \
  --enable_xformers_memory_efficient_attention \
  --use_8bit_adam \
  --snr_gamma=5 \
  # --do_edm_style_training \
  

  # optimizer 
  # attn2.to_ cross_attn

  # --use_8bit_adam \
  # --gradient_checkpointing
  # --enable_xformers_memory_efficient_attention \

  # playgroundai/playground-v2.5-1024px-aesthetic
  # stabilityai/stable-diffusion-xl-base-1.0

  # --do_edm_style_training
  # --snr_gamma=5 \