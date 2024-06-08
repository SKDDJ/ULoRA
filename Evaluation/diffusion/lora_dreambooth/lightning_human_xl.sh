export WANDB_MODE=offline
export WANDB_PROJECT=xl-human-sara
export WANDB_NAME=xl-lightning-edm-style-right-cfg

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="./single_woman_file/woman"
export OUTPUT_DIR="output-human-xl-lightning"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

gpu=1

CUDA_VISIBLE_DEVICES=$gpu accelerate launch train_dreambooth_sdxl_lightning.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of woman" \
  --resolution=1024 \
  --train_batch_size=4 \
  --checkpointing_steps=700 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1. \
  --report_to="wandb" \
  --wandb_proj=$WANDB_PROJECT \
  --optimizer="prodigy" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of woman" \
  --negative_prompt="2d art, 3d art, ((illustration)), anime, cartoon, bad_pictures, bad-artist, EasyNegative,(worst quality:1.6), (low quality:1.6), (normal quality:1.6), low res, bad anatomy, bad hands, vaginas in breasts, ((monochrome)), ((grayscale)), collapsed eyeshadow, multiple eyebrow, (cropped), oversaturated, extra limb, missing limbs, deformed hands, long neck, long body, imperfect, (bad hands), signature, watermark, username, artist name, conjoined fingers, deformed fingers, ugly eyes, imperfect eyes, skewed eyes, unnatural face, unnatural body, error, bad image, bad photo" \
  --validation_epochs=8 \
  --seed="42" \
  --use_lora \
  --enable_xformers_memory_efficient_attention \
  --use_8bit_adam \
  --do_edm_style_training 
  # --snr_gamma=5 \
  

  # optimizer 
  # attn2.to_ cross_attn

  # --use_8bit_adam \
  # --gradient_checkpointing
  # --enable_xformers_memory_efficient_attention \

  # playgroundai/playground-v2.5-1024px-aesthetic
  # stabilityai/stable-diffusion-xl-base-1.0

  # --do_edm_style_training
  # --snr_gamma=5 \