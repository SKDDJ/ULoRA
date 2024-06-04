# export WANDB_MODE=offline
export WANDB_PROJECT=xl-woman-sara
export WANDB_NAME=xl-edm-all-attn-newprodigy

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# export MODEL_NAME="playgroundai/playground-v2.5-1024px-aesthetic"
export INSTANCE_DIR="./single_woman_file/woman"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

export OUTPUT_DIR="output-human-xl/$WANDB_NAME"

gpu=0

CUDA_VISIBLE_DEVICES=$gpu accelerate launch train_dreambooth_sdxl.py \
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
  --entity=$WANDB_NAME \
  --optimizer="prodigy" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="a photo of woman" \
  --validation_epochs=4 \
  --seed="42" \
  --use_lora \
  --enable_xformers_memory_efficient_attention \
  --use_8bit_adam \
  --adam_beta2=0.99 \
  --adam_weight_decay=0.01 \
  --do_edm_style_training \
  --lora_target_modules="attn1.to_k","attn1.to_v","attn1.to_q","attn1.to_out","attn2.to_k","attn2.to_v","attn2.to_q","attn2.to_out" \


#   --snr_gamma=5 \
# --adam_beta2 default = 0.999 but prodigy use 0.99
# --adam_weight_decay default = 1e-4 but prodigy use 0.01
# --prodigy_beta3 default is None, we can change it later.(todo)
# --prodigy_use_bias_correction default is True, we don't change
# --prodigy_safeguard_warmup default is True, we don't change
# --prodigy_decouple default is True, we don't change

# # test only self attn
# --lora_target_modules="attn1.to_k","attn1.to_v","attn1.to_q","attn1.to_out" \

# # test only cross attn
# --lora_target_modules="attn2.to_k","attn2.to_v","attn2.to_q","attn2.to_out" \

# # test both self and cross attn
# --lora_target_modules="attn1.to_k","attn1.to_v","attn1.to_q","attn1.to_out","attn2.to_k","attn2.to_v","attn2.to_q","attn2.to_out" \



  # --gradient_checkpointing

