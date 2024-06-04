#!/bin/bash
# CUDA_VISIBLE_DEVICES=6 python main.py fit --config /root/shiym_proj/Sara/vit-finetune/configs/full/flowers102.yaml --trainer.accelerator gpu  --trainer.precision 16-mixed --trainer.max_epochs 10 --model.warmup_steps 30 --model.lr 0.001 --data.batch_size 128 &

CUDA_VISIBLE_DEVICES=7 python main.py fit --config /root/shiym_proj/Sara/vit-finetune/configs/full/aircraft.yaml --trainer.accelerator gpu  --trainer.precision 16-mixed --trainer.max_epochs 10 --model.warmup_steps 30 --model.lr 0.0006 --data.batch_size 128 &

# CUDA_VISIBLE_DEVICES=7 python main.py fit --config /root/shiym_proj/Sara/vit-finetune/configs/lora/flowers102.yaml --trainer.accelerator gpu  --trainer.precision 16-mixed --trainer.max_epochs 10 --model.warmup_steps 30 --model.lr 0.05 --data.batch_size 128 