# default epoch 100
# default rank 4
python train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="/data/projects/dongsoo/clab/textures/test_dataset" \
  --resolution=1024 --random_flip \
  --train_batch_size=2 \
  --num_train_epochs=300 --checkpointing_steps=10000 \
  --learning_rate=1e-04 --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --seed=2024 \
  --output_dir="lora" \
  --mixed_precision="fp16" \
  --rank=512 \
  --validation_prompt="s3wnf3lt dog" \
  --report_to="wandb"

