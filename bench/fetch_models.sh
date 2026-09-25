#!/bin/sh
cd /c/Users/hyper/Kiln/models
B=https://huggingface.co
curl -fL -C - -o anima-base-v1.0.safetensors $B/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors
curl -fL -C - -o qwen_3_06b_base.safetensors $B/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors
curl -fL -C - -o qwen_image_vae.safetensors $B/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors
curl -fL -C - -o anima-turbo-lora-v0.2.safetensors $B/circlestone-labs/Anima-Official-LoRAs/resolve/main/anima-turbo-lora-v0.2.safetensors
echo FETCH_DONE
