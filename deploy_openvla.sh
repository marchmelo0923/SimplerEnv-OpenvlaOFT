#!/bin/bash

# Default values for host and port
HOST="0.0.0.0"
PORT="8777"


: "${OFT_MODEL_PATH:=/SSD/LSY/huggingface_cache/hub/models--shylee--bridge_oft2/snapshots/3d1c2ee60502718d21c32814910b7d3a9b8bf08b}"
export OFT_MODEL_PATH

# python openvla-oft/vla-scripts/deploy.py \
#   --pretrained_checkpoint /PATH/TO/FINETUNED/MODEL/CHECKPOINT/DIR/ \
#   --use_l1_regression True \
#   --use_film True \
#   --num_images_in_input 3 \
#   --use_proprio True \
#   --center_crop True \
#   --unnorm_key aloha1_put_X_into_pot_300_demos

python openvla-oft/vla-scripts/deploy.py \
  --pretrained_checkpoint $OFT_MODEL_PATH \
  --use_l1_regression True \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio True \
  --center_crop False \
  --unnorm_key bridge_dataset \
  --host $HOST \
  --port $PORT