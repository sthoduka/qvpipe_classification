#!/bin/sh
python main.py \
  --video_root=classification_track \
  --batch_size=64 \
  --group=4 \
  --crop_size=300 \
  --resize_size=224 \
  --default_root_dir=logs/ \
  --learning_rate=0.1 \
  --max_epochs=60 \
  --gpus=1 \
  --accelerator=gpu \
#  --checkpoint=path_to_previous_checkpoint_if_applicable.ckpt

