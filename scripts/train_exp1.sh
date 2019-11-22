#!/bin/bash

CUDA_VISIBLE_DEVICES=0 screen python train.py \
    --exp_name exp1 \
    --dataset vg \

    --batch_size 32 \
    --num_iterations 1000000 \
    --learning_rate 0.0001 \
    
    --bbox_loss_weight 0.5 \
    --mask_loss_weight 0 \
    --pixel_loss_weight 0.5 \
    --img_gan_loss_weight 0.1 \
    --obj_gan_loss_weight 0.1 \
    --obj_cls_loss_weight 0.1