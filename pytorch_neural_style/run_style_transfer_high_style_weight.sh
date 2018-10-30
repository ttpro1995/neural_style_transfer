#!/usr/bin/env bash
# $1 content 
# $2 style 
# $3 gpu id 
echo "run $1 $2 on gpu $3"

CUDA_VISIBLE_DEVICES=$3 python main.py \
--content ../contents/$1 \
--style ../styles/$2 \
--steps 1000 \
--output ./output_high_$1_$2/  \
--save_every 20 \
--style_weight 100000000 
