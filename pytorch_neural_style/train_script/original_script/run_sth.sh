#!/usr/bin/env bash
# $1 style file name 
echo "run $1"

CUDA_VISIBLE_DEVICES=1 python main.py \
--content ../contents/team_party.jpg \
--style ../styles/$1 \
--steps 1000 \
--output ./output_party_$1/  \
--save_every 20
