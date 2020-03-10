#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main.py \
--content ../contents/team_party.jpg \
--style ../styles/water_color1.jpg \
--steps 1000 \
--output ./output_party_water_color1.jpg  \
--save_every 20