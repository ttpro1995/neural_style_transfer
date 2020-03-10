#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main.py \
--content ../contents/team_party.jpg \
--style ../styles/drawing_style1.jpg \
--steps 1000 \
--output ./output_party_drawing_style1  \
--save_every 20