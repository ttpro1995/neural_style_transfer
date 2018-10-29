#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main.py \
--content ../contents/team_party.jpg \
--style ../styles/pen_landscape_wc5021.jpg \
--steps 1000 \
--output ./output_party_pen_landscape_wc5021  \
--save_every 20