#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main.py \
--content ../contents/team_party.jpg \
--style ../styles/pen_stalin.jpg \
--steps 1000 \
--output ./output_party_pen_stalin  \
--save_every 20