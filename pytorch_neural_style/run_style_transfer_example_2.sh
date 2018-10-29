#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main.py \
--content ../contents/team_party.jpg \
--style ../styles/style_painting_people.jpg \
--steps 1000 \
--output ./output_party_style_painting_people  \
--save_every 20