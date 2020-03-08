#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 train.py \
--cuda \
--content_image images/corgi.jpg \
--style_images images/picasso.jpg \
