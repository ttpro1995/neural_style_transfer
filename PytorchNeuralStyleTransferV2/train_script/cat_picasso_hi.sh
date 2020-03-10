#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train.py \
--name cat_picasso_hi_5_100   \
--cuda \
--content_image content/cat.jpg \
--style_image style/picasso.jpg \
--outf output/  \
--content_weight 5 \
--style_weight 100 \
--imageSize 256 \
--save_niter  10 \
--color_histogram_matching  \
--niter 51