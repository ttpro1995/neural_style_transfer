#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train.py \
--name cat_picasso_lu_5_100_sv   \
--cuda \
--content_image content/cat.jpg \
--style_image style/picasso.jpg \
--outf output/  \
--content_weight 5 \
--style_weight 100 \
--imageSize 256 \
--save_niter  10 \
--luminance_only  \
--niter 51 > log/cat_picasso_lu_5_100_sv.log  &