#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train.py \
--name picasso_corgi_lu  \
--cuda \
--content_image images/corgi.jpg \
--style_image images/candy.jpg \
--outf output/  \
--luminance_only \
--niter 51
