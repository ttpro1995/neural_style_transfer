#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train.py \
--cuda \
--content_image images/corgi.jpg \
--style_image images/candy.jpg \
--outf output/  \
--niter 51
