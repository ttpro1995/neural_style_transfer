CUDA_VISIBLE_DEVICES=2 nohup python train.py \
--name quy_oil_o_C1_T200 \
--cuda \
--content_image content/quy.jpg \
--style_image style/oil_pastel.png \
--content_weight 1 \
--style_weight 200 \
--outf output/  \
--save_niter  50 \
--lr 10 \
--niter 1001 > log/quy_oil_o_C1_T200.log  &
