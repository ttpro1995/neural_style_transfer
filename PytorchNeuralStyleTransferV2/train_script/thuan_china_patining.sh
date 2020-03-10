#CUDA_VISIBLE_DEVICES=0 python train.py \
#--name thuan_china_o_C40_T100_ \
#--cuda \
#--content_image content/thuan.jpg \
#--style_image style/china_painting.jpg \
#--content_weight 40 \
#--style_weight 100 \
#--outf output/  \
#--save_niter  10 \
#--lr 5 \
#--niter 1001


CUDA_VISIBLE_DEVICES=0 python train.py \
--name thuan_china_lu_C40_T100_ \
--cuda \
--content_image content/thuan.jpg \
--style_image style/china_painting.jpg \
--content_weight 40 \
--style_weight 100 \
--outf output/  \
--save_niter  10 \
--lr 5 \
--luminance_only \
--niter 1001