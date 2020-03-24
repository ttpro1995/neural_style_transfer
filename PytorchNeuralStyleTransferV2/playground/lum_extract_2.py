from util import bgr_to_yiq, yiq_to_bgr
import util
import numpy as np
from skimage.color import convert_colorspace
from PIL import Image
styleImg = util.open_and_resize_image("../style/picasso.jpg", 256)  # 1x3x512x512
contentImg = util.open_and_resize_image("../content/cat.jpg", 256)  # 1x3x512x512

contentImgNp = np.expand_dims(np.asarray(contentImg), 0) # n . h . w . c
print(contentImgNp.shape)
contentImgNp = contentImgNp.transpose(0, 3, 1, 2) # to n, c, h, w
print(contentImgNp.shape)

yiq = bgr_to_yiq(contentImgNp)
bgr_revert = yiq_to_bgr(yiq)

y = yiq[:,0:1, :, :]
i = yiq[:,1:2, :, :]
q = yiq[:,2:3, :, :]
z = np.zeros((1, 1, 256, 341))

yiq00 = np.concatenate([y,i,q], axis=1)
print(yiq00.shape)
yiq00RGB = yiq_to_bgr(yiq00)
yiq00RGBout = np.squeeze(yiq00RGB.transpose(0, 2, 3, 1), axis=0)  # n h w c

bgr_revert_out = np.squeeze(contentImgNp.transpose(0, 2, 3, 1), axis=0)  # n h w c

print(yiq00RGBout.shape)

img = Image.fromarray(bgr_revert_out, 'RGB')
img.save('../output/lu_sample/y0.png')
# # contentImgNp.transpose()
#
#
# # img = Image.fromarray(contentImgOutAgain, 'RGB')
# # img.save('../output/lu_sample/y0.png')
#
#
