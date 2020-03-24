import util
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import os
###############   DATASET   ##################
transform = transforms.Compose([
    #transforms.Scale(opt.imageSize),  #  UserWarning: The use of the transforms.Scale transform is deprecated,
    transforms.Resize(256),  #please use transforms.Resize instead.  warnings.warn("The use of the transforms.Scale transform is deprecated, " +
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],std=[1,1,1]),
    transforms.Lambda(lambda x: x.mul_(255)),
    ])
def load_image(path,style=False):
    img = Image.open(path)
    img = Variable(transform(img))
    img = img.unsqueeze(0)
    return img

def save_image(img, epoch, luminance_only=False, note=""):
    post = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
         transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1]),
         transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
         ])
    img = post(img)
    img = img.clamp_(0,1)
    folder = os.path.join("../output", "lu_sample")
    if not os.path.exists(folder):
        os.mkdir(folder)
    save_path = os.path.join(folder, "lu_sample" + "_" + str(epoch))
    if luminance_only:
        save_path+= "_luminance_only"
    if len(note) > 0:
        save_path+= note
    save_path+= ".png"
    vutils.save_image(img,
                save_path,
                normalize=True)
    return


styleImg = transform(util.open_and_resize_image("../style/picasso.jpg", 256))
contentImg = transform(util.open_and_resize_image("../content/cat.jpg", 256))  # 1x3x512x512
styleImg = styleImg.unsqueeze(0)
contentImg = contentImg.unsqueeze(0)
styleImg, contentImg, content_iq, style_iq = util.luminance_transfer(styleImg.numpy(), contentImg.numpy())
content_i = content_iq[:, 0:1, :, :]
content_q = content_iq[:, 1:2, :, :]
content_y = contentImg[:, 0:1, :, :]
content_y0 = np.zeros(content_y.shape)
content_i0 = np.zeros(content_i.shape)
content_q0 = np.zeros(content_q.shape)

print(content_y0.shape)
print(content_i0.shape)
print(content_q0.shape)

styleImg = Variable(torch.from_numpy(styleImg))
contentImg = Variable(torch.from_numpy(contentImg))

# print style and content image
styleImgOut = styleImg.data[0].cpu()
contentImgOut = contentImg.data[0].cpu()

styleImgOut = np.expand_dims(styleImgOut.numpy(), 0)

styleImgOut = util.join_y_without_iq(styleImgOut, style_iq)
save_image(torch.from_numpy(styleImgOut).squeeze(), 0, luminance_only=True,
           note="style_black_white_")  # save lu channel

contentImgOut = np.expand_dims(contentImgOut.numpy(), 0)
contentImgOutJoin = util.join_y_without_iq(contentImgOut, content_iq)
save_image(torch.from_numpy(contentImgOutJoin).squeeze(), 0, luminance_only=True,
           note="content_start_black_white_")  # save lu channel
contentImgOutJoinv2 = util.join_y_without_iq2(contentImgOut, content_i0, content_q0)
save_image(torch.from_numpy(contentImgOutJoinv2).squeeze(), 0, luminance_only=True,
           note="content_start_black_white_v2")  # save lu channel

contentImgOutJoinv2 = util.join_y_without_iq2(contentImgOut, content_i0, content_q0)
save_image(torch.from_numpy(contentImgOutJoinv2).squeeze(), 0, luminance_only=True,
           note="content_start_black_white_v2")  # save lu channel

contentImgOut_I_Join = util.join_i_without_yq(np.repeat(content_i, 3, axis=1), content_y0, content_q0)
save_image(torch.from_numpy(contentImgOut_I_Join).squeeze(), 0, luminance_only=True,
           note="content_i")  # save lu channel


contentImgOut_Q_Join = util.join_q_without_yi(np.repeat(content_q, 3, axis=1), content_y0, content_i0)
save_image(torch.from_numpy(contentImgOut_Q_Join).squeeze(), 0, luminance_only=True,
           note="content_q")  # save lu channel

# contentImgOutQJoin = util.join_i_without_yq(np.repeat(content_i, 3, axis=1), content_y0, content_q0)
# save_image(torch.from_numpy(contentImgOutIJoin).squeeze(), 0, luminance_only=True,
#            note="content_i")  # save lu channel