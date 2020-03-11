import torchvision.transforms as transforms
import util
import torch

transform = transforms.Compose([
    #transforms.Scale(opt.imageSize),  #  UserWarning: The use of the transforms.Scale transform is deprecated,
    transforms.Resize(256),  #please use transforms.Resize instead.  warnings.warn("The use of the transforms.Scale transform is deprecated, " +
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],std=[1,1,1]),
    transforms.Lambda(lambda x: x.mul_(255)),
    ])

styleImg = transform(util.open_and_resize_image("../style/picasso.jpg", 256))  # 1x3x512x512
contentImg = transform(util.open_and_resize_image("../content/cat.jpg", 256))  # 1x3x512x512
styleImg = styleImg.unsqueeze(0)
contentImg = contentImg.unsqueeze(0)
styleImg, contentImg, content_iq, style_iq = util.luminance_transfer(styleImg.numpy(), contentImg.numpy())

print(content_iq.shape)