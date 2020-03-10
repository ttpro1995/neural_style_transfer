import torchvision.models as models
cnn = models.vgg19(pretrained=True)
print(cnn)
