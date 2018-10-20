from PIL import Image
import torchvision.transforms as transforms
import torch
import logging


class DataManager():
    def __init__(self, device, content_path = None, style_path = None):
        """

        :param device: cuda or cpu
        """
        self.content_path = None
        self.style_path = None
        self.device = device
        self.set_content_path(content_path)
        self.set_style_path(style_path)

    def set_content_path(self, content_path):
        """
        set path to content image
        :param content_path:
        :return:
        """
        self.content_path = content_path

    def set_style_path(self, style_path):
        """
        set path to style image
        :param style_path:
        :return:
        """
        self.style_path = style_path

    def load_image(self, size=None):
        """
        load content and size image
        resize content image to size (keep aspect)
        resize style image to size of content image
        load as tensor
        :param size: resize both content and style image to size (keep aspect)
        :return:
        """
        original_content = Image.open(self.content_path)
        original_style = Image.open(self.style_path)
        if size is None:
            size = original_content.size[0]
        size = int(size)
        size2 = int(size * 1.0 / original_content.size[0] * original_content.size[1])
        self.size1 = size
        self.size2 = size2
        loader = transforms.Compose([
            transforms.Resize((self.size2, self.size1)),  # scale imported image h w
            transforms.ToTensor()])  # transform it into a torch tensor
        self.content = loader(original_content).unsqueeze(0).to(self.device, torch.float)
        self.style = loader(original_style).unsqueeze(0).to(self.device, torch.float)
        logging.info("loaded image with size " + str(self.size1) + " " + str(self.size2))

    def get_content_image(self):
        return self.content.clone()

    def get_style_image(self):
        return self.style.clone()

