from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import copy
import os
from option import parse_args
from data_manager import DataManager
from tutorial_net import run_style_transfer
import logging

if __name__ == "__main__":
    logging.basicConfig(filename='pytorch_neural_style.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    arg = parse_args()
    logging.info(arg)

    dataman = DataManager(device, arg.content, arg.style)
    dataman.load_image(arg.size)

    style_img = dataman.get_style_image()
    content_img = dataman.get_content_image()
    input_img = dataman.get_content_image()

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, device, output_dir=arg.output, num_steps=arg.num_steps)

    torchvision.utils.save_image(output, os.path.join(arg.output, "final_output.png"))

