from __future__ import print_function
from PIL import Image
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt
import six
import colorsys
import time

def open_and_resize_image(path, size):

    image = Image.open(path).convert('RGB')
    width, height = image.size
    if (height > width):
        target_width = size
        target_height = int(round(float(height * target_width) / width))
    else:
        target_height = size
        target_width = int(round(float(width * target_height) / height))

    image = image.resize((target_width, target_height), Image.BILINEAR)
    return image

def match_color_histogram(x, y):
    z = np.zeros_like(x)
    shape = x[0].shape
    for i in six.moves.range(len(x)):
        a = x[i].reshape((3, -1))
        a_mean = np.mean(a, axis=1, keepdims=True)
        a_var = np.cov(a)
        d, v = np.linalg.eig(a_var)
        d += 1e-6
        a_sigma_inv = v.dot(np.diag(d ** (-0.5))).dot(v.T)

        b = y[i].reshape((3, -1))
        b_mean = np.mean(b, axis=1, keepdims=True)
        b_var = np.cov(b)
        d, v = np.linalg.eig(b_var)
        b_sigma = v.dot(np.diag(d ** 0.5)).dot(v.T)

        transform = b_sigma.dot(a_sigma_inv)
        z[i,:] = (transform.dot(a - a_mean) + b_mean).reshape(shape)
    return z

def bgr_to_yiq(x):
    transform = np.asarray([[0.114, 0.587, 0.299], [-0.322, -0.274, 0.596], [0.312, -0.523, 0.211]], dtype=np.float32)
    n, c, h, w = x.shape
    x = x.transpose((1, 0, 2, 3)).reshape((c, -1))
    x = transform.dot(x)
    return x.reshape((c, n, h, w)).transpose((1, 0, 2, 3))

def yiq_to_bgr(x):
    transform = np.asarray([[1, -1.106, 1.703], [1, -0.272, -0.647], [1, 0.956, 0.621]], dtype=np.float32)
    n, c, h, w = x.shape
    x = x.transpose((1, 0, 2, 3)).reshape((c, -1))
    x = transform.dot(x)
    return x.reshape((c, n, h, w)).transpose((1, 0, 2, 3))

def split_bgr_to_yiq(x):
    x = bgr_to_yiq(x)
    y = x[:,0:1,:,:]
    iq = x[:,1:,:,:]
    return np.repeat(y, 3, axis=1), iq

def join_yiq_to_bgr(y, iq):
    y = bgr_to_yiq(y)[:,0:1,:,:]
    return yiq_to_bgr(np.concatenate((y, iq), axis=1))

def join_y_without_iq(y, iq):
    y = bgr_to_yiq(y)[:,0:1,:,:]
    iq_zeros = np.zeros(iq.shape)
    return yiq_to_bgr(np.concatenate((y, iq_zeros), axis=1))

def join_y_without_iq2(y, i, q):
    y = bgr_to_yiq(y)[:,0:1,:,:]
    return yiq_to_bgr(np.concatenate([y, i, q], axis=1))

def join_i_without_yq(i, y, q):
    i = bgr_to_yiq(i)[:,0:1,:,:]
    return yiq_to_bgr(np.concatenate([y, i, q], axis=1))

def join_q_without_yi(q, y, i):
    q = bgr_to_yiq(q)[:,0:1,:,:]
    return yiq_to_bgr(np.concatenate([y, i, q], axis=1))

def luminance_transfer(x,y):
    # x: style, y:content
    x_l, x_iq = split_bgr_to_yiq(x) # 1x3x512x512
    y_l, y_iq = split_bgr_to_yiq(y)

    x_l_mean = np.mean(x_l)
    y_l_mean = np.mean(y_l)
    x_l_std = np.std(x_l)
    y_l_std = np.std(y_l)

    x_l = (y_l_std/x_l_std)*(x_l - x_l_mean) + y_l_mean
    return x_l, y_l, y_iq, x_iq

def luminance_transfer_full(x,y):
    # x: style, y:content
    x_l, x_iq = split_bgr_to_yiq(x) # 1x3x512x512
    y_l, y_iq = split_bgr_to_yiq(y)

    x_l_mean = np.mean(x_l)
    y_l_mean = np.mean(y_l)
    x_l_std = np.std(x_l)
    y_l_std = np.std(y_l)

    x_l = (y_l_std/x_l_std)*(x_l - x_l_mean) + y_l_mean
    return x_l, y_l, y_iq, x_iq


def get_readable_time():
    """
    make human readable time with format year-month-day hour-minute
    :return: a string of human readable time (ex: '2020-02-24 10:31' )
    """
    return time.strftime('%Y-%m-%d %H:%M')