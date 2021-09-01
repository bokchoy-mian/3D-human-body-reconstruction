#!/usr/bin/python
#-- coding:utf8 --
"""
Preprocessing stuff.
"""
import numpy as np
import cv2


def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)#强行将图片转到224X224
    new_img = cv2.resize(img, (new_size[1], new_size[0]))#强行将图片转到224X224
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def scale_and_crop(image, scale, center, img_size):
    """

    :param image: 原图
    :param scale: 压缩比
    :param center: 中心点
    :param img_size: 标准图像大小
    :return:
    """
    """
    输入：原始图像放缩比
        输出：224X224的图像crop 和相关参数
        过程: 若图片大于224则将图片下采样到224，再将小于224的那一边通过边缘复制的方法扩展到224
    """
    image_scaled, scale_factors = resize_img(image, scale) #将图片强制转换成标准大小，并返回实际压缩比
    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]#为何要倒过来？
    center_scaled = np.round(center * scale_factors).astype(np.int)

    margin = int(img_size / 2)
    image_pad = np.pad(
        image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')# 图像以边缘复制的方法向左右上下各扩充112
    center_pad = center_scaled + margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    #注意这里下标是反的，因为在坐标系中和在矩阵中像素坐标是相反的
    # 前面将图像长或宽其中像素大的一边强制转换为224
    # 这里则将另一边也转为224
    proc_param = {
        'scale': scale,
        'center': center,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': image.shape[:2]
    }

    return crop, proc_param
