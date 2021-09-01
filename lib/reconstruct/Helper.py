#!/usr/bin/python
#-- coding:utf8 --
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
##工具函数

#可视化
def draw_points(pts):
    """
    :param pts:
    :return:
    """
    fig = plt.figure ()
    axe = fig.add_subplot (111)
    X, Y = [], []
    for point in pts:
        X.append (point[0])
        Y.append (-point[1])
    axe.plot (X, Y, 'ro')
    fig.show ()

def dispCorres(img_size, rgb_bound, smpl_bound, match):
    """
    显示轮廓
    :param img_size: 图片大小
    :param rgb_bound: 图片轮廓
    :param smpl_bound: smpl轮廓
    :param match: 对应值
    :return:
    """

    disp = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    # cv2.drawContours(disp, rgb_bound, -1, (0, 255, 0), 1)  # green
    # cv2.drawContours(disp, smpl_bound, -1, (255, 0, 0), 1)  # blue

    # rgb_bound = np.array(rgb_bound)
    # smpl_bound = np.array(smpl_bound)

    len = rgb_bound.shape[0]
    for i in range(0, len, 10):  # do not show all the points when display
        cv2.circle(disp, (rgb_bound[i, 0], rgb_bound[i, 1]), 1, (255, 0, 0), -1)
        corresPoint = smpl_bound[match[i]]
        cv2.circle(disp, (corresPoint[0], corresPoint[1]), 1, (0, 255, 0), -1)
        cv2.line(disp, (rgb_bound[i, 0], rgb_bound[i, 1]), (corresPoint[0], corresPoint[1]),
                 (255, 255, 255), 1)

    cv2.imshow('point correspondence', disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_matching(b1, b2):
    """

    :param b1:
    :param b2:
    :return:
    """
    L = b1.shape[0]

    fig = plt.figure ()
    axe = fig.add_subplot (111)
    X, Y = [], []
    X1, Y1 = [], []
    fig.show ()
    for i in range (L):
        p1 = b1[i]
        p2 = b2[i]
        X.append (p1[0])
        Y.append (p1[1])
        axe.cla ()
        axe.plot (X, Y, 'ro')
        X1.append (p2[0])
        Y1.append (p2[1])
        axe.plot (X1, Y1, 'bo')
        fig.canvas.draw ()
    fig.show ()


def distance(p1, p2):
    """

    :param p1:
    :param p2:
    :return:
    """
    return math.sqrt(math.pow(p1[0]-p2[0],2) + math.pow((p1[1]-p2[1]),2))

def to_closest(x):
    """

    :param x:
    :return:
    """
    i = math.floor(x)
    if x-i<0.5:
        return i
    else:
        return i+1