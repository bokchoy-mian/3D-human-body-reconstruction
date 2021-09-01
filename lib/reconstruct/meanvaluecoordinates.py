#!/usr/bin/python
#-- coding:utf8 --
import multiprocessing
import numpy as np
import cv2 as cv
import math
import lib.reconstruct.Helper as h


def det(p1, p2):
    return p1[0] * p2[1] - p1[1] * p2[0]

def dot_prod(p1, p2):
    return p1[0] * p2[0] + p1[1] * p2[1]

def F(p, verts):
    """f(x)函数

    :param p:内点
    :param verts:边界点[[;(m,2)
    :return:
    """
    N = verts.shape[0]

    S = [0] * N
    for i in range (N):
        v = verts[i]
        S[i] = v - p
    # S=np.subtract(verts,p)
    R = [0] * N
    A = [0] * N
    D = [0] * N
    ret = [0] * N
    for i in range (N):
        i_1 = (i + 1) % N
        R[i] = h.distance (S[i], [0, 0])#判断是否属于边界点
        A[i] = det (S[i], S[i_1]) * 0.5#叉乘
        D[i] = dot_prod (S[i], S[i_1])#点乘
        if R[i] == 0:
            ret[i] = 1
            ret = np.array (ret)
            return ret
        if A[i] == 0 and D[i] < 0:#方向相反，且平行
            R[i_1] = h.distance (S[i_1], [0, 0])
            ret[i] = 1.0 / (R[i] + R[i_1])
            ret[i_1] = R[i] * 1.0 / (R[i] + R[i_1])
            ret = np.array (ret)
            return ret

    for i in range (N):
        ip1 = (i + 1) % N
        im1 = i - 1
        w = 0
        if A[im1] != 0:
            w = w + (R[im1] - D[im1] * 1.0 / R[i]) * 1.0 / A[im1]
        if A[i] != 0:
            w = w + (R[ip1] - D[i] * 1.0 / R[i]) * 1.0 / A[i]
        ret[i] = w

    ret = np.array (ret)
    s = np.sum (ret)
    ret = ret / s
    return ret

def get_mv_coordinate(in_pts, b_pts):
    """

    :param in_pts:内点集
    :param b_pts:二维边界点
    :return:
    """
    # M = in_pts.shape[0]  # M,2
    # N = b_pts.shape[0]  # N,2
    cord = []
    # dp = 0
    for pt in in_pts:
        w = F (pt, b_pts)
        # pp = np.dot(w, b_pts)
        # dp += distance(pt, pp)
        cord.append (w)

    cord = np.array (cord)  # shape:M,N
    # print("mead dp:",dp/M)
    print ("cord shape", cord.shape)
    return cord

def F_miltiporocessing(param):
    return F(param[0],param[1])

def get_mv_coordinate_multiprocessing(in_pts, b_pts):
    """

    :param in_pts:
    :param b_pts:二维边界点
    :return:
    """

    items = [(x,b_pts) for x in in_pts]
    p = multiprocessing.Pool (5)
    cord = p.map (F_miltiporocessing, items)
    p.close ()
    p.join ()

    cord = np.array (cord)  # shape:M,N
    # print("mead dp:",dp/M)
    print ("cord shape", cord.shape)
    return cord

if __name__ == "__main__":
    pass