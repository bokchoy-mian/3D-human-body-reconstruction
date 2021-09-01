#!/usr/bin/python
#-- coding:utf8 --
import time
import cv2 as cv
import numpy as np
from lib.reconstruct.Helper import *
#获取边界点
def get_boundary(img, thresh, combine=True):
    """轮廓检测

    :param img:原图
    :param thresh:阈值
    :param combine:是否连成线
    :return:
    """
    if len(img.shape)==3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)#转为灰度图
    ret, thresh = cv.threshold(img, thresh, 255,cv.THRESH_BINARY)#二值化，，thresh为输出
    thresh = cv.medianBlur(thresh,5)#中值滤波
    # cv.imshow('tresh',thresh)
    # cv.waitKey ()
    # kernel = np.ones((7,7),np.uint8)
    # thresh = cv.erode(thresh, kernel,iterations = 1)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # res = cv.drawContours(img,contours,0, (0, 0, 255), 3)
    # cv.imshow('_',res)
    # cv.waitKey()
    # cv.destroyAllWindows()
    N = len(contours)
    contour = []
    print(N)
    if combine:
        for i in range(N):
            c = contours[i]
            if(len(c)<100):
                continue
            if contour ==[]:
                contour = c
            else:
                contour = np.concatenate((c,contour),axis=0)
    else:
        contour = contours[0]
    contour = contour.reshape(contour.shape[0],2)
    print("boudary before fill - shape :",contour.shape)
    return contour

def get_boundary_from_mask(mask):
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contour = contours[0].reshape (contours[0].shape[0], 2)
    # img=np.zeros(mask.shape)
    # for pt in contour:
    #     img[pt[1]][pt[0]]=255
    # cv2.imshow('1',img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return contour

def fill_points(pts, min_dis, max_dis=10):
    """线性插值法填充使其更密集
    :param pts:所有边界和边界内的点
    :param min_dis:
    :param max_dis:
    :return:
    """
    full = False
    print ("point number before fill:%d" % (pts.shape[0]))
    while not full:
        full = True
        for i in range (pts.shape[0]):
            dis = distance (pts[i], pts[i - 1])
            if dis > min_dis and dis < max_dis:
                # print(dis)
                full = False
                x = (pts[i][0] + pts[i - 1][0]) / 2
                y = (pts[i][1] + pts[i - 1][1]) / 2
                pts = np.insert (pts, i, [x, y], axis=0)

    print ("point number after fill:%d" % (pts.shape[0]))
    return pts

def select_points(pts, N=300):
    """从边界上等距离挑选点

    :param pts:边界点
    :param N: 数目
    :return:
    """
    n = pts.shape[0]
    print("point number before select:%d"%(n))
    if n<N:
        print("sample number is too large!")
    step = n*1.0/N
    ret = []
    for i in range(N):
        ret.append(pts[math.floor(i*step)])
    print("point number selected:%d"%(N))
    ret = np.array(ret)
    return ret

def getinnerpts(image):
    """
    读取内部点
    :param image:
    :return:
    """
    h, w = image.shape[:2]
    c = len(image.shape)

    inner_pts = []

    if c == 3:
        for i in range(0, h):
            for j in range(0, w):
                # if image[i, j, 0] != 255 or image[i, j, 1] != 255 or image[i, j, 2] != 255:
                if (image[i,j,:] !=[255,255,255]).all():
                    location = np.array([j, i])
                    inner_pts.append(location)

    if c == 2:
        for i in range(0, h):
            for j in range(0, w):
                if image[i, j] != 0:
                    location = np.array([j, i])
                    inner_pts.append(location)

    inner_pts = np.array(inner_pts)

    return inner_pts

if __name__ == "__main__":
    dr = 'data/baoluo/'
    rgb_mask = cv.imread (dr + 'baoluomask.png',cv.IMREAD_GRAYSCALE)
    smpl_mask = cv.imread (dr + 'smpl_mask.png', cv.IMREAD_GRAYSCALE)
    smpl_normal = cv.imread (dr + 'nomalsMap.png')
    bound=get_boundary_from_mask(rgb_mask)
    smpl_bond = get_boundary (smpl_mask, 1, False)
    rgb_bond = get_boundary (rgb_mask, 1, False)

    from_N = smpl_bond.shape[0]
    to_N = rgb_bond.shape[0]
    sp_N = min (from_N, to_N)
    # draw_points(smpl_bond)
    # draw_points(rgb_bond)

    # 挑选边界点

    smpl_mv_bound_points = select_points (fill_points (smpl_bond, min_dis=3, max_dis=60), N=int (sp_N * 0.8))
    # target
    rgb_mv_bound_points = select_points (fill_points (rgb_bond, min_dis=3, max_dis=60), N=int (sp_N * 0.5))

    # draw_points (smpl_mv_bound_points)
    # draw_points (rgb_mv_bound_points)

    smpl_innerpoints=getinnerpts(smpl_normal)

    from meanvaluecoordinates import get_mv_coordinate

    startime=time.time()
    smpl_innerpoints_mv=get_mv_coordinate(smpl_innerpoints,smpl_mv_bound_points)
    print ('--time:  %d', time.time () - startime)

    print('done')