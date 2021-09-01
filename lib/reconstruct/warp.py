#!/usr/bin/python
#-- coding:utf8 --
import numpy as np
import cv2
from lib.reconstruct.meanvaluecoordinates import get_mv_coordinate,get_mv_coordinate_multiprocessing

def get_corres(smpl_inner_points, smpl_match_bound, rgb_bound, file_name=None):
    """
    计算变换后的内点位置

    :param smpl_inner_points:
    :param smpl_match_bound:
    :param rgb_bound:
    :param file_name:
    :return:
    """
    # mv_cord = get_mv_coordinate(smpl_inner_points, smpl_match_bound)
    mv_cord = get_mv_coordinate_multiprocessing (smpl_inner_points, smpl_match_bound)

    pts_to = np.dot(mv_cord, rgb_bound)
    if file_name != None:
        np.save(file_name,pts_to.astype(int))
    return pts_to.astype(int)

def warp_normal_map(src_img, smpl_inner_points, smpl_match_bound, rgb_bound,out_path=None):
    """

    :param smpl_inner_points:
    :param smpl_match_bound:
    :param rgb_bound:
    :return: 返回warp后的normal_map
    """
    smpl_warp_inner_points=get_corres(smpl_inner_points,smpl_match_bound,rgb_bound)
    np.save ('data/baoluo/mv_cord.npy', smpl_warp_inner_points)
    # smpl_warp_inner_points=np.load('mv_cord.npy').astype(int)
    warp_img=np.zeros(src_img.shape)


    for i in range(smpl_inner_points.shape[0]):
        x1, y1 = smpl_inner_points[i, :]
        x2, y2 = smpl_warp_inner_points[i, :]
        # for c in range(3):
        #     warp_img[y2, x2, c] = src_img[y1, x1, c]
        warp_img[y2, x2,:]=src_img[y1,x1,:]
        # print('%d/%d warp map' % (i, pts_num))
    if out_path:
        cv2.imwrite(out_path, warp_img)

    # cv2.imshow('q',warp_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return warp_img

def warp_map(input_map,inner_points,warped_inner_points,out_path=None):
    warp_img = np.zeros (input_map.shape)
    h,w,_=input_map.shape
    for i in range (inner_points.shape[0]):
        x1, y1 = inner_points[i, :]
        x2, y2 = warped_inner_points[i, :]
        x2=0 if x2>=w else x2
        y2 = 0 if y2 >= h else y2
        # for c in range(3):
        #     warp_img[y2, x2, c] = src_img[y1, x1, c]
        warp_img[y2, x2, :] = input_map[y1, x1, :]
        # print('%d/%d warp map' % (i, pts_num))
    if out_path:

        cv2.imwrite (out_path, (warp_img* 255).astype ('uint8')[:, :, ::-1])

    return warp_img
