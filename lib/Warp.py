#!/usr/bin/python
#-- coding:utf8 --
import pickle
import time
import matplotlib.pyplot as plt
import os
import math
import cv2
import numpy as np
from models.smplh_np import SMPLHModel
from numpy import linalg
from sklearn.neighbors import NearestNeighbors
from lib.reconstruct.meanvaluecoordinates import get_mv_coordinate_multiprocessing
from lib.reconstruct.meanvaluecoordinates import F as MVC
import PIL.Image as pil_img

class Wrap():
    def __init__(self,rgb_mask,smplh_value,outpath):
        self.out_path=outpath
        # self.rgb_img=rgb_img
        self.rgb_mask=rgb_mask
        self.smplh_value=smplh_value #(h,w,_)
        # self.smplh_mask = np.where (self.smplh_value[:,:,0] == 1, 0, 255).astype('uint8')
        self.smplh_mask = np.where (np.logical_or(np.all(self.smplh_value[:, :, 0:3] == 1,axis=2),np.all(self.smplh_value[:, :, 3:6] == 1,axis=2)), 0, 255).astype ('uint8')
        # self.smplh_mask = np.where (np.any (self.smplh_value[:, :, 0:6] != 1, axis=2), 0, 255).astype ('uint8')

        # self.smplh_mask = cv2.morphologyEx (self.smplh_mask, cv2.MORPH_CLOSE, np.ones ((int (3), int (3)), np.uint8))  # 填充孔洞
    def __call__(self):
        self.rgb_bound = self.get_boundary (self.rgb_mask, 1)#(n,2) [:,0]->w ;[:,1]->h
        self.smplh_bound = self.get_smplh_boundary (self.smplh_mask,1)#(m,2) [:,0]->w ;[:,1]->h
        # self.draw_points(self.rgb_bound)
        # self.draw_points (self.smplh_bound[::5])
        self.match=self.boundary_match(self.smplh_bound,self.rgb_bound,64)
        self.smplh_bound_match = self.smplh_bound[self.match]#(n,2) [:,0]->w ;[:,1]->h
        self.draw_points(self.smplh_bound_match)
        # self.draw_matching(self.rgb_bound,self.smplh_bound_match)
        self.smplh_innerpoints = self.getinnerpts (self.smplh_mask)#(n,2) [:,0]->w ;[:,1]->h
        self.rgb_innerpoints=self.getinnerpts(self.rgb_mask)#(n,2) [:,0]->w ;[:,1]->h
        self.smplh_warp_inner_points = self.get_corres (self.smplh_innerpoints, self.smplh_bound_match,
                                             self.rgb_bound, os.path.join(self.out_path ,'mv_points.npy')) #(n,2) [:,0]->w ;[:,1]->h
        # self.smplh_warp_inner_points =np.load(self.out_path + 'mv_points.npy')
        self.warp_smplh_value=self.warp_map(self.smplh_value,self.smplh_innerpoints,self.smplh_warp_inner_points) #(h,w,_)
        # self.show_img(self.warp_smplh_value)
        self.fill_smplh_value=self.holefilling(self.warp_smplh_value,self.rgb_innerpoints,self.rgb_mask)

        return self.fill_smplh_value

    def get_smplh_boundary(self,mask,eps):
        """

        :param mask:
        :param eps: 选取率
        :return:
        """
        contours, hierarchy = cv2.findContours (mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour = contours[0].reshape (contours[0].shape[0], 2)
        n = contour.shape[0]
        N=int (n * eps)
        step = n * 1.0 / N
        ret = []
        for i in range (N):
            ret.append (contour[math.floor (i * step)])

        ret = np.array (ret)

        return ret

    def get_boundary(self,img, thresh):
        """轮廓检测
        :param img:原图
        :param thresh:阈值
        :return:
        """
        if len(img.shape)==3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转为灰度图
        ret, thresh = cv2.threshold(img, thresh, 255,cv2.THRESH_BINARY)#二值化，，thresh为输出
        # thresh = cv2.medianBlur(thresh,5)#中值滤波
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour = contours[0]
        contour = contour.reshape(contour.shape[0],2)

        return contour

    def pairwise_dist(self,A, B):
        # squared norms of each row in A and B
        na = np.sum (np.square (A), axis=1)
        nb = np.sum (np.square (B), axis=1)

        # na as a row and nb as a column vectors
        na = np.reshape (na, [-1, 1])
        nb = np.reshape (nb, [1, -1])

        # return pairwise euclidead difference matrix
        D = np.maximum (na - 2 * np.matmul (A, B.T) + nb, 0.0)

        return D

    def boundary_match(self,smpl_bound, rgb_bound, k):
        """动态规划求解边界对应点

        :param smpl_bound:
        :param rgb_bound:
        :param k:
        :return:
        """

        def distance(p1, p2):
            """

            :param p1:
            :param p2:
            :return:
            """
            return math.sqrt (math.pow (p1[0] - p2[0], 2) + math.pow ((p1[1] - p2[1]), 2))

        # dis_a2b = self.pairwise_dist (smpl_bound, rgb_bound)
        # index = np.unravel_index (np.argmin (dis_a2b, axis=None), dis_a2b.shape)
        # smpl_bound=np.roll (smpl_bound, -index[0], axis=0)
        # rgb_bound=np.roll (rgb_bound, -index[1], axis=0)
        # dis_a2b = self.pairwise_dist (smpl_bound, rgb_bound)
        # index = np.unravel_index (np.argmin (dis_a2b, axis=None), dis_a2b.shape)
        smpl_len = smpl_bound.shape[0]
        rgb_len = rgb_bound.shape[0]
        dp = []

        min = 999999
        tmp = []
        for i in range (smpl_len):
            dist = distance (rgb_bound[0], smpl_bound[i])
            if (dist < min):
                min = dist
                tmp.append ((min, i))
            else:
                tmp.append (tmp[-1])
        dp.append (tmp)

        for i in range (1, rgb_len):
            tmp = []
            prev = dp[i - 1]
            p = rgb_bound[i]
            tmp.append ((999999, -1))
            for j in range (1, smpl_len):#range (i, smpl_len)??对不对
                d1 = tmp[j - 1][0]
                prev = dp[i - 1][j - 1]
                if (j > prev[1] and j - prev[1] <= k):
                    dist = distance (p, smpl_bound[j])
                    d2 = dist + prev[0]
                    if (d1 < d2):
                        tmp.append (tmp[-1])
                    else:
                        tmp.append ((d2, j))
                else:
                    tmp.append (tmp[-1])
            # print(tmp[-1])
            dp.append (tmp)

        match = []
        idx = smpl_len - 1
        for i in range (rgb_len - 1, -1, -1):
            idx = dp[i][idx][1]
            match = [idx] + match
            idx -= 1

        return match

    def getinnerpts(self,image):
        """
        读取内部点
        :param image:(h,w)
        :return:
        """
        # h, w = image.shape[:2]
        # c = len (image.shape)
        #
        # inner_pts = []
        #
        # if c == 3:
        #     for i in range (0, h):
        #         for j in range (0, w):
        #             # if image[i, j, 0] != 255 or image[i, j, 1] != 255 or image[i, j, 2] != 255:
        #             if (image[i, j, :] != [255, 255, 255]).all ():
        #                 location = np.array ([j, i])
        #                 inner_pts.append (location)
        #     location = np.argwhere(np.all(image!=[255, 255, 255],axis=3))
        # if c == 2:
        #     for i in range (0, h):
        #         for j in range (0, w):
        #             if image[i, j] != 0:
        #                 location = np.array ([j, i])
        #                 inner_pts.append (location)
        #     location = np.roll(np.argwhere (image != 0),1,axis=1)
        # inner_pts = np.array (inner_pts)
        inner_pts = np.roll (np.argwhere (image != 0), 1, axis=1)
        return inner_pts #(n,2) [:,0]->w ;[:,1]->h
    
    def get_corres(self,smpl_inner_points, smpl_match_bound, rgb_bound, file_name=None):
        """
        计算变换后的内点位置

        :param smpl_inner_points:
        :param smpl_match_bound:
        :param rgb_bound:
        :param file_name:
        :return:
        """
        # mv_cord = get_mv_coordinate(smpl_inner_points, smpl_match_bound)
        # mv_cord = get_mv_coordinate_multiprocessing (smpl_inner_points, smpl_match_bound)
        if smpl_inner_points.shape[0]>100000:
            mv_cord_1 = self.get_mult_mv_coordinate (smpl_inner_points[:100000,:], smpl_match_bound)
            mv_cord_2 = self.get_mult_mv_coordinate (smpl_inner_points[100000:, :], smpl_match_bound)
            mv_cord=np.concatenate((mv_cord_1,mv_cord_2),axis=0)
        else:
            mv_cord =self.get_mult_mv_coordinate(smpl_inner_points,smpl_match_bound)
        pts_to = np.dot (mv_cord, rgb_bound)
        if file_name != None:
            np.save (file_name, pts_to.astype (int))
        return pts_to.astype (int)

    def warp_map(self,input_map, inner_points, warped_inner_points):
        '''

        :param input_map: (h,w,_)
        :param inner_points: (n,2)
        :param warped_inner_points: (n,2)  [:,0]->w ;[:,1]->h
        :return:
        '''
        # warp_map = np.zeros (input_map.shape)
        # h, w, _ = input_map.shape
        # for i in range (inner_points.shape[0]):
        #     x1, y1 = inner_points[i, :]
        #     x2, y2 = warped_inner_points[i, :]
        #     x2 = 0 if x2 >= w else x2
        #     y2 = 0 if y2 >= h else y2
        #     # for c in range(3):
        #     #     warp_img[y2, x2, c] = src_img[y1, x1, c]
        #     warp_map[y2, x2, :] = input_map[y1, x1, :]
        #     # print('%d/%d warp map' % (i, pts_num))

        warp_map = np.zeros (input_map.shape)
        h, w, _ = input_map.shape
        warped_inner_points[:,0]=np.where(warped_inner_points[:,0]<w,warped_inner_points[:,0],0)
        warped_inner_points[:, 1] = np.where (warped_inner_points[:, 1] < h, warped_inner_points[:, 1], 0)
        warped_inner_points[:, 0] = np.where (warped_inner_points[:, 0] >0, warped_inner_points[:, 0], 0)
        warped_inner_points[:, 1] = np.where (warped_inner_points[:, 1] >0, warped_inner_points[:, 1], 0)
        warp_map[warped_inner_points[:, 1], warped_inner_points[:,0], :] = input_map[ inner_points[:, 1],inner_points[:, 0], :]

        return warp_map

    def holefilling_(self,coarse_warp_img, inner_pts, input_rgb_mask):
        """

        :param coarse_warp_img: 初步变换的map图  #(h,w,_)
        :param inner_pts: mask内部点集  (n,2)  [:,0]->w ;[:,1]->h
        :param input_rgb_mask:mask边界点集 (m,2)  [:,0]->w ;[:,1]->h
        :return:
        """


        z_shape=coarse_warp_img.shape[2]
        pts_num = inner_pts.shape[0]
        edge_hole = []
        filled_warp_img = coarse_warp_img.copy ()

        for i in range (0, pts_num):
            # print('%d/%d hole filling' % (i, pts_num))
            edge_flag = False
            x, y = inner_pts[i, :]
            if self.isvalidpix (coarse_warp_img, input_rgb_mask, x, y) == 'inner point not filled':

                neighbor_8 = []

                # search for 8-neighbor, in anti-clockwise
                for a, b in zip ([-1, -1, -1, 0, 1, 1, 1, 0], [-1, 0, 1, 1, 1, 0, -1, -1]):
                    edge_flag = self.collectneighbor (coarse_warp_img, input_rgb_mask, x + a, y + b, neighbor_8, edge_flag)

                if edge_flag:
                    edge_hole.append ([x, y])
                    continue

                else:

                    neighbor_8 = np.array (neighbor_8).reshape (-1, 2)

                    if neighbor_8.shape[0] >= 4:
                        weights = MVC ([x, y], neighbor_8)
                        # bound_normal=[]
                        # for  k in range(0, neighbor_8.shape[0]):
                        #     bound_normal.append(filled_warp_img[neighbor_8[k, 1], neighbor_8[k, 0], :])
                        # filled_warp_img[y, x, :] = weights*bound_normal
                        num_weights = weights.shape[0]

                        filled_warp_img[y, x, :] = np.zeros([z_shape])

                        for k in range (0, num_weights):
                            filled_warp_img[y, x, :] += filled_warp_img[neighbor_8[k, 1], neighbor_8[k, 0], :] * \
                                                        weights[k]
                    else:
                        edge_hole.append ([x, y])

        edge_hole = np.array (edge_hole).reshape (-1, 2)
        edge_hole_num = edge_hole.shape[0]

        filled_pixel = []
        for i in range (inner_pts.shape[0]):
            x, y = inner_pts[i, :]
            if self.isvalidpix (filled_warp_img, input_rgb_mask, x, y) == 'inner point filled':
                filled_pixel.append ([x, y])

        filled_pixel = np.array (filled_pixel)
        nbrs = NearestNeighbors (n_neighbors=1, algorithm='auto').fit (filled_pixel)  # 最邻近算法

        for i in range (edge_hole_num):
            x, y = edge_hole[i, :]
            idx = nbrs.kneighbors (np.array ([x, y]).reshape (-1, 2), return_distance=False)
            x_n, y_n = filled_pixel[idx[0, 0], :]

            filled_warp_img[y, x, :] = filled_warp_img[y_n, x_n, :]


        for i in range (inner_pts.shape[0]):
            # 平滑操作
            x, y = inner_pts[i, :]
            sum = np.zeros([z_shape])
            count = 0
            max_x = input_rgb_mask.shape[1]
            max_y = input_rgb_mask.shape[0]
            for a in range (-1, 2):
                for b in range (-1, 2):
                    if x + a >= max_x or y + b >= max_y:
                        continue
                    elif self.isvalidpix (filled_warp_img, input_rgb_mask, x + a, y + b) == 'inner point filled':
                        sum += filled_warp_img[y + b, x + a, :]
                        count += 1
            if count == 0:
                continue

            average = sum / count

            filled_warp_img[y, x, :] = average

        ##平滑weigth
        filled_warp_img_temp=filled_warp_img[:,:,6:]
        for i in range (8):
            temp_weigth = filled_warp_img_temp[:, :, i * 3:(i + 1) * 3]
            temp_weigth= (cv2.medianBlur ((temp_weigth * 255).astype ('uint8'), 5))
            temp_weigth=cv2.blur (temp_weigth, (3, 3))
            filled_warp_img[:, :, 6 + i * 3:6 + (i + 1) * 3] =(cv2.medianBlur (temp_weigth, 3))/255.0
            # filled_warp_img = cv2.bitwise_and (filled_warp_img, filled_warp_img, mask=input_rgb_mask)
        filled_warp_with_mask=filled_warp_img *(input_rgb_mask[:,:,None].astype('bool'))
        # cv2.imwrite (output_path, (filled_warp_img * 255).astype ('uint8')[:, :, ::-1])
        return filled_warp_with_mask

    def holefilling(self,coarse_warp_img, inner_pts, input_rgb_mask):
        """

        :param coarse_warp_img: 初步变换的map图  #(h,w,_)
        :param inner_pts: mask内部点集  (n,2)  [:,0]->w ;[:,1]->h
        :param input_rgb_mask:mask边界点集 (m,2)  [:,0]->w ;[:,1]->h
        :return:
        """


        z_shape=coarse_warp_img.shape[2]
        pts_num = inner_pts.shape[0]
        edge_hole = []
        filled_warp_img = coarse_warp_img.copy ()

        for i in range (0, pts_num):
            # print('%d/%d hole filling' % (i, pts_num))
            edge_flag = False
            x, y = inner_pts[i, :]
            if self.isvalidpix (coarse_warp_img, input_rgb_mask, x, y) == 'inner point not filled':

                neighbor_8 = [] #(edg_n,2)  [:,0]->w ;[:,1]->h

                # search for 8-neighbor, in anti-clockwise
                for a, b in zip ([-1, -1, -1, 0, 1, 1, 1, 0], [-1, 0, 1, 1, 1, 0, -1, -1]):
                    edge_flag = self.collectneighbor (coarse_warp_img, input_rgb_mask, x + a, y + b, neighbor_8, edge_flag)

                if edge_flag:
                    edge_hole.append ([x, y])
                    continue

                else:

                    neighbor_8 = np.array (neighbor_8).reshape (-1, 2)

                    if neighbor_8.shape[0] >= 4:
                        weights = MVC ([x, y], neighbor_8)
                        # bound_normal=[]
                        # for  k in range(0, neighbor_8.shape[0]):
                        #     bound_normal.append(filled_warp_img[neighbor_8[k, 1], neighbor_8[k, 0], :])
                        # filled_warp_img[y, x, :] = weights*bound_normal
                        num_weights = weights.shape[0]

                        filled_warp_img[y, x, :] = np.zeros([z_shape])

                        for k in range (0, num_weights):
                            filled_warp_img[y, x, :] += filled_warp_img[neighbor_8[k, 1], neighbor_8[k, 0], :] * \
                                                        weights[k]
                    else:
                        edge_hole.append ([x, y])

        edge_hole = np.array (edge_hole).reshape (-1, 2)
        edge_hole_num = edge_hole.shape[0]

        filled_pixel = []
        for i in range (inner_pts.shape[0]):
            x, y = inner_pts[i, :]
            if self.isvalidpix (filled_warp_img, input_rgb_mask, x, y) == 'inner point filled':
                filled_pixel.append ([x, y])

        filled_pixel = np.array (filled_pixel)
        nbrs = NearestNeighbors (n_neighbors=1, algorithm='auto').fit (filled_pixel)  # 最邻近算法

        for i in range (edge_hole_num):
            x, y = edge_hole[i, :]
            idx = nbrs.kneighbors (np.array ([x, y]).reshape (-1, 2), return_distance=False)
            x_n, y_n = filled_pixel[idx[0, 0], :]

            filled_warp_img[y, x, :] = filled_warp_img[y_n, x_n, :]


        for i in range (inner_pts.shape[0]):
            # 平滑操作
            x, y = inner_pts[i, :]
            sum = np.zeros([z_shape])
            count = 0
            max_x = input_rgb_mask.shape[1]
            max_y = input_rgb_mask.shape[0]
            for a in range (-2, 2):
                for b in range (-2, 2):
                    if x + a >= max_x or y + b >= max_y:
                        continue
                    elif self.isvalidpix (filled_warp_img, input_rgb_mask, x + a, y + b) == 'inner point filled':
                        sum += filled_warp_img[y + b, x + a, :]
                        count += 1
            if count == 0:
                continue

            average = sum / count

            filled_warp_img[y, x, :] = average

        ##平滑weigth
        filled_warp_img_temp=filled_warp_img[:,:,6:]
        for i in range (8):
            temp_weigth = filled_warp_img_temp[:, :, i * 3:(i + 1) * 3]
            temp_weigth= (cv2.medianBlur ((temp_weigth * 255).astype ('uint8'), 5))
            temp_weigth=cv2.blur (temp_weigth, (3, 3))
            filled_warp_img[:, :, 6 + i * 3:6 + (i + 1) * 3] =(cv2.medianBlur (temp_weigth, 3))/255.0
            # filled_warp_img = cv2.bitwise_and (filled_warp_img, filled_warp_img, mask=input_rgb_mask)
        filled_warp_with_mask=filled_warp_img *(input_rgb_mask[:,:,None].astype('bool'))
        # cv2.imwrite (output_path, (filled_warp_img * 255).astype ('uint8')[:, :, ::-1])
        return filled_warp_with_mask

    def isvalidpix(self,image, mask, x, y):

        if sum([mask[y, x]]) == 0:

            return 'not inner point'
        elif sum(image[y, x]) == 0:
            return 'inner point not filled'

        else:
            return 'inner point filled'

    def collectneighbor(self,image, mask, x, y, neighbor, edge_flag):

        if self.isvalidpix (image, mask, x, y) == 'not inner point':
            edge_flag = True
        elif self.isvalidpix (image, mask, x, y) == 'inner point filled':
            neighbor.append ([x, y])

        return edge_flag

    def get_one_mv_coordinate(self,pt, counts):
        counts2vert = counts - pt  # (n,2)
        dist_counts2vert = linalg.norm (counts2vert, axis=1)  # (n,)

        if np.any (dist_counts2vert == 0):
            return np.where (dist_counts2vert == 0, 1, 0)
        counts2vert_det = np.cross (counts2vert, np.roll (counts2vert, -1, axis=0))  # (n,)
        counts2vert_dot = np.sum (np.multiply (counts2vert, np.roll (counts2vert, -1, axis=0)), axis=1)  # (n,)

        if np.any (np.logical_and (counts2vert_det == 0, counts2vert_dot < 0)):
            index = np.argwhere (np.logical_and (counts2vert_det == 0, counts2vert_dot < 0))
            cord = np.zeros (counts2vert_det.shape)
            cord[index] = dist_counts2vert[index] / (dist_counts2vert[index] + dist_counts2vert[index + 1])
            cord[index + 1] = dist_counts2vert[index + 1] / (dist_counts2vert[index] + dist_counts2vert[index + 1])
            return cord
        # tan_alpa = np.divide ((np.multiply (dist_counts2vert, np.roll (dist_counts2vert, -1)) - counts2vert_dot),
        #                       counts2vert_det)
        tan_alpa = np.divide (counts2vert_det,
                              (np.multiply (dist_counts2vert, np.roll (dist_counts2vert, -1)) + counts2vert_dot))
        w = np.divide ((np.roll (tan_alpa, 1) + tan_alpa), dist_counts2vert)
        cord = w / np.sum (w)
        return cord

    def get_mult_mv_coordinate(self,pts, counts):
        '''

        :param pts: (m,2)
        :param counts: (n,2)
        :return:
        '''
        cord = np.zeros ((pts.shape[0], counts.shape[0]))

        counts2vert = np.subtract (counts[None, :, :], pts[:, None, :])  # (m,n,2)

        dist_counts2vert = linalg.norm (counts2vert, axis=2)  # (m,n)

        counts2vert_det = np.cross (counts2vert, np.roll (counts2vert, -1, axis=1), axis=2)  # (m,n)
        counts2vert_dot = np.sum (np.multiply (counts2vert, np.roll (counts2vert, -1, axis=1)), axis=2)  # (m,n)

        # if np.any(dist_counts2vert==0,axis=0):
        indexs_1 = np.argwhere (np.any (dist_counts2vert == 0, axis=1))[:, 0]  # (m1,1)
        cord[indexs_1] = np.where (dist_counts2vert[indexs_1] == 0, 1, 0)

        # if np.any(np.logical_and(counts2vert_det==0,counts2vert_dot<0),axis=0):
        # indexs_2=np.argwhere(np.logical_and(np.any(counts2vert_det==0,axis=0),
        #                                     np.any(counts2vert_dot<0,axis=0),
        #                                     np.any(dist_counts2vert!=0,axis=0)))#(m2,1)
        indexs_2 = np.argwhere (
            np.logical_and (np.any (np.logical_and (counts2vert_det == 0, counts2vert_dot < 0), axis=1),
                            np.all (dist_counts2vert != 0, axis=1)))[:, 0]  # (m2,1)
        cord_temp = np.zeros ((indexs_2.shape[0], counts.shape[0]))  # (m2,n)
        cord_temp_index = np.argwhere (
            np.logical_and (counts2vert_det[indexs_2] == 0, counts2vert_dot[indexs_2] < 0))  # (m2,2)
        cord_temp_index_next=(cord_temp_index+[0,1])%[cord_temp_index.shape[0],counts.shape[0]]
        cord_temp[cord_temp_index[:, 0], cord_temp_index[:, 1]] = dist_counts2vert[indexs_2][cord_temp_index[:, 0], cord_temp_index[:, 1]] / (
                                                                          dist_counts2vert[indexs_2][cord_temp_index[:, 0], cord_temp_index[:,1]] +
                                                                          dist_counts2vert[indexs_2][cord_temp_index_next[:, 0], cord_temp_index_next[:,1]])
        cord_temp[cord_temp_index_next[:, 0], cord_temp_index_next[:, 1]] = dist_counts2vert[indexs_2][cord_temp_index_next[:, 0], cord_temp_index_next[:,1]] / (
                                                                              dist_counts2vert[indexs_2][cord_temp_index[:,0], cord_temp_index[:, 1]] +
                                                                              dist_counts2vert[indexs_2][cord_temp_index_next[:,0], cord_temp_index_next[:, 1]])
        cord[indexs_2] = cord_temp

        indexs_3 = np.argwhere (
            np.logical_and (np.all (np.logical_or (counts2vert_det != 0, counts2vert_dot >= 0), axis=1),
                            np.all (dist_counts2vert != 0, axis=1)))[:, 0]  # (m2,1)
        temp= np.multiply (dist_counts2vert[indexs_3], np.roll (dist_counts2vert[indexs_3], -1, axis=1)) +counts2vert_dot[indexs_3]
        tan_alpa = np.divide (counts2vert_det[indexs_3],temp)
        temp_tan_alpa=np.roll (tan_alpa, 1, axis=1) + tan_alpa
        w = np.divide (temp_tan_alpa, dist_counts2vert[indexs_3])
        cord[indexs_3] = w / np.sum (w, axis=1)[:, None]
        return cord

    def draw_points(self,pts):
        """
        :param pts:
        :return:
        """
        fig = plt.figure (figsize=(10.24, 10.24))
        axe = fig.add_subplot (111)
        X, Y = [], []
        for point in pts:
            X.append (point[0])
            Y.append (-point[1])
        axe.plot (X, Y, 'ro',markersize=1)
        fig.show ()

    def draw_matching(self,b1, b2):
        """

        :param b1:
        :param b2:
        :return:
        """
        L = b1.shape[0]

        fig = plt.figure (figsize=(10.24, 10.24))
        axe = fig.add_subplot (111)
        X, Y = [], []
        X1, Y1 = [], []
        # fig.show ()
        for i in range (L):
            p1 = b1[i]
            p2 = b2[i]
            X.append (p1[0])
            Y.append (-p1[1])
            axe.cla ()
            # axe.plot (X, Y, 'ro', markersize=1)
            X1.append (p2[0])
            Y1.append (-p2[1])
            # axe.plot (X1, Y1, 'bo', markersize=1)
            # fig.canvas.draw ()
        axe.plot (X1, Y1, 'bo', markersize=1)
        axe.plot (X, Y, 'ro', markersize=1)
        fig.show ()

    def show_img(self,value):
        cv2.imshow ('front.png',
                     (value[:, :, 0:3] * 255).astype ('uint8')[:, :, ::-1])
        cv2.waitKey ()
        cv2.imshow ( 'back.png',
                     (value[:, :, 3:6] * 255).astype ('uint8')[:, :, ::-1])
        cv2.waitKey()
        cv2.destroyAllWindows()

    def save2npy(self):
        np.save (os.path.join(self.out_path ,'warp_and_filled.npy'), self.fill_smplh_value)

    def save2img(self):
        cv2.imwrite (os.path.join(self.out_path ,'filled_front.png'),
                     (self.fill_smplh_value[:, :, 0:3] * 255).astype ('uint8')[:, :, ::-1])
        cv2.imwrite (os.path.join(self.out_path ,'filled_back.png'),
                     (self.fill_smplh_value[:, :, 3:6] * 255).astype ('uint8')[:, :, ::-1])
        cv2.imwrite (os.path.join(self.out_path ,'warp_front.png'),
                     (self.warp_smplh_value[:, :, 0:3] * 255).astype ('uint8')[:, :, ::-1])
        cv2.imwrite (os.path.join(self.out_path ,'warp_back.png'),
                     (self.warp_smplh_value[:, :, 3:6] * 255).astype ('uint8')[:, :, ::-1])
        self.save_weigth2img(os.path.join(self.out_path ,'filled_weigth.png'),self.fill_smplh_value[:, :, 6:])
        self.save_weigth2img (os.path.join (self.out_path, 'warp_weigth.png'), self.warp_smplh_value[:, :, 6:])

    def save_weigth2img(self,save_path,weights):
        img=np.zeros((weights.shape[0],weights.shape[1],3))
        colormap = [(0, 0, 0.5),  (0, 0.5,  0.5), (0, 0.75,  0.5), (0, 1,  0.5),
                    (0.5, 0,  0.5),  (0.5, 0.5,  0.5), (0.5, 0.75,  0.5), (0.5, 1,  0.5),
                    (1, 0,  0.5),  (1, 0.5,  0.5), (1, 0.75,  0.5), (1, 1,  0.5),

                    (0, 0, 0), (0, 0.5, 0), (0, 0.75, 0), (0,1,0),
                    (0.5, 0, 0), (0.5, 0.5, 0), (0.5, 0.75, 0), (0.5, 1, 0),
                    (1, 0, 0),  (1, 0.5, 0), (1, 0.75, 0), (1, 1, 0),

                    (0, 0, 1),  (0, 0.5, 1), (0, 0.75, 1), (0, 1, 1),
                    (0.5, 0, 1),  (0.5, 0.5, 1), (0.5, 0.75, 1), (0.5, 1, 1),
                    (1, 0, 1),  (1, 0.5, 1), (1, 0.75, 1), (1, 1, 1)
                    ]
        for i in range(24):
            temp_weigth=weights[:,:,i]
            img += (temp_weigth[:,:,None] *colormap[i]* 255).astype ('uint8')
        cv2.imwrite (save_path, img)
if __name__ == '__main__':
    pass
    # dr = 'data/reconstruct_output/'
    # rgb_img_path = 'data/images/baoluo.png'
    # rgb_mask_path = 'data/images/rgb_mask.png'
    #
    # smplh_model_path = 'models/model/smplh/SMPLH_MALE.pkl'
    # rgb_img = cv2.imread (rgb_img_path).astype (np.float32)[:, :, ::-1] / 255.0
    # rgb_mask = cv2.imread (rgb_mask_path, cv2.IMREAD_GRAYSCALE)
    # # cv2.imshow ('rend_normal_img', rgb_mask)
    # # cv2.waitKey()
    # H, W, _ = rgb_img.shape
    # # smplh_result,_=gen_smplh()
    # with open ('../data/output/smplh_1/baoluo_smplh.pkl', 'rb') as f:
    #     smplh_result = pickle.load (f, encoding='iso-8859-1')
    #
    # camera_rotation = smplh_result['camera_rotation'].astype ('float64')
    # camera_transl = smplh_result['camera_translation'].astype ('float64')
    # camera_center = smplh_result['camera_center'].astype ('float64')
    #
    # pose = smplh_result['spmlh_pose'].reshape ([-1, 3]).astype ('float64')
    # beta = smplh_result['spmlh_shape'].astype ('float64')
    #
    # smplh_model = SMPLHModel (smplh_model_path)
    #
    # smplh_model.set_params (beta=beta, pose=pose)
    #
    # # outmesh_path = './smpl.obj'
    # # smplh_model.output_mesh (outmesh_path)
    # # out_rendimg_path = './rend_smplh_img.png'
    # # rendermodel2img (camera_center, camera_transl, img, out_rendimg_path)
    #
    # # front_face, front_verts, back_face, back_verts =smplh_model.divide_face()
    #
    # ##render smplh
    #
    # render = render_model.Render (smplh_model, rgb_img, camera_center, camera_transl, camera_rotation,
    #                               out_path='data/render_result')
    # normals_img = render.normals_renderer ()
    # # re_back_normals_img = render.re_back_normals_renderer ()
    # front_normals_img = render.front_normals_renderer ()
    # back_normals_img = render.back_normals_renderer ()
    #
    # render_weigth = render.weigth_render ()