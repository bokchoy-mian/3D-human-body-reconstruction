#!/usr/bin/python
#-- coding:utf8 --
import cv2
import os
import numpy as np
from functools import reduce
import copy
import trimesh
import trimesh.smoothing as smoothing
from utils import B_Spline

class Depth2Mesh_Bspline():
    def __init__(self,front_depth,front_color,back_depth,back_color,weigths,J_2d,out_path=None):
        self.front_depth=front_depth
        self.front_color=front_color
        self.weigths=weigths
        self.back_depth=back_depth
        self.back_color=back_color
        self.J_2d=J_2d
        # self.front_depth_mean = np.mean (self.front_depth)
        # back_bound_depth_mean = np.mean (back_bound_depth)

        mask=front_depth>0
        mask=mask.astype (np.float32)
        self.mask=cv2.morphologyEx (mask, cv2.MORPH_CLOSE, np.ones ((int (3), int (3)), np.uint8))#闭运算
        self.out_path=out_path

    def __call__(self):
        # self.verts,self.faces=self.stich_mesh()
        # if self.out_path:
        #     self.writeobj(self.out_path,self.verts,self.faces)
        return self.stich_mesh()
    def depth2trimesh(self,depth,color,n,is_back=False):

        high, wigth = depth.shape
        idx = np.arange (0, (high * wigth)).reshape ((high, wigth))

        # init X, Y coordinate tensors 生成网格
        X, Y = np.meshgrid (np.arange (wigth), np.arange (high))
        X = np.expand_dims (X, axis=2)  # (H,W,1)
        Y = np.expand_dims (Y, axis=2)

        # convert the images to 3D mesh
        depth = np.expand_dims (depth, axis=2)

        axis = 6 + self.weigths.shape[2]
        fpc = np.concatenate ((X, Y, depth, color, self.weigths), axis=2)

        # get the edge region for the edge point interpolation

        # kernel = cv2.getStructuringElement (cv2.MORPH_RECT, (3, 3))
        # eroded = cv2.erode (self.mask, kernel)
        # edge = (self.mask - eroded).astype (np.bool)

        fpc = fpc.reshape (-1, axis)

        # f_faces = getfrontFaces (self.mask, fp_idx)
        ##getfrontFace
        valid_idx = idx * self.mask  # 背景序号归0
        p00_idx = valid_idx[:-1, :-1].reshape (-1, 1)
        p10_idx = valid_idx[1:, :-1].reshape (-1, 1)
        p11_idx = valid_idx[1:, 1:].reshape (-1, 1)
        p01_idx = valid_idx[:-1, 1:].reshape (-1, 1)
        if is_back:
            all_faces = np.vstack (
                (np.hstack ((p00_idx, p01_idx, p10_idx)), np.hstack ((p01_idx, p11_idx, p10_idx))))
        else:
            all_faces = np.vstack (
                (np.hstack ((p00_idx, p10_idx, p01_idx)), np.hstack ((p01_idx, p10_idx, p11_idx))))
        # edge_valid_idx = idx * edge  # 背景序号归0
        # edge_p00_idx = edge_valid_idx[:-1, :-1].reshape (-1, 1)
        # edge_p10_idx = edge_valid_idx[1:, :-1].reshape (-1, 1)
        # edge_p11_idx = edge_valid_idx[1:, 1:].reshape (-1, 1)
        # edge_p01_idx = edge_valid_idx[:-1, 1:].reshape (-1, 1)
        # all_faces = np.vstack (
        #     (np.hstack ((p00_idx, p10_idx, p01_idx)), np.hstack ((p01_idx, p10_idx, p11_idx)),
        # np.hstack ((edge_p00_idx, edge_p10_idx, edge_p11_idx)),
        # np.hstack ((edge_p00_idx, edge_p11_idx, edge_p01_idx))))

        faces = all_faces[np.where (all_faces[:, 0] * all_faces[:, 1] * all_faces[:, 2] > 0)]

        points = fpc

        # 去除模型外点
        # a = np.array ([1, 2, 6, 4, 2, 3, 2])
        # u, indices = np.unique (a, return_inverse=True)
        # u
        # array ([1, 2, 3, 4, 6])
        # indices
        # array ([0, 1, 4, 3, 1, 2, 1])
        # u[indices]
        # array ([1, 2, 6, 4, 2, 3, 2])
        verts_index, inverse_index = np.unique (np.ravel (faces).astype ('int32'), return_inverse=True)


        faces = inverse_index.reshape (-1, 3)

        re_points = points[verts_index]

        vertices = re_points[:, 0:3]
        colors = re_points[:, 3:6].astype (np.uint8)
        # weigths = re_points[:, 6:]
        mesh = trimesh.Trimesh (vertices=vertices, faces=faces, vertex_colors=colors,process=False)#process=False不对数据预处理
        if is_back:
            mesh.apply_transform(self.back_trimesh_trans_angel())
            re_points[:, 0:3]=mesh.vertices
        re_faces=(inverse_index+n).reshape (-1, 3)# inverse_index+=n#添加front的点序号
        return mesh,re_faces,re_points

    def verts2faces (self ,verts_points,points_verts_index,front_bound_verts_color_weigth,back_bound_verts_color_weigth,n,axis=30):
        '''

        :param verts_points: 缝合部位顶点（m,n,3)
        :param points_verts_index: 边界顶点在原顶点集中的index应用于生成面的下标
        :param front_bound_verts_color_weigth ；back_bound_verts_color_weigth: 权重(n,27)
        :param n: frond面顶点数，所有back都要加上该值
        :param axis: 权重维度
        :return:
        '''
        remove_n=1#去除两行在原frond back 中存在的顶点（可以考虑去除四行）
        verts=verts_points[remove_n:-remove_n, :, :]#
        heigh, wigth = verts.shape[0:2]
        idx = np.arange (0, (heigh * wigth)).reshape ((heigh, wigth))#
        idx = np.concatenate ((idx, np.expand_dims (idx[:, 0], axis=1)), axis=1)# (h，w+1)

        p00_idx = idx[:-1, :-1].reshape (-1, 1)
        p10_idx = idx[1:, :-1].reshape (-1, 1)
        p11_idx = idx[1:, 1:].reshape (-1, 1)
        p01_idx = idx[:-1, 1:].reshape (-1, 1)
        trimesh_faces = np.vstack (
            (np.hstack ((p00_idx, p01_idx, p10_idx)), np.hstack ((p01_idx, p11_idx, p10_idx))))

        mesh = trimesh.Trimesh (vertices=verts.reshape (-1,3), faces=trimesh_faces, process=False)
        # mesh.rezero()
        # mesh.invert()
        # mesh.fix_normals()
        # mesh.show ()
        smoothing_mesh = smoothing.filter_humphrey (mesh)

        # smoothing_mesh = smoothing.filter_taubin (mesh)
        # smoothing_mesh=smoothing.filter_laplacian (mesh, iterations=5)#顶点产生偏移
        # smoothing_mesh.show ()

        smoothing_verts = smoothing_mesh.vertices.reshape (heigh, wigth, 3)

        alph=np.expand_dims(np.expand_dims(np.arange(0,1,1/heigh),axis=1),axis=2)#(heigh,1,1)

        front_bound_verts_color_weigth=np.tile (front_bound_verts_color_weigth, [heigh, 1, 1])
        back_bound_verts_color_weigth = np.tile (back_bound_verts_color_weigth, [heigh, 1, 1])
        points_color_weigth=np.add(np.multiply(front_bound_verts_color_weigth,1-alph),np.multiply(back_bound_verts_color_weigth,alph))
        points = np.concatenate ((smoothing_verts,points_color_weigth), axis=2).reshape (-1,axis)  # (n*m,axis)

        #idx (h+2,w+1)
        points_verts_index=np.append(points_verts_index,points_verts_index[0])#w+1

        stich_idx=np.concatenate ((np.expand_dims (points_verts_index, axis=0),idx+2*n,
                             np.expand_dims (points_verts_index+n, axis=0) ), axis=0)

        stich_p00_idx = stich_idx[:-1, :-1].reshape (-1, 1)
        stich_p10_idx = stich_idx[1:, :-1].reshape (-1, 1)
        stich_p11_idx = stich_idx[1:, 1:].reshape (-1, 1)
        stich_p01_idx = stich_idx[:-1, 1:].reshape (-1, 1)
        stich_faces = np.vstack ((np.hstack ((stich_p00_idx, stich_p01_idx, stich_p10_idx)),
                                  np.hstack ((stich_p01_idx, stich_p11_idx, stich_p10_idx))))

        # stich_faces = np.vstack ((np.hstack ((stich_p00_idx, stich_p10_idx, stich_p01_idx)),
        #                           np.hstack ((stich_p01_idx, stich_p10_idx, stich_p11_idx))))
        return stich_faces,points

    def smoothing (self ,verts):

        heigh, wigth = verts.shape[0:2]

        idx = np.arange (0, (heigh * wigth)).reshape ((heigh, wigth))#
        idx = np.concatenate ((idx, np.expand_dims (idx[:, 0], axis=1)), axis=1)  # (h+2，w+1)首尾相连

        p00_idx = idx[:-1, :-1].reshape (-1, 1)
        p10_idx = idx[1:, :-1].reshape (-1, 1)
        p11_idx = idx[1:, 1:].reshape (-1, 1)
        p01_idx = idx[:-1, 1:].reshape (-1, 1)
        faces = np.vstack (
            (np.hstack ((p00_idx, p10_idx, p01_idx)), np.hstack ((p01_idx, p10_idx, p11_idx))))

        verts=verts.reshape (-1, 3)
        mesh = trimesh.Trimesh (vertices=verts, faces=faces,process=False)
        mesh.show()
        smoothing_mesh=smoothing.filter_humphrey(mesh)
        smoothing_mesh.show()

        smoothing_verts=smoothing_mesh.vertices.reshape(heigh, wigth, 3)

        return smoothing_verts

        # return faces,points

    def get_bound_verts_index(self,mesh):
        edges_unique,edges_index,edges_count=np.unique(mesh.edges_unique_inverse,axis=0,return_counts=True,return_index=True)
        bound_index=np.select([edges_count==1],[edges_index],-1)
        bound_index =np.delete(bound_index,np.where(bound_index==-1))
        # bound_unique = np.select (edges_count == 1, edges_unique)
        bound_unique =mesh.edges_unique[mesh.edges_unique_inverse[bound_index]]
        bound_verts_index=np.zeros(bound_unique.shape[0]+1).astype(np.int)
        bound_verts_index[0] =bound_unique[0,0]
        bound_verts_index[1]=bound_unique[0,1]
        bound_unique_temp=np.delete (bound_unique, 0, axis=0)

        for i in range(1,bound_verts_index.shape[0]-1):#得到有序顶点索引
            index=np.where(bound_unique_temp[:,0]==bound_verts_index[i])
            if len(index[0])>1:
                print(index)
            if len(index[0])==1:
                bound_verts_index[i+1]=bound_unique_temp[index,1]

            else:
                index=np.where(bound_unique_temp[:,1]==bound_verts_index[i])
                if len (index[0]) ==0:
                    print (index)
                    print(i)
                bound_verts_index[i + 1] = bound_unique_temp[index, 0]
            bound_unique_temp = np.delete (bound_unique_temp, index, axis=0)

        in_bound_verts_index_list=[]#内层的顶点
        for i in range(bound_verts_index.shape[0]-1):
            neighbors_1=mesh.vertex_neighbors[bound_verts_index[i]] #[,]
            neighbors_2 = mesh.vertex_neighbors[bound_verts_index[i+1]]
            ret = list (set (neighbors_1).intersection (set (neighbors_2)))
            in_bound_verts_index_list.append(ret[0])

        # in_bound_verts_index_list_set=list(set(in_bound_verts_index_list))#去重，不改变位置
        # in_bound_verts_index_list_set.sort(key=in_bound_verts_index_list.index)
        # in_bound_verts_index_list_set.append(in_bound_verts_index_list_set[0])#首尾相连
        in_bound_verts_index=np.asarray(in_bound_verts_index_list)

        return bound_verts_index[:-1],in_bound_verts_index

    def gen_Bspline_curve(self,points,degree):
        curve=B_Spline.B_spline_curve(points,degree)
        return curve

    def gen_Bspline_curve_multi(self,points,degree):
        curves=B_Spline.B_spline_curve_multi(points,degree)
        return curves

    def gen_mesh_ply(self,is_save=False):

        high, wigth = self.front_depth.shape
        # fix_p = (np.mean (self.front_depth) + np.mean (self.back_depth)) / 2
        fp_idx = np.arange (0, (high * wigth)).reshape ((high, wigth))
        bp_idx = np.arange ((high * wigth), (high * wigth) * 2).reshape ((high, wigth))

        # init X, Y coordinate tensors 生成网格
        X, Y = np.meshgrid (np.arange (wigth), np.arange (high))
        X = np.expand_dims (X, axis=2)  # (H,W,1)
        Y = np.expand_dims (Y, axis=2)
        dx = 1.0
        dy = 1.0
        x_cord = X * dx
        y_cord = Y * dy

        # convert the images to 3D mesh
        front_depth = np.expand_dims (self.front_depth, axis=2)
        back_depth = np.expand_dims (self.back_depth, axis=2)
        axis=6+self.weigths.shape[2]
        fpc = np.concatenate ((x_cord, y_cord, front_depth, self.front_color,self.weigths), axis=2)
        bpc = np.concatenate ((x_cord, y_cord, back_depth, self.back_color,self.weigths), axis=2)

        # get the edge region for the edge point interpolation
        
        kernel = cv2.getStructuringElement (cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode (self.mask, kernel)
        self.edge = (self.mask - eroded).astype (np.bool)
        # interpolate 2 points for each edge point pairs
        # fpc[edge, 2:] = (fpc[edge, 2:] * 2 + bpc[edge, 2:] * 1) / 3  # 前后各加一点
        # bpc[edge, 2:] = (fpc[edge, 2:] * 1 + bpc[edge, 2:] * 2) / 3
        # re_mask = self.mask
        # eroded = cv2.erode (re_mask, kernel)
        # edge_1 = (re_mask - eroded).astype (np.bool)#(h,w)
        # re_mask = eroded
        #
        # eroded = cv2.erode (re_mask, kernel)
        # edge_2 = (re_mask - eroded).astype (np.bool)

        fpc = fpc.reshape (-1, axis)
        bpc = bpc.reshape (-1, axis)

        # f_faces = getfrontFaces (self.mask, fp_idx)
        ##getfrontFace
        fp_valid_idx = fp_idx * self.mask #背景序号归0
        fp00_idx = fp_valid_idx[:-1, :-1].reshape (-1, 1)
        fp10_idx = fp_valid_idx[1:, :-1].reshape (-1, 1)
        fp11_idx = fp_valid_idx[1:, 1:].reshape (-1, 1)
        fp01_idx = fp_valid_idx[:-1, 1:].reshape (-1, 1)
        f_all_faces = np.vstack ((np.hstack ((fp00_idx, fp10_idx, fp01_idx)), np.hstack ((fp01_idx, fp10_idx, fp11_idx)),
                                np.hstack ((fp00_idx, fp10_idx, fp11_idx)), np.hstack ((fp00_idx, fp11_idx, fp01_idx))))

        fp_edge_valid_idx = fp_idx * self.edge  # 背景序号归0
        edge_fp00_idx = fp_edge_valid_idx[:-1, :-1].reshape (-1, 1)
        edge_fp10_idx = fp_edge_valid_idx[1:, :-1].reshape (-1, 1)
        edge_fp11_idx = fp_edge_valid_idx[1:, 1:].reshape (-1, 1)
        edge_fp01_idx = fp_edge_valid_idx[:-1, 1:].reshape (-1, 1)
        f_all_faces = np.vstack (
            (np.hstack ((fp00_idx, fp10_idx, fp01_idx)), np.hstack ((fp01_idx, fp10_idx, fp11_idx)),
             np.hstack ((edge_fp00_idx, edge_fp10_idx, edge_fp11_idx)), np.hstack ((edge_fp00_idx, edge_fp11_idx, edge_fp01_idx))))
        f_faces = f_all_faces[np.where (f_all_faces[:, 0] * f_all_faces[:, 1] * f_all_faces[:, 2] > 0)]
        
        ##get_back_face
        bp_valid_idx = bp_idx * self.mask
        bp00_idx = bp_valid_idx[:-1, :-1].reshape (-1, 1)
        bp10_idx = bp_valid_idx[1:, :-1].reshape (-1, 1)
        bp11_idx = bp_valid_idx[1:, 1:].reshape (-1, 1)
        bp01_idx = bp_valid_idx[:-1, 1:].reshape (-1, 1)
        b_all_faces = np.vstack ((np.hstack ((bp00_idx, bp01_idx, bp10_idx)), np.hstack ((bp01_idx, bp11_idx, bp10_idx)),
                                np.hstack ((bp00_idx, bp11_idx, bp10_idx)), np.hstack ((bp00_idx, bp01_idx, bp11_idx))))

        bp_edge_valid_idx = bp_idx * self.edge
        edge_bp00_idx = bp_edge_valid_idx[:-1, :-1].reshape (-1, 1)
        edge_bp10_idx = bp_edge_valid_idx[1:, :-1].reshape (-1, 1)
        edge_bp11_idx = bp_edge_valid_idx[1:, 1:].reshape (-1, 1)
        edge_bp01_idx = bp_edge_valid_idx[:-1, 1:].reshape (-1, 1)
        b_all_faces = np.vstack (
            (np.hstack ((bp00_idx, bp01_idx, bp10_idx)), np.hstack ((bp01_idx, bp11_idx, bp10_idx)),
             np.hstack ((edge_bp00_idx, edge_bp11_idx, edge_bp10_idx)), np.hstack ((edge_bp00_idx, edge_bp01_idx, edge_bp11_idx))))

        b_faces = b_all_faces[np.where (b_all_faces[:, 0] * b_all_faces[:, 1] * b_all_faces[:, 2] > 0)]

        contours, _ = cv2.findContours (self.mask.astype (np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        edges = contours[0][:, 0, :]
        nextedges = np.vstack ((edges[1:], edges[0]))
        fp_edge_idx = fp_idx[edges[:, 1], edges[:, 0]].reshape (-1, 1)
        bp_edge_idx = bp_idx[edges[:, 1], edges[:, 0]].reshape (-1, 1)

        faces = np.vstack ((f_faces, b_faces))
        points = np.concatenate ((fpc, bpc), axis=0)

        #去除模型外点
        # a = np.array ([1, 2, 6, 4, 2, 3, 2])
        # u, indices = np.unique (a, return_inverse=True)
        # u
        # array ([1, 2, 3, 4, 6])
        # indices
        # array ([0, 1, 4, 3, 1, 2, 1])
        # u[indices]
        # array ([1, 2, 6, 4, 2, 3, 2])
        verts_index, inverse_index = np.unique (np.ravel (faces).astype ('int32'), return_inverse=True)
        self.faces = inverse_index.reshape (-1, 3)

        self.re_points = points[verts_index]
        self.fp_edge_points=points[fp_edge_idx]
        self.bp_edge_points = points[bp_edge_idx]

        if is_save:
            self.write_ply()
        return self.re_points,self.faces

    def back_trimesh_trans_angel(self):
        # v1=np.asarray([self.J_2d[22, 1],self.J_2d[22, 0],self.front_depth[self.J_2d[22, 1], self.J_2d[22, 0]]])-np.asarray([self.J_2d[23, 1],self.J_2d[23, 0],self.front_depth[self.J_2d[23, 1], self.J_2d[23, 0]]])
        # v2=np.asarray([self.J_2d[22, 1],self.J_2d[22, 0],self.back_depth[self.J_2d[22, 1], self.J_2d[22, 0]]])-np.asarray([self.J_2d[23, 1],self.J_2d[23, 0],self.back_depth[self.J_2d[23, 1], self.J_2d[23, 0]]])
        v1 = np.asarray (
            [self.J_2d[16, 1], self.J_2d[16, 0], self.front_depth[self.J_2d[16, 1], self.J_2d[16, 0]]]) - np.asarray (
            [self.J_2d[17, 1], self.J_2d[17, 0], self.front_depth[self.J_2d[17, 1], self.J_2d[17, 0]]])
        v2 = np.asarray (
            [self.J_2d[16, 1], self.J_2d[16, 0], self.back_depth[self.J_2d[16, 1], self.J_2d[16, 0]]]) - np.asarray (
            [self.J_2d[17, 1], self.J_2d[17, 0], self.back_depth[self.J_2d[17, 1], self.J_2d[17, 0]]])

        r = np.arccos (np.dot (v1, v2) / (np.linalg.norm (v1, 2) * np.linalg.norm (v2, 2)))
        deg = r * 180 / np.pi
        trans=trimesh.transformations.rotation_matrix (np.radians (1*deg), [0, 1, 0])
        return trans

    def stich_mesh(self):
        '''
        front_points+back_points+stich_points
        :return:
        '''
        front_mesh,front_faces,front_points=self.depth2trimesh(self.front_depth,self.front_color,0,is_back=False)
        back_mesh,back_faces,back_points=self.depth2trimesh(self.back_depth,self.back_color,len(front_points),is_back=True)
        # front_mesh.apply_scale (0.1).show ()
        # trimes_verts_index_out,trimes_verts_index_in=self.get_bound_verts_index(front_mesh)
        # back_bound_verts_index_out,back_bound_verts_index_in=self.gen_bound_verts(back_mesh)
        points_verts_index_out,points_verts_index_in = self.get_bound_verts_index (front_mesh)
        # re_bound_verts_index_out,re_bound_verts_index_in
        # points_verts_index_out_,points_verts_index_in_ = self.trimes_index2points_index (trimes_verts_index_out,trimes_verts_index_in, front_faces, front_mesh)
        # points_verts_index_in = self.trimes_index2points_index (trimes_verts_index_in, front_faces, front_mesh)

        # #(n,3+3+24)
        front_bound_verts_points_out = front_points[points_verts_index_out]
        front_bound_verts_points_in = front_points[points_verts_index_in]
        back_bound_verts_points_out = back_points[points_verts_index_out]
        back_bound_verts_points_in = back_points[points_verts_index_in]
        #(n,3)

        # bound_diff =np.max(front_bound_verts_points_out[:,2]-back_bound_verts_points_out[:,2])
        front_bound_depth_mean = np.mean (front_bound_verts_points_out[:,2])
        back_bound_depth_mean = np.mean (back_bound_verts_points_out[:, 2])
        bound_diff=front_bound_depth_mean-back_bound_depth_mean

        front_depth_mean=np.mean (front_points[:,2])
        mesh_diff=front_bound_depth_mean-front_depth_mean
        diff=bound_diff+mesh_diff*1.5

        back_bound_verts_points_out[:,2]=back_bound_verts_points_out[:,2]+diff
        back_bound_verts_points_in[:, 2] = back_bound_verts_points_in[:, 2] + diff
        back_points[:,2]=back_points[:,2]+diff
        bound_number=len(points_verts_index_out)

        J_3d_z = (self.front_depth[self.J_2d[:, 1], self.J_2d[:, 0]] + self.back_depth[self.J_2d[:, 1], self.J_2d[:, 0]]+diff) / 2.0
        J_3d = np.concatenate ((self.J_2d[:, 0][:, None], self.J_2d[:, 1][:, None], J_3d_z[:, None]), axis=1)

        # points_list_for_bound_curv = np.stack ((front_bound_verts_points_in, front_bound_verts_points_out,
        #                          back_bound_verts_points_out, back_bound_verts_points_in), axis=0)[:, :, 0:3]
        #
        # bound_curve_container = self.gen_Bspline_curve_multi (points_list_for_bound_curv[:,::5,:], 3)
        # points_list = bound_curve_container ((1/bound_number))
        #(n,4,3)

        front_bound_verts_points_out_roll = np.concatenate ((front_bound_verts_points_out[1:, :],
                                                   front_bound_verts_points_out[-1:, :]), axis=0)
        front_bound_verts_points_out_inter = (np.add (front_bound_verts_points_out, front_bound_verts_points_out_roll) / 2)

        back_bound_verts_points_out_roll = np.concatenate ((back_bound_verts_points_out[1:, :],
                                                             back_bound_verts_points_out[-1:, :]), axis=0)
        back_bound_verts_points_out_inter = (
                    np.add (back_bound_verts_points_out, back_bound_verts_points_out_roll) / 2)

        points_list=np.stack((front_bound_verts_points_in,front_bound_verts_points_out_inter,
                              back_bound_verts_points_out_inter,back_bound_verts_points_in),axis=1)[:,:,0:3]

        # test_curv=self.gen_Bspline_curve(points_list[0].tolist(),3)
        # test_curv.show_VTK()
        # test_curv.show_plot()
        # test_curv_ = self.gen_Bspline_curve (points_list[0].tolist (), 2)
        # test_curv_.show_VTK ()
        # test_curv_.show_plot()
        Curve_Container=self.gen_Bspline_curve_multi(points_list[::2,:,:],2)#n/2

        # Curve_Container.show_VTK()
        # Curve_Container.show_plot()

        stich_verts=Curve_Container(0.1)#(m,n,3)

        stich_verts_repeat = np.repeat (stich_verts, 2, axis=1)
        stich_verts_repeat_roll = np.concatenate ((stich_verts_repeat[:,1:,  :],
                                                         stich_verts_repeat[:,-1:,  :]), axis=1)
        points_list_smooth = (np.add (stich_verts_repeat, stich_verts_repeat_roll) / 2)[:,:bound_number,:]

        # smoothing_verts=self.smoothing(verts_points)

        # verts_points=np.concatenate((verts_points,np.expand_dims(verts_points[0],axis=0)),axis=0)
        front_bound_verts_color_weigth=front_points[points_verts_index_in][:,3:]#权重,颜色(n,27)
        back_bound_verts_color_weigth = back_points[points_verts_index_in][:, 3:]
        front_points_n = len (front_points)  # front points number
        stich_faces,stich_points=self.verts2faces(points_list_smooth,points_verts_index_out,front_bound_verts_color_weigth,back_bound_verts_color_weigth,front_points_n)

        full_faces=np.concatenate((front_faces,back_faces,stich_faces),axis=0)
        full_points=np.concatenate((front_points,back_points,stich_points),axis=0)

        # mesh=trimesh.Trimesh(full_points[:,:3],full_faces,vertex_colors=full_points[:,3:6]).apply_scale(0.1).show()
        # mesh.fix_normals()
        # mesh.apply_scale(0.1).show()
        if self.out_path:
            self.save_mesh(full_points,full_faces,self.out_path)
        recover_J_3d=self.recover_3d_J(full_points,full_faces,J_3d)
        return full_points,full_faces,recover_J_3d

    def recover_3d_J(self,full_points,full_faces,J_3d):
        mesh = trimesh.Trimesh (full_points[:, :3], full_faces, vertex_colors=full_points[:, 3:6])

        # recover_J_3d=J_3d[:]
        recover_J_3d=copy.deepcopy(J_3d)
        # mesh.show ()
        #[18,19,20,21,22,23]
        #18-16;19-17;20-18;21-19;22-20;23-21
        # test_mesh=mesh.slice_plane (J_3d[20], self.norm (J_3d[2] - J_3d[1])).show()
        diff = (np.sum (np.square (J_3d[20] - J_3d[18])) ** (1 / 2)) / 2
        mesh = mesh.slice_plane (J_3d[23] -self.norm (J_3d[3] - J_3d[0]) * diff , self.norm (J_3d[3] - J_3d[0]))
        l_recover_vector= self.norm (J_3d[1] - J_3d[2])
        l_slice_mesh = mesh.slice_plane (J_3d[16]+l_recover_vector*diff*0.3, l_recover_vector)
        r_recover_vector = self.norm (J_3d[2] - J_3d[1])
        r_slice_mesh = mesh.slice_plane (J_3d[17]+r_recover_vector*diff*0.3, r_recover_vector)

        for index in [21,23]:
            recover_vector = self.norm (J_3d[index] - J_3d[index-2])
            recover_vector=self.norm (J_3d[19] - J_3d[17])
            recover_curv = r_slice_mesh.section (recover_vector, J_3d[index])
            recover_J_3d[index]=recover_curv.centroid
        for index in [18,20,22]:
            recover_vector = self.norm (J_3d[index] - J_3d[index-2])
            recover_vector = self.norm (J_3d[18] - J_3d[16])
            recover_curv = l_slice_mesh.section (recover_vector, J_3d[index])
            recover_J_3d[index]=recover_curv.centroid
        return recover_J_3d

    def norm(self,vec):
        n = np.linalg.norm (vec)
        if n == 0:
            return None
        return vec / n

    def trimes_index2points_index(self, trimes_index_out,trimes_index_in, faces, mesh):
        vertex_faces_index_out = mesh.vertex_faces[trimes_index_out]
        vertex_faces_index_out = [np.delete (i, np.where (i == -1)) for i in vertex_faces_index_out]  # 去除-1
        faces_vertex_out = [faces[i] for i in vertex_faces_index_out]
        points_index_out = [reduce (np.intersect1d, (i)) for i in faces_vertex_out]

        vertex_faces_index_in = mesh.vertex_faces[trimes_index_in]
        vertex_faces_index_in = [np.delete (i, np.where (i == -1)) for i in vertex_faces_index_in]  # 去除-1
        faces_vertex_in = [faces[i] for i in vertex_faces_index_in]
        points_index_in = [reduce (np.intersect1d, (i)) for i in faces_vertex_in]

        points_index_out_2points=np.where(np.asarray([len(index) for index in points_index_out])>1)[0]
        points_index_in_2points = np.where (np.asarray ([len (index) for index in points_index_in])>1)[0]

        #去除非边界点
        for i in points_index_out_2points:
            points_index_out[i] = np.setdiff1d (points_index_out[i], points_index_in[i])

        for i in points_index_in_2points:
            points_index_in[i] = np.setdiff1d (points_index_in[i], points_index_out[i])

        # # points_index_out=[np.setdiff1d(points_index_out[i], points_index_in[i]) for i  in points_index_out_2points ]
        # points_index_in = [np.setdiff1d (points_index_in[i], points_index_out[i]) for i in points_index_in_2points]

        return np.asarray([ index[0] for index in points_index_out]),np.asarray([ index[0] for index in points_index_in])

    def save_mesh(self,points,faces,out_path):
        output_file = os.path.join (out_path,  'out.ply')
        wigth = np.mean (points[:,0])
        high= np.mean (points[:,1])
        fix_p = np.mean (points[:, 2])
        vertices = points[:, 0:3].copy()
        vertices[:, 0:3] = -(vertices[:, 0:3] - np.array ([[wigth / 2 , high / 2, fix_p]])) / (
                        (wigth + high) / 4.0)
            # points[:, 0:2] = -(points[:, 0:2] - np.array ([[wigth / 2 , high/ 2 ]]))
        vertices[:, 0] = -vertices[:, 0]
        colors = points[:, 3:6].astype (np.uint8)
        # weigths = points[:, 6:]
        mesh = trimesh.Trimesh (vertices=vertices, faces=faces, vertex_colors=colors)
        mesh.export(output_file)

    def writeobj(self,filepath, vertices, triangles):
        with open (filepath, "w") as f:
            for i in range (vertices.shape[0]):
                f.write ("v {} {} {}\n".format (vertices[i, 0], vertices[i, 1], vertices[i, 2]))
            for i in range (triangles.shape[0]):
                f.write ("f {} {} {}\n".format (triangles[i, 0] + 1, triangles[i, 1] + 1, triangles[i, 2] + 1))

    def write_ply(self,points,faces,out_path):
        output_file = os.path.join (out_path,  'out.ply')
        wigth = np.mean (points[:,0])
        high= np.mean (points[:,1])
        fix_p = np.mean (points[:, 2])
        vertices = points[:, 0:3].copy()
        vertices[:, 0:3] = -(vertices[:, 0:3] - np.array ([[wigth / 2 , high / 2, fix_p]])) / (
                        (wigth + high) / 4.0)
            # points[:, 0:2] = -(points[:, 0:2] - np.array ([[wigth / 2 , high/ 2 ]]))
        vertices[:, 0] = -vertices[:, 0]
        colors = points[:, 3:6].astype (np.uint8)
        # weigths = points[:, 6:]
        mesh = trimesh.Trimesh (vertices=vertices, faces=faces, vertex_colors=colors)
        points, colors, faces=mesh.vertices, mesh.visual.vertex_colors, mesh.faces
        num_points = len (points)
        num_triangles = len (faces)

        header = '''ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    element face {}
    property list uchar int vertex_indices
    end_header\n'''.format (num_points, num_triangles)

        with open (output_file, 'w') as f:
            f.writelines (header)
            index = 0
            for item in points:
                f.write ("{0:0.6f} {1:0.6f} {2:0.6f} {3} {4} {5}\n".format (item[0], item[1], item[2],
                                                                            colors[index, 0], colors[index, 1],
                                                                            colors[index, 2]))
                index = index + 1

            for item in faces:
                number = len (item)
                row = "{0}".format (number)
                for elem in item:
                    row += " {0} ".format (elem)
                row += "\n"
                f.write (row)

    def save2npy(self,path,map):
        np.save(path,map)

if __name__ == '__main__':
    front_depth = np.load ('../data/reconstruct_output/depth_front.npy')
    front_color = cv2.imread ('../data/images/baoluo.png')
    front_color = cv2.cvtColor (front_color, cv2.COLOR_BGR2RGB)
    back_depth = np.load ('../data/reconstruct_output/depth_back.npy')
    back_color = cv2.imread ('../data/images/back_rgb.png')
    back_color = cv2.cvtColor (back_color, cv2.COLOR_BGR2RGB)
    warp_smplh_value = np.load ('../data/baoluo/' + 'warp_and_filled.npy')
    rgb_mask = np.where (front_depth + back_depth == 0, 0, 255).astype ('uint8')
    D=Depth2Mesh_Bspline (front_depth, front_color, back_depth, back_color, warp_smplh_value[:,:,6:], '../data/baoluo')
    # D()
    D.stich_mesh()