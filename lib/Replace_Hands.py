#!/usr/bin/python
#-- coding:utf8 --
# !/usr/bin/python
# -- coding:utf8 --
import cv2
import os
# import math
import numpy as np
from functools import reduce
# import transforms3d
import trimesh
from trimesh import grouping,geometry,intersections,smoothing
# import trimesh.smoothing as smoothing
from utils import B_Spline

class Replace_Hands ():

    def __init__(self, recover_points,recover_faces, recover_J, smpl_points,smpl_faces,smpl_J):
        self.recover_points=recover_points
        # self.recover_verts =  recover_points[:,:3]
        # self.recover_color = recover_points[:, 3:6]
        # self.recover_weigths = recover_points[:, 6:]
        self.recover_faces = recover_faces
        self.recover_J = recover_J

        self.smpl_points=smpl_points
        # self.smpl_verts = smpl_points[:,:3]
        # self.smpl_color = smpl_points[:, 3:6]
        # self.smpl_weigths = smpl_points[:, 6:]
        self.smpl_faces = smpl_faces
        self.smpl_J = smpl_J

        self.recover_trimesh=trimesh.Trimesh (vertices=self.recover_points[:,:3], faces=self.recover_faces,
                                              vertex_colors=self.recover_points[:, 3:6],process=False)  # process=False不对数据预处理
        self.smpl_trimesh = trimesh.Trimesh (vertices=self.smpl_points[:,:3], faces=self.smpl_faces,
                                                vertex_colors=self.smpl_points[:, 3:6],process=False)  # process=False不对数据预处理
        #
        # self.recover_trimesh.show()
        # screen = self.smpl_trimesh.scene ()
        # screen.add_geometry (self.recover_trimesh)
        # screen.show()

    def __call__(self):
        # self.verts,self.faces=self.stich_mesh()
        # if self.out_path:
        #     self.writeobj(self.out_path,self.verts,self.faces)
        return self.stich_mesh ()

    def slice_faces_plane_q(self,points,faces,plane_normal,plane_origin,pre_bound_index=None):
        """
       根据向量夹角切割模型
        Parameters
        ---------
        points : (n, 30) float
            points of source mesh to slice
        faces : (n, 3) int
            Faces of source mesh to slice
        plane_normal : (3,) float
            Normal vector of plane to intersect with mesh
        plane_origin :  (3,) float
            Point on plane to intersect with mesh
        pre_bound_index：（n,) int 原模型的边界顶点
        Returns
        ----------
        new_points : (n, 30) float
            points of sliced mesh
        new_faces : (n, 3) int
            Faces of sliced mesh
        new_bound_index :(n,)切割后产生的边界顶点下标
        """

        if len (points) == 0:
            return points, faces

        # dot product of each vertex with the plane normal indexed by face
        # so for each face the dot product of each vertex is a row
        # shape is the same as faces (n,3)
        #sum((3,)*(n,3).T)->(n)
        dots = np.einsum ('i,ij->j', plane_normal,(points[:,:3] - plane_origin).T) #（n,) 每个顶点与原向量夹角

        flag_points=np.zeros(points.shape[0])
        flag_points[dots < -1e-8] = 1 #outside
        flag_points[dots > 1e-8] = -1 #inside
        flag_points[np.logical_and (dots >= -1e-8, dots <= 1e-8)] = 0 #on plane
        # Find vertex orientations w.r.t. faces for all triangles:
        #  -1 -> vertex "inside" plane (positive normal direction)
        #   0 -> vertex on plane
        #   1 -> vertex "outside" plane (negative normal direction)
        # signs = np.zeros (faces.shape, dtype=np.int8)#(n,3)
        #
        # signs[dots[faces] < -1e-8] = 1
        # signs[dots[faces] > 1e-8] = -1
        # signs[np.logical_and (dots[faces] >= -1e-8, dots[faces] <= 1e-8)] = 0
        signs = flag_points[faces]#(faces_n,3)
        #signs->(n,3) 通过每个面内的顶点与原向量夹角确定该点在平面哪一侧

        # Find all triangles that intersect this plane
        # onedge <- 边与面相交
        # inside <- indices of all triangles "inside" the plane (positive normal)
        signs_sum = signs.sum (axis=1, dtype=np.int8) #(faces_n,)
        signs_asum = np.abs (signs).sum (axis=1, dtype=np.int8)

        # Cases:
        # (0,0,0),  (-1,0,0),  (-1,-1,0), (-1,-1,-1) <- inside
        # (1,0,0),  (1,1,0),   (1,1,1)               <- outside
        # (1,0,-1), (1,-1,-1), (1,1,-1)              <- onedge
        onedge = np.logical_and (signs_asum >= 2,
                                 np.abs (signs_sum) <= 1)
        inside = (signs_sum == -signs_asum)

        #获取边界点，
        bount_index_1=np.argwhere(flag_points==0)[:,0] #在平面上的点为边界点
        #与平面相交的面中在inside一侧的点为边界点
        temp_index=np.argwhere(signs[onedge]<0) #(m,2)
        bound_index_2 = faces[onedge][temp_index[:,0],temp_index[:,1]]
        bount_index=np.unique(np.append(bount_index_1,bound_index_2))

        # Automatically include all faces that are "inside"
        new_faces = faces[inside].reshape (-1)
        if pre_bound_index is not None:
            new_index = np.concatenate ((new_faces, bount_index,pre_bound_index), axis=0)
        else:
            new_index=np.concatenate((new_faces,bount_index),axis=0)
        # find the unique indices in the new faces
        # using an integer-only unique function
        unique, inverse = grouping.unique_bincount (new_index,
                                                    minlength=len (points),
                                                    return_inverse=True)

        # use the unique indices for our final points and faces
        final_vert = points[unique]
        if pre_bound_index is not None:
            final_face = inverse[:new_faces.shape[0]].reshape ((-1, 3))
            new_bound_index=inverse[new_faces.shape[0]:new_faces.shape[0]+bount_index.shape[0]]
            pre_bound_index=inverse[new_faces.shape[0]+bount_index.shape[0]:]
            return final_vert, final_face,new_bound_index,pre_bound_index
        final_face = inverse[:new_faces.shape[0]].reshape ((-1, 3))
        new_bound_index = inverse[new_faces.shape[0]:]

        return final_vert, final_face,new_bound_index

    def slice_faces_plane(self,points,faces,plane_normal,plane_origin,pre_bound_index=None):
        """
        Slice a mesh (given as a set of faces and points) with a plane, returning a
        new mesh (again as a set of faces and points) that is the
        portion of the original mesh to the positive normal side of the plane.

        Parameters
        ---------
        points : (n, 30) float
            Vertices of source mesh to slice
        faces : (n, 3) int
            Faces of source mesh to slice
        plane_normal : (3,) float
            Normal vector of plane to intersect with mesh
        plane_origin :  (3,) float
            Point on plane to intersect with mesh
        cached_dots : (n, 3) float
            If an external function has stored dot
            products pass them here to avoid recomputing

        Returns
        ----------
        new_points : (n, 3) float
            Vertices of sliced mesh
        new_faces : (n, 3) int
            Faces of sliced mesh
        """

        if len (points) == 0:
            return points, faces
        # dot product of each vertex with the plane normal indexed by face
        # so for each face the dot product of each vertex is a row
        # shape is the same as faces (n,3)
        # dots = np.einsum ('i,ij->j', plane_normal,(points - plane_origin).T)[faces]
        dots = np.einsum ('i,ij->j', plane_normal, (points[:, :3] - plane_origin).T)  # （n,) 每个顶点与原向量夹角

        flag_points = np.zeros (points.shape[0])
        flag_points[dots < -1e-8] = 1  # outside
        flag_points[dots > 1e-8] = -1  # inside
        flag_points[np.logical_and (dots >= -1e-8, dots <= 1e-8)] = 0  # on plane
        # Find vertex orientations w.r.t. faces for all triangles:
        #  -1 -> vertex "inside" plane (positive normal direction)
        #   0 -> vertex on plane
        #   1 -> vertex "outside" plane (negative normal direction)
        signs = flag_points[faces]  # (faces_n,3)

        # Find all triangles that intersect this plane
        # onedge <- indices of all triangles intersecting the plane
        # inside <- indices of all triangles "inside" the plane (positive normal)
        signs_sum = signs.sum (axis=1, dtype=np.int8)
        signs_asum = np.abs (signs).sum (axis=1, dtype=np.int8)

        # Cases:
        # (0,0,0),  (-1,0,0),  (-1,-1,0), (-1,-1,-1) <- inside
        # (1,0,0),  (1,1,0),   (1,1,1)               <- outside
        # (1,0,-1), (1,-1,-1), (1,1,-1)              <- onedge
        onedge = np.logical_and (signs_asum >= 2,np.abs (signs_sum) <= 1)

        inside = (signs_sum == -signs_asum)

        bount_index = np.argwhere (flag_points == 0)[:, 0]  # 在平面上的点为边界点
        bount_color_weigth=np.mean(points[faces[onedge].reshape((-1,))][:,3:],axis=0) #(27,)
        # Automatically include all faces that are "inside"
        new_faces = faces[inside]

        # Separate faces on the edge into two cases: those which will become
        # quads (two points inside plane) and those which will become triangles
        # (one vertex inside plane)
        triangles = points[faces][:,:,0:3]#(face_n,3,3)
        cut_triangles = triangles[onedge]##(onedge_n,3,3)
        cut_faces_quad = faces[np.logical_and (onedge, signs_sum < 0)] #(1,-1,-1) 得到点的index #(cut_faces_quad_n,3) 3->points_index
        cut_faces_tri = faces[np.logical_and (onedge, signs_sum >= 0)]#(1,1,-1),(1,0,-1)

        cut_signs_quad = signs[np.logical_and (onedge, signs_sum < 0)] #对应点在平面哪一边#(cut_faces_quad_n,3)  3-> 0;-1;1
        cut_signs_tri = signs[np.logical_and (onedge, signs_sum >= 0)]

        # If no faces to cut, the surface is not in contact with this plane.
        # Thus, return a mesh with only the inside faces
        if len (cut_faces_quad) + len (cut_faces_tri) == 0:

            if len (new_faces) == 0:
                # if no new faces at all return empty arrays
                empty = (np.zeros ((0, 3), dtype=np.float64),
                         np.zeros ((0, 3), dtype=np.int64))
                return empty

            # 获取边界点，


            bount_index = np.unique (bount_index)

            # Automatically include all faces that are "inside"
            new_faces = faces[inside].reshape (-1)
            if pre_bound_index is not None:
                new_index = np.concatenate ((new_faces, bount_index, pre_bound_index), axis=0)
            else:
                new_index = np.concatenate ((new_faces, bount_index), axis=0)
            # find the unique indices in the new faces
            # using an integer-only unique function
            unique, inverse = grouping.unique_bincount (new_index,
                                                        minlength=len (points),
                                                        return_inverse=True)

            # use the unique indices for our final points and faces
            final_vert = points[unique]
            if pre_bound_index is not None:
                final_face = inverse[:new_faces.shape[0]].reshape ((-1, 3))
                new_bound_index = inverse[new_faces.shape[0]:new_faces.shape[0] + bount_index.shape[0]]
                pre_bound_index = inverse[new_faces.shape[0] + bount_index.shape[0]:]

                return final_vert, final_face, new_bound_index, pre_bound_index

            final_face = inverse[:new_faces.shape[0]].reshape ((-1, 3))
            new_bound_index = inverse[new_faces.shape[0]:]


            return final_vert, final_face,new_bound_index

        # Extract the intersections of each triangle's edges with the plane
        o = cut_triangles  # origins #(onedge_n,3,3)
        d = np.roll (o, -1, axis=1) - o  # directions
        num = (plane_origin - o).dot (plane_normal)  # compute num/denom
        denom = np.dot (d, plane_normal)
        denom[denom == 0.0] = 1e-12  # prevent division by zero
        dist = np.divide (num, denom)
        # intersection points for each segment
        int_points = np.einsum ('ij,ijk->ijk', dist, d) + o

        # Initialize the array of new points with the current points
        new_points = points

        # Handle the case where a new quad is formed by the intersection
        # First, extract the intersection points belonging to a new quad
        quad_int_points = int_points[(signs_sum < 0)[onedge], :, :] #(quad_int_points_n,3)
        num_quads = len (quad_int_points)
        if num_quads > 0:
            # Extract the vertex on the outside of the plane, then get the points
            # (in CCW order of the inside points)
            quad_int_inds = np.where (cut_signs_quad == 1)[1]
            quad_int_verts = cut_faces_quad[
                np.stack ((range (num_quads), range (num_quads)), axis=1),
                np.stack (((quad_int_inds + 1) % 3, (quad_int_inds + 2) % 3), axis=1)]
            # (cut_faces_quad_n,3) 3->points_index
            # Fill out new quad faces with the intersection points as points
            new_quad_faces = np.append (
                quad_int_verts,
                np.arange (len (new_points),
                           len (new_points) +
                           2 * num_quads).reshape (num_quads, 2), axis=1)

            # Extract correct intersection points from int_points and order them in
            # the same way as they were added to faces
            new_quad_points = quad_int_points[
                                np.stack ((range (num_quads), range (num_quads)), axis=1),
                                np.stack ((((quad_int_inds + 2) % 3).T, quad_int_inds.T),
                                          axis=1), :].reshape (2 * num_quads, 3)
            new_quad_points=np.concatenate((new_quad_points,np.tile(bount_color_weigth,[len(new_quad_points),1])),axis=1)
            # Add new points to existing points, triangulate quads, and add the
            # resulting triangles to the new faces
            bount_index=np.append(bount_index,range(len (new_points),len (new_points) +len(new_quad_points)),axis=0)
            new_points = np.append (new_points, new_quad_points, axis=0)
            new_tri_faces_from_quads = geometry.triangulate_quads (new_quad_faces)
            new_faces = np.append (new_faces, new_tri_faces_from_quads, axis=0)

        # Handle the case where a new triangle is formed by the intersection
        # First, extract the intersection points belonging to a new triangle
        tri_int_points = int_points[(signs_sum >= 0)[onedge], :, :]
        num_tris = len (tri_int_points)
        if num_tris > 0:
            # Extract the single vertex for each triangle inside the plane and get the
            # inside points (CCW order)
            tri_int_inds = np.where (cut_signs_tri == -1)[1]
            tri_int_verts = cut_faces_tri[range (
                num_tris), tri_int_inds].reshape (num_tris, 1)

            # Fill out new triangles with the intersection points as points
            new_tri_faces = np.append (
                tri_int_verts,
                np.arange (len (new_points),
                           len (new_points) +
                           2 * num_tris).reshape (num_tris, 2),
                axis=1)

            # Extract correct intersection points and order them in the same way as
            # the points were added to the faces
            new_tri_points = tri_int_points[
                               np.stack ((range (num_tris), range (num_tris)), axis=1),
                               np.stack ((tri_int_inds.T, ((tri_int_inds + 2) % 3).T),
                                         axis=1),
                               :].reshape (2 * num_tris, 3)
            new_tri_points = np.concatenate((new_tri_points,np.tile(bount_color_weigth,[len(new_tri_points),1])),axis=1)  # (new_tri_points_n,30)
            # Append new points and new faces
            bount_index = np.append (bount_index, range (len (new_points), len (new_points) + len (new_tri_points)),
                                     axis=0)
            new_points = np.append (new_points, new_tri_points, axis=0)
            new_faces = np.append (new_faces, new_tri_faces, axis=0)

        new_faces = new_faces.reshape (-1)
        if pre_bound_index is not None:
            new_index = np.concatenate ((new_faces, bount_index, pre_bound_index), axis=0)
        else:
            new_index = np.concatenate ((new_faces, bount_index), axis=0)
        # find the unique indices in the new faces
        # using an integer-only unique function
        unique, inverse = grouping.unique_bincount (new_index,
                                                    minlength=len (new_points),
                                                    return_inverse=True)

        # use the unique indices for our final points and faces
        final_vert = new_points[unique]
        if pre_bound_index is not None:
            final_face = inverse[:new_faces.shape[0]].reshape ((-1, 3))
            new_bound_index = inverse[new_faces.shape[0]:new_faces.shape[0] + bount_index.shape[0]]
            pre_bound_index = inverse[new_faces.shape[0] + bount_index.shape[0]:]

            return final_vert, final_face, new_bound_index, pre_bound_index

        final_face = inverse[:new_faces.shape[0]].reshape ((-1, 3))
        new_bound_index = inverse[new_faces.shape[0]:]

        return final_vert, final_face, new_bound_index

    def verts2faces(self, verts_points, recover_verts_index, smpl_verts_index,body_verts_color_weigth,smpl_verts_color_weigth, n, axis=30):
        '''

        :param verts_points: 缝合部位顶点（m,n,3) recover->smpl m表示recover->smpl横向点数，n表示边界点数
        :param smpl_verts_index: 边界顶点在原顶点集中的index应用于生成面的下标
        :param recover_verts_index
        :param verts_color_weigth ； 权重(n,27)
        :param n: 已经组合的 顶点数
        :param axis: 权重维度
        :return:
        '''
        remove_n = 1  # 去除两行在原frond back 中存在的顶点（可以考虑去除四行）
        verts = verts_points[remove_n:-remove_n, :, :]  #(m,n,3)
        heigh, wigth = verts.shape[0:2]
        idx = np.arange (0, (heigh * wigth)).reshape ((heigh, wigth))  #
        idx = np.concatenate ((idx, np.expand_dims (idx[:, 0], axis=1)), axis=1)  # (h，w+1)

        p00_idx = idx[:-1, :-1].reshape (-1, 1)
        p10_idx = idx[1:, :-1].reshape (-1, 1)
        p11_idx = idx[1:, 1:].reshape (-1, 1)
        p01_idx = idx[:-1, 1:].reshape (-1, 1)
        trimesh_faces = np.vstack (
            (np.hstack ((p00_idx, p01_idx, p10_idx)), np.hstack ((p01_idx, p11_idx, p10_idx))))

        mesh = trimesh.Trimesh (vertices=verts.reshape (-1, 3), faces=trimesh_faces, process=False)
        # mesh.rezero()
        # mesh.invert()
        # mesh.fix_normals()
        # mesh.show ()
        smoothing_mesh = smoothing.filter_humphrey (mesh)
        # smoothing_mesh = smoothing.filter_taubin (mesh)
        # smoothing_mesh=smoothing.filter_laplacian (mesh, iterations=5)#顶点产生偏移
        # smoothing_mesh.show ()

        smoothing_verts = smoothing_mesh.vertices.reshape (heigh, wigth, 3)

        alph = np.expand_dims (np.expand_dims (np.arange (0, 1, 1 / heigh), axis=1), axis=2)  # (heigh,1,1)

        tile_smpl_verts_color_weigth = np.tile (smpl_verts_color_weigth, [heigh, 1, 1])
        tile_body_verts_color_weigth = np.tile (body_verts_color_weigth, [heigh, 1, 1])
        points_color_weigth = np.add (np.multiply (tile_body_verts_color_weigth,  1-alph),
                                      np.multiply (tile_smpl_verts_color_weigth, alph))

        # points_color_weigth = np.tile (verts_color_weigth, [heigh, 1, 1])

        points = np.concatenate ((smoothing_verts, points_color_weigth), axis=2).reshape (-1, axis)  # (n*m,axis)

        # idx (h+2,w+1)
        smpl_verts_index = np.append (smpl_verts_index, smpl_verts_index[0])  # w+1
        recover_verts_index = np.append (recover_verts_index, recover_verts_index[0])  # w+1
        stich_idx = np.concatenate ((np.expand_dims (recover_verts_index, axis=0), idx + n,
                                     np.expand_dims (smpl_verts_index, axis=0)), axis=0)

        stich_p00_idx = stich_idx[:-1, :-1].reshape (-1, 1)
        stich_p10_idx = stich_idx[1:, :-1].reshape (-1, 1)
        stich_p11_idx = stich_idx[1:, 1:].reshape (-1, 1)
        stich_p01_idx = stich_idx[:-1, 1:].reshape (-1, 1)
        stich_faces = np.vstack ((np.hstack ((stich_p00_idx, stich_p01_idx, stich_p10_idx)),
                                  np.hstack ((stich_p01_idx, stich_p11_idx, stich_p10_idx))))

        return stich_faces, points

    def smoothing(self, verts):

        heigh, wigth = verts.shape[0:2]

        idx = np.arange (0, (heigh * wigth)).reshape ((heigh, wigth))  #
        idx = np.concatenate ((idx, np.expand_dims (idx[:, 0], axis=1)), axis=1)  # (h+2，w+1)首尾相连

        p00_idx = idx[:-1, :-1].reshape (-1, 1)
        p10_idx = idx[1:, :-1].reshape (-1, 1)
        p11_idx = idx[1:, 1:].reshape (-1, 1)
        p01_idx = idx[:-1, 1:].reshape (-1, 1)
        faces = np.vstack (
            (np.hstack ((p00_idx, p10_idx, p01_idx)), np.hstack ((p01_idx, p10_idx, p11_idx))))

        verts = verts.reshape (-1, 3)
        mesh = trimesh.Trimesh (vertices=verts, faces=faces, process=False)
        mesh.show ()
        smoothing_mesh = smoothing.filter_humphrey (mesh)
        smoothing_mesh.show ()

        smoothing_verts = smoothing_mesh.vertices.reshape (heigh, wigth, 3)

        return smoothing_verts

        # return faces,points

    def get_bound_verts_index(self, mesh):
        edges_unique, edges_index, edges_count = np.unique (mesh.edges_unique_inverse, axis=0, return_counts=True,
                                                            return_index=True)
        bound_index = np.select ([edges_count == 1], [edges_index], -1)
        bound_index = np.delete (bound_index, np.where (bound_index == -1))
        # bound_unique = np.select (edges_count == 1, edges_unique)
        bound_unique = mesh.edges_unique[mesh.edges_unique_inverse[bound_index]]
        bound_verts_index = np.zeros (bound_unique.shape[0] + 1).astype (np.int)
        bound_verts_index[0] = bound_unique[0, 0]
        bound_verts_index[1] = bound_unique[0, 1]
        bound_unique_temp = np.delete (bound_unique, 0, axis=0)

        for i in range (1, bound_verts_index.shape[0] - 1):  # 得到有序顶点索引
            index = np.where (bound_unique_temp[:, 0] == bound_verts_index[i])
            # if len(index[0])>1:
            #     print(index)
            if len (index[0]) == 1:
                bound_verts_index[i + 1] = bound_unique_temp[index, 1]

            else:
                index = np.where (bound_unique_temp[:, 1] == bound_verts_index[i])
                # if len (index[0]) ==0:
                #     print (index)
                bound_verts_index[i + 1] = bound_unique_temp[index, 0]
            bound_unique_temp = np.delete (bound_unique_temp, index, axis=0)

        in_bound_verts_index_list = []  # 内层的顶点
        for i in range (bound_verts_index.shape[0] - 1):
            neighbors_1 = mesh.vertex_neighbors[bound_verts_index[i]]  # [,]
            neighbors_2 = mesh.vertex_neighbors[bound_verts_index[i + 1]]
            ret = list (set (neighbors_1).intersection (set (neighbors_2)))
            in_bound_verts_index_list.append (ret[0])

        # in_bound_verts_index_list_set=list(set(in_bound_verts_index_list))#去重，不改变位置
        # in_bound_verts_index_list_set.sort(key=in_bound_verts_index_list.index)
        # in_bound_verts_index_list_set.append(in_bound_verts_index_list_set[0])#首尾相连
        in_bound_verts_index = np.asarray (in_bound_verts_index_list)

        return bound_verts_index[:-1], in_bound_verts_index

    def gen_Bspline_curve(self, points, degree):
        points=np.concatenate((points,points[0][None,:]),axis=0).tolist()
        curve = B_Spline.B_spline_curve (points, degree)
        return curve

    def gen_Bspline_curve_multi(self, points, degree):
        curves = B_Spline.B_spline_curve_multi (points, degree)
        return curves

    def gen_Bspline_surf(self, verts, degree_u,degree_v):
        '''

        :param verts: [verts_1,verts_2,verts_3,verts_4];  verts_1->(n,3),np
        :param degree_u: 3
        :param degree_v: 3
        :return:
        '''
        size_u = len(verts)#4
        size_v = len(verts[0])+1
        roll_n=self.align_bounds(verts[1],verts[2])
        # roll_n=0
        # mindis_pair_index=self.get_mindis_pair_index(verts[0],verts[3])
        # verts_1=(np.roll(verts[0],-mindis_pair_index[0],axis=0))
        verts_1=verts[0]
        verts_1 = np.concatenate((verts_1,verts_1[0][None,:]),axis=0)

        # verts_2=np.roll(verts[1],-mindis_pair_index[0],axis=0)
        verts_2 = verts[1]
        verts_2 = np.concatenate ((verts_2, verts_2[0][None,:]), axis=0)

        # verts_3 = np.roll (verts[2], -mindis_pair_index[1], axis=0)
        verts_3 = np.roll (verts[2], -roll_n, axis=0)
        verts_3 = np.concatenate ((verts_3, verts_3[0][None,:]), axis=0)

        verts_4 = np.roll (verts[3], -roll_n, axis=0)
        verts_4 = np.concatenate ((verts_4, verts_4[0][None,:]), axis=0)
        # self.show_points ([verts_1,verts_2,verts_3,verts_4])
        # self.gen_Bspline_curve_multi (np.asarray ((verts_1, verts_2, verts_3, verts_4)), 3).show_VTK ()
        curve_points=np.concatenate((verts_1,verts_2,verts_3,verts_4),axis=0).tolist()
        surface = B_Spline.B_spline_surface (curve_points,size_u ,size_v, degree_u,degree_v)
        return surface

    def norm(self,vec):
        n = np.linalg.norm (vec)
        if n == 0:
            return None
        return vec / n

    def get_transform(self,trans):

        transform=np.eye(4)
        # theta = np.arccos (np.inner (vertor_1, vertor_2) / (np.linalg.norm (vertor_1) * np.linalg.norm (vertor_2)))
        # axis = np.squeeze (np.cross (vertor_1, vertor_2))
        # transform[:3,:3] = transforms3d.axangles.axangle2mat (axis, theta)
        transform[:3,3]=trans.T

        return  transform

    def get_mindis_pair_index(self,verts_a,verts_b):
        '''
        找到verts_a与verts_b距离最小的对应点下标
        :param verts_a:
        :param verts_b:
        :return:
        '''
        dis_a2b=self.pairwise_dist(verts_a,verts_b)
        index_a2b=np.unravel_index(np.argmin(dis_a2b,axis=None),dis_a2b.shape)
        return index_a2b

    def align_bounds(self,bounds_a,bounds_b):
        '''
        通过旋转bounds_b,计算二者彼此距离和，对齐两bounds
        :param bounds_a:
        :param bounds_b:
        :return: roll_n
        '''

        dis_a2b=self.pairwise_dist(bounds_a,bounds_b)
        index_a2b = np.argmin (dis_a2b[0], axis=None)

        return index_a2b

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

    def Sort_verts(self,curve,index,vector):
        '''
        根据curve的node性质以及给定的初始点得到有序的顶点序列 并且保证顺序为逆时针顺序
        :param curve:
        :param index:
        :return:
        '''

        verts_index=[]
        pre_index=index
        for i in range(curve.vertex_nodes.shape[0]):
            verts_index.append (pre_index)
            next_index=curve.vertex_nodes[np.argwhere(curve.vertex_nodes[:,0]==pre_index)[0][0],1]
            # verts.append (curve.points[next_index])
            pre_index=next_index
        vertices=curve.vertices[verts_index]
        vertor_1=vertices[0]-curve.centroid
        vertor_2=vertices[3]-curve.centroid
        theta=np.inner(np.cross (vertor_1, vertor_2),vector)
        if theta<0:
            return vertices[::-1]
        return vertices

    def Sort_B_verts(self,verts,vector):
        centroid=np.sum(verts,axis=0)/verts.shape[0]
        vertor_1 = verts[0] - centroid
        vertor_2 = verts[3] - centroid
        theta = np.inner (np.cross (vertor_1, vertor_2), vector)
        if theta < 0:
            return verts[::-1]
        return verts
    def show_points(self, points_list):
        n = len (points_list)
        color_list = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 128, 0],
                      [128, 0, 0], [0, 0, 128]]
        show_points = np.zeros ([0, 3])
        points_color = np.zeros ([0, 3])
        for i in range (n):
            show_points = np.concatenate ((show_points, points_list[i]), axis=0)
            color = [color_list[i]] * points_list[i].shape[0]
            color[0] = [0, 0, 0]
            color[-1] = [128, 128, 128]
            points_color = np.concatenate ((points_color, color), axis=0)
        pointsclouds = trimesh.points.PointCloud (show_points, points_color)
        pointsclouds.show ()

    def get_hand_color(self,mesh,plane_normal,plane_origin):
        lines,face_index=intersections.mesh_plane(mesh,plane_normal,plane_origin,return_faces = True)
        verts_index, inverse_index = np.unique (np.ravel (mesh.faces[face_index]).astype ('int32'), return_inverse=True)
        color=mesh.visual.vertex_colors[verts_index][:10,:3].astype(np.float)
        color=(np.roll(color,-1,axis=0)+color+np.roll(color,1,axis=0))/3
        return color

    def out_bound2in_bound_max(self,out_bound_curv,scale):
        '''
        所有顶点向内放大
        :param out_bound:
        :return:
        '''
        in_bound_curv=out_bound_curv.copy()
        # vector=in_bound_curv.centroid-in_bound_curv.vertices #(n,3)
        vector = -in_bound_curv.centroid +in_bound_curv.vertices  # (n,3)放大
        # in_bound_curv.vertices=in_bound_curv.vertices+vector*in_bound_curv.scale
        in_bound_curv.vertices = in_bound_curv.vertices + vector * scale
        return in_bound_curv

    def out_bound2in_bound_min(self,out_bound_curv,scale):
        '''
        所有顶点向内收缩
        :param out_bound:
        :return:
        '''
        in_bound_curv=out_bound_curv.copy()
        vector=in_bound_curv.centroid-in_bound_curv.vertices #(n,3)
        # vector = -in_bound_curv.centroid +in_bound_curv.vertices  # (n,3)放大
        # in_bound_curv.vertices=in_bound_curv.vertices+vector*in_bound_curv.scale
        in_bound_curv.vertices = in_bound_curv.vertices + vector * scale
        return in_bound_curv
    def replace(self):
        diff=(np.sum(np.square(self.recover_J[20] - self.recover_J[22]))**(1/2))/8
        self.recover_J[20]=self.recover_J[20]-self.norm(self.recover_J[20] - self.recover_J[18])*diff*3#只对特定模型
        self.recover_J[21] = self.recover_J[21] - self.norm (self.recover_J[21] - self.recover_J[19]) * diff*3  # 只对特定模型
        smpl_l_vector=self.norm(self.smpl_J[20]-self.smpl_J[18])
        smpl_r_vector = self.norm(self.smpl_J[21] - self.smpl_J[19])
        # recover_l_vector = self.norm(self.recover_J[20] - self.recover_J[18])
        recover_l_vector = self.norm (self.recover_J[1] - self.recover_J[2])
        # recover_r_vector = self.norm(self.recover_J[21] - self.recover_J[19])
        recover_r_vector = self.norm (self.recover_J[2] - self.recover_J[1])
        # recover_l_hand_curv_out->recover_l_hand_curv_in->smpl_l_hand_curv_in->smpl_l_hand_curv_out
        recover_l_hand_curv_out = self.recover_trimesh.section (recover_l_vector, self.recover_J[20] )
        recover_r_hand_curv_out = self.recover_trimesh.section (recover_r_vector, self.recover_J[21])
        # self.recover_trimesh.slice_plane (self.recover_J[20],self.norm(self.recover_J[2] - self.recover_J[1])).show()
        # self.smpl_trimesh.slice_plane (self.smpl_J[20], self.norm (self.smpl_J[1] - self.smpl_J[2])).show ()


        # recover_l_hand_curv_in = self.recover_trimesh.section (recover_l_vector,
        #                                                        self.recover_J[20] - recover_l_vector * diff)  # 先切，后移
        # recover_r_hand_curv_in = self.recover_trimesh.section (recover_r_vector,
        #                                                        self.recover_J[21] - recover_r_vector * diff)
        # self.show_points ([recover_l_hand_curv_in.vertices, recover_l_hand_curv_out.vertices, recover_r_hand_curv_in.vertices, recover_r_hand_curv_out.vertices])

        smpl_l_hand_curv_out = self.smpl_trimesh.section (smpl_l_vector, self.smpl_J[20])
        smpl_r_hand_curv_out = self.smpl_trimesh.section (smpl_r_vector, self.smpl_J[21])
        # smpl_l_hand_curv_in = self.smpl_trimesh.section (smpl_l_vector, self.smpl_J[20] - smpl_l_vector * diff)
        # smpl_r_hand_curv_in = self.smpl_trimesh.section (smpl_r_vector, self.smpl_J[21] - smpl_r_vector * diff)
        if recover_l_hand_curv_out.length>smpl_l_hand_curv_out.length:
            scale=recover_l_hand_curv_out.length-smpl_l_hand_curv_out.length
            smpl_l_hand_curv_in =self.out_bound2in_bound_max (smpl_l_hand_curv_out,scale)
            recover_l_hand_curv_in = self.out_bound2in_bound_min (recover_l_hand_curv_out,scale)
        else:
            scale = -recover_l_hand_curv_out.length +smpl_l_hand_curv_out.length
            smpl_l_hand_curv_in = self.out_bound2in_bound_min (smpl_l_hand_curv_out,scale)
            recover_l_hand_curv_in = self.out_bound2in_bound_max (recover_l_hand_curv_out,scale)

        if recover_r_hand_curv_out.length>smpl_r_hand_curv_out.length:
            scale = recover_r_hand_curv_out.length - smpl_r_hand_curv_out.length
            smpl_r_hand_curv_in =self.out_bound2in_bound_max (smpl_r_hand_curv_out,scale)
            recover_r_hand_curv_in = self.out_bound2in_bound_min (recover_r_hand_curv_out,scale)
        else:
            scale = -recover_r_hand_curv_out.length + smpl_r_hand_curv_out.length
            smpl_r_hand_curv_in = self.out_bound2in_bound_min (smpl_r_hand_curv_out,scale)
            recover_r_hand_curv_in = self.out_bound2in_bound_max (recover_r_hand_curv_out,scale)

        recover_l_in_transform = self.get_transform (recover_l_vector * diff)
        recover_l_hand_curv_in.apply_transform (recover_l_in_transform)
        recover_r_in_transform = self.get_transform (recover_r_vector * diff)
        recover_r_hand_curv_in.apply_transform (recover_r_in_transform)

        smpl_l_in_transform = self.get_transform (-smpl_l_vector * diff)
        smpl_l_hand_curv_in.apply_transform (smpl_l_in_transform)
        smpl_r_in_transform = self.get_transform (-smpl_r_vector * diff)
        smpl_r_hand_curv_in.apply_transform (smpl_r_in_transform)

        # if recover_l_hand_curv_out.length>smpl_l_hand_curv_in.length:#大就缩小
        #     recover_l_hand_curv_in = self.out_bound2in_bound_min (recover_l_hand_curv_out)
        # else:
        #     recover_l_hand_curv_in = self.out_bound2in_bound_max (recover_l_hand_curv_out)
        # if recover_r_hand_curv_out.length>smpl_r_hand_curv_in.length:#大就缩小
        #     recover_r_hand_curv_in = self.out_bound2in_bound_min (recover_r_hand_curv_out)
        # else:
        #     recover_r_hand_curv_in = self.out_bound2in_bound_max (recover_r_hand_curv_out)


        # self.show_points([smpl_l_hand_curv_in.vertices,smpl_l_hand_curv_out.vertices,smpl_r_hand_curv_in.vertices,smpl_r_hand_curv_out.vertices])
        smpl_l_transform = self.get_transform ( -smpl_l_hand_curv_out.centroid+recover_l_hand_curv_out.centroid+smpl_l_vector * diff*4)
        smpl_r_transform = self.get_transform (-smpl_r_hand_curv_out.centroid+recover_r_hand_curv_out.centroid+smpl_r_vector * diff*4)

        #移动smpl_hands到合适位置
        smpl_l_hand_curv_in.apply_transform(smpl_l_transform)
        smpl_l_hand_curv_out.apply_transform (smpl_l_transform)
        smpl_r_hand_curv_in.apply_transform (smpl_r_transform)
        smpl_r_hand_curv_out.apply_transform (smpl_r_transform)

        # centence=np.vstack((smpl_l_hand_curv_in.centroid,smpl_l_hand_curv_out.centroid,recover_l_hand_curv_in.centroid,recover_l_hand_curv_out.centroid))
        # self.show_points ([smpl_l_hand_curv_in.vertices, smpl_l_hand_curv_out.vertices, smpl_r_hand_curv_in.vertices, smpl_r_hand_curv_out.vertices])
        # self.show_points ([smpl_l_hand_curv_in.vertices, smpl_l_hand_curv_out.vertices, recover_l_hand_curv_in.vertices, recover_l_hand_curv_out.vertices,centence])
        # screen =smpl_l_hand_curv_in.scene()
        # screen.add_geometry (smpl_l_hand_curv_out)
        # screen.add_geometry (recover_l_hand_curv_in)
        # screen.add_geometry (recover_l_hand_curv_out)
        #
        # screen.add_geometry (smpl_r_hand_curv_in)
        # screen.add_geometry (smpl_r_hand_curv_out)
        # screen.add_geometry (recover_r_hand_curv_in)
        # screen.add_geometry (recover_r_hand_curv_out)
        # screen.show ()

        #找对齐的点位置(index_1,index_2)
        # smpl_l_out2in_index=self.get_mindis_pair_index(smpl_l_hand_curv_out.vertices,smpl_l_hand_curv_in.vertices)
        # smpl_r_out2in_index = self.get_mindis_pair_index (smpl_r_hand_curv_out.vertices, smpl_r_hand_curv_in.vertices)
        # recover_l_out2in_index = self.get_mindis_pair_index (recover_l_hand_curv_out.vertices, recover_l_hand_curv_in.vertices)
        # recover_r_out2in_index = self.get_mindis_pair_index (recover_r_hand_curv_out.vertices, recover_r_hand_curv_in.vertices)
        smpl_l_out2in_index=[0,0]
        smpl_r_out2in_index = [0, 0]
        recover_l_out2in_index = [0, 0]
        recover_r_out2in_index = [0, 0]
        #对边界点排序
        smpl_l_hand_verts_in=self.Sort_verts(smpl_l_hand_curv_in,smpl_l_out2in_index[1],smpl_l_vector)
        smpl_l_hand_verts_out=self.Sort_verts(smpl_l_hand_curv_out,smpl_l_out2in_index[0],smpl_l_vector)
        smpl_r_hand_verts_in = self.Sort_verts (smpl_r_hand_curv_in, smpl_r_out2in_index[1],smpl_r_vector)
        smpl_r_hand_verts_out = self.Sort_verts (smpl_r_hand_curv_out, smpl_r_out2in_index[0],smpl_r_vector)
        # self.show_vertices ([smpl_l_hand_verts_in, smpl_l_hand_verts_out, smpl_r_hand_verts_in, smpl_r_hand_verts_out])
        recover_l_hand_verts_in = self.Sort_verts (recover_l_hand_curv_in, recover_l_out2in_index[1],smpl_l_vector)
        recover_l_hand_verts_out = self.Sort_verts (recover_l_hand_curv_out, recover_l_out2in_index[0],smpl_l_vector)
        recover_r_hand_verts_in = self.Sort_verts (recover_r_hand_curv_in, recover_r_out2in_index[1],smpl_r_vector)
        recover_r_hand_verts_out = self.Sort_verts (recover_r_hand_curv_out, recover_r_out2in_index[0],smpl_r_vector)
        # self.show_points (
        #     [recover_l_hand_verts_in, recover_l_hand_verts_out, recover_r_hand_verts_in, recover_r_hand_verts_out])
        # self.show_points (
        #     [recover_l_hand_verts_in, recover_l_hand_verts_out, smpl_l_hand_verts_in, smpl_l_hand_verts_out])
        #生成B样条曲线
        u_degree=3
        v_degree=2
        smpl_l_hand_bspline_curv_in=self.gen_Bspline_curve(smpl_l_hand_verts_in,v_degree)
        smpl_l_hand_bspline_curv_out = self.gen_Bspline_curve (smpl_l_hand_verts_out, v_degree)
        smpl_r_hand_bspline_curv_in = self.gen_Bspline_curve (smpl_r_hand_verts_in, v_degree)
        smpl_r_hand_bspline_curv_out = self.gen_Bspline_curve (smpl_r_hand_verts_out, v_degree)

        recover_l_hand_bspline_curv_in = self.gen_Bspline_curve (recover_l_hand_verts_in, v_degree)
        recover_l_hand_bspline_curv_out = self.gen_Bspline_curve (recover_l_hand_verts_out, v_degree)
        recover_r_hand_bspline_curv_in = self.gen_Bspline_curve (recover_r_hand_verts_in, v_degree)
        recover_r_hand_bspline_curv_out = self.gen_Bspline_curve (recover_r_hand_verts_out, v_degree)

        # L_temp = self.gen_Bspline_surf ([recover_l_hand_verts_out, recover_l_hand_verts_in,
        #                                     smpl_l_hand_verts_in, smpl_l_hand_verts_out], u_degree,
        #                                    v_degree)#为了绘制曲线
        #得到相等数量的边界点  初始输出 点集 闭合
        delta=1/smpl_l_hand_verts_in.shape[0]
        smpl_l_hand_bspline_verts_in=smpl_l_hand_bspline_curv_in(delta)[:-1,:]
        smpl_l_hand_bspline_verts_out = smpl_l_hand_bspline_curv_out (delta)[:-1,:]
        smpl_r_hand_bspline_verts_in = smpl_r_hand_bspline_curv_in (delta)[:-1,:]
        smpl_r_hand_bspline_verts_out = smpl_r_hand_bspline_curv_out (delta)[:-1,:]
        # self.show_points ([smpl_l_hand_bspline_verts_in, smpl_l_hand_bspline_verts_out, smpl_r_hand_bspline_verts_in,
        #                    smpl_r_hand_bspline_verts_out])

        recover_l_hand_bspline_verts_in = recover_l_hand_bspline_curv_in (delta)[:-1,:]
        recover_l_hand_bspline_verts_out = recover_l_hand_bspline_curv_out (delta)[:-1,:]
        recover_r_hand_bspline_verts_in = recover_r_hand_bspline_curv_in (delta)[:-1,:]
        recover_r_hand_bspline_verts_out = recover_r_hand_bspline_curv_out (delta)[:-1,:]
        # self.show_points ([recover_l_hand_bspline_verts_in, recover_l_hand_bspline_verts_out, recover_r_hand_bspline_verts_in,
        #                    recover_r_hand_bspline_verts_out])
        # self.show_points ([smpl_l_hand_bspline_verts_in, smpl_l_hand_bspline_verts_out, recover_l_hand_bspline_verts_in,
        #                    recover_l_hand_bspline_verts_out])
        smpl_l_hand_bspline_verts_in=self.Sort_B_verts(smpl_l_hand_bspline_verts_in,smpl_l_vector)
        smpl_l_hand_bspline_verts_out = self.Sort_B_verts (smpl_l_hand_bspline_verts_out, smpl_l_vector)
        smpl_r_hand_bspline_verts_in = self.Sort_B_verts (smpl_r_hand_bspline_verts_in, smpl_r_vector)
        smpl_r_hand_bspline_verts_out = self.Sort_B_verts (smpl_r_hand_bspline_verts_out, smpl_r_vector)

        recover_l_hand_bspline_verts_in = self.Sort_B_verts (recover_l_hand_bspline_verts_in, smpl_l_vector)
        recover_l_hand_bspline_verts_out = self.Sort_B_verts (recover_l_hand_bspline_verts_out, smpl_l_vector)
        recover_r_hand_bspline_verts_in = self.Sort_B_verts (recover_r_hand_bspline_verts_in, smpl_r_vector)
        recover_r_hand_bspline_verts_out = self.Sort_B_verts (recover_r_hand_bspline_verts_out, smpl_r_vector)
        #生成曲面 输入点（4，n,3) -> u=4,v=n
        L_surface=self.gen_Bspline_surf([recover_l_hand_bspline_verts_out,recover_l_hand_bspline_verts_in,
                                         smpl_l_hand_bspline_verts_in,smpl_l_hand_bspline_verts_out],u_degree,v_degree)

        R_surface = self.gen_Bspline_surf ([recover_r_hand_bspline_verts_out, recover_r_hand_bspline_verts_in,
                                            smpl_r_hand_bspline_verts_in, smpl_r_hand_bspline_verts_out], u_degree,v_degree)
        delta_v=1/recover_l_hand_verts_in.shape[0] #竖向，边界点数量
        delta_u=0.05 #横向，两模型之间要添加的点数量
        R_surface_verts,R_surface_faces=R_surface(delta_v,delta_u) #R_surface_points(u,v,3)
        L_surface_verts, L_surface_faces = L_surface (delta_v, delta_u)
        # self.show_points ([R_surface_verts[0,:,:],R_surface_verts[:,0,:],R_surface_verts[-1,:,:]])
        # self.show_points ([L_surface_verts[0, :, :], L_surface_verts[:, 0, :],L_surface_verts[-1,:,:]])
        # smoothing_verts=self.smoothing(verts_points)

        smpl_l_hand_points,smpl_l_hand_faces,smpl_l_hand_bound_index = self.slice_faces_plane (self.smpl_points,self.smpl_faces, smpl_l_vector,self.smpl_J[20])
        smpl_r_hand_points,smpl_r_hand_faces,smpl_r_hand_bound_index = self.slice_faces_plane (self.smpl_points,self.smpl_faces, smpl_r_vector,self.smpl_J[21])

        recover_l_body_points, recover_l_body_faces,pre_l_hand_bound_index= self.slice_faces_plane (self.recover_points, self.recover_faces, -recover_l_vector,self.recover_J[20])
        recover_body_points, recover_body_faces,r_hand_bound_index,l_hand_bound_index= self.slice_faces_plane (recover_l_body_points, recover_l_body_faces, -recover_r_vector,self.recover_J[21],pre_bound_index=pre_l_hand_bound_index)

        smpl_l_hand_mesh=trimesh.Trimesh(smpl_l_hand_points[:,:3],smpl_l_hand_faces,process=False)
        smpl_r_hand_mesh = trimesh.Trimesh (smpl_r_hand_points[:,:3], smpl_r_hand_faces, process=False)
        # recover_body_mesh=trimesh.Trimesh (recover_body_points[:,:3], recover_body_faces, process=False)

        smpl_l_hand_mesh.apply_transform (smpl_l_transform)
        smpl_r_hand_mesh.apply_transform (smpl_r_transform)

        # screen=recover_body_mesh.scene()
        # screen.add_geometry (recover_body_mesh)
        # screen.add_geometry(smpl_l_hand_mesh)
        # screen.add_geometry(smpl_r_hand_mesh)
        # screen.show()
        #得到几个部件的点与面
        smpl_l_hand_points[:,:3]=smpl_l_hand_mesh.vertices#transform vertices
        smpl_r_hand_points[:,:3] = smpl_r_hand_mesh.vertices

        #recover_body_points->smpl_l_hand_points->smpl_r_hand_points->l_surface_points->r_surface_points
        recover_body_len=recover_body_points.shape[0]#身体部位有多少顶点
        smpl_l_hand_len=smpl_l_hand_points.shape[0]
        smpl_r_hand_len = smpl_r_hand_points.shape[0]
        smpl_l_hand_faces +=recover_body_len
        smpl_r_hand_faces +=(recover_body_len+smpl_l_hand_len)
        # trimesh.points.PointCloud(np.concatenate((L_surface_verts[-1,:,:],recover_body_points[l_hand_bound_index][:,:3]),axis=0)).show()
        #计算生成的曲面上的边缘点与原模型的匹配
        l_surface2recover_parase=l_hand_bound_index[np.argmin(self.pairwise_dist(L_surface_verts[0,:,:],
                                                              recover_body_points[l_hand_bound_index][:,:3]),axis=1)]
        # trimesh.points.PointCloud(recover_body_points[np.argmin(self.pairwise_dist(L_surface_verts[0,:,:],
        #                                                       recover_body_points[l_hand_bound_index][:,:3]),axis=1)][:,:3]).show()
        r_surface2recover_parase = r_hand_bound_index[np.argmin (
            self.pairwise_dist (R_surface_verts[0,:,:], recover_body_points[r_hand_bound_index][:,:3]), axis=1)]

        l_surface2smpl_parase = smpl_l_hand_bound_index[np.argmin (
            self.pairwise_dist (L_surface_verts[-1,:,:], smpl_l_hand_points[smpl_l_hand_bound_index][:,:3]), axis=1)]

        r_surface2smpl_parase = smpl_r_hand_bound_index[np.argmin (
            self.pairwise_dist (R_surface_verts[-1,:,:], smpl_r_hand_points[smpl_r_hand_bound_index][:,:3]), axis=1)]


        smpl_l_hand_color=self.get_hand_color(self.recover_trimesh,recover_l_vector ,self.recover_J[22]+recover_l_vector * diff*0)
        # smpl_r_hand_color=self.get_hand_color(self.recover_trimesh,recover_r_vector ,self.recover_J[23]+recover_r_vector * diff*2)
        # l_body_points, l_body_faces,l_hand_bound_index=self.slice_faces_plane (self.recover_points, self.recover_faces, recover_l_vector, self.recover_J[22]+recover_l_vector * diff*4)
        # temp_mesh=trimesh.Trimesh(vertices=l_body_points[:,:3],faces=l_body_faces,vertex_colors=l_body_points[:,3:6])
        # temp_mesh.show()
        smpl_r_hand_color=smpl_l_hand_color
        smpl_l_hand_points_n=smpl_l_hand_points.shape[0]
        smpl_r_hand_points_n = smpl_r_hand_points.shape[0]
        smpl_l_hand_points[:, 3:6] = np.tile(smpl_l_hand_color,((smpl_l_hand_points_n//smpl_l_hand_color.shape[0])+1,1))[:smpl_l_hand_points_n,:]
        smpl_r_hand_points[:, 3:6] = np.tile(smpl_r_hand_color,((smpl_r_hand_points_n//smpl_r_hand_color.shape[0])+1,1))[:smpl_r_hand_points_n,:]


        l_body_bound_verts_color_weigth = recover_body_points[l_surface2recover_parase][:, 3:]  # 权重,颜色(n,27)
        r_body_bound_verts_color_weigth = recover_body_points[r_surface2recover_parase][:, 3:]
        l_smpl_bound_verts_color_weigth = smpl_l_hand_points[l_surface2smpl_parase][:, 3:]  # 权重,颜色(n,27)
        r_smpl_bound_verts_color_weigth = smpl_r_hand_points[r_surface2smpl_parase][:, 3:]

        l_surface2smpl_parase += recover_body_len
        r_surface2smpl_parase += recover_body_len + smpl_l_hand_len
        total_len = recover_body_len + smpl_l_hand_len + smpl_r_hand_len
        # 左右顺序相反 保证构建的三角形顺序u正确
        #L --> smpl_L_hand -> body
        l_surface_face,l_surface_points =self.verts2faces (L_surface_verts, l_surface2recover_parase,l_surface2smpl_parase,
                                                           l_body_bound_verts_color_weigth,l_smpl_bound_verts_color_weigth,total_len)
        #R --> body-> smpl_r_hand
        r_surface_face, r_surface_points = self.verts2faces (R_surface_verts, r_surface2recover_parase,r_surface2smpl_parase,
                                                             r_body_bound_verts_color_weigth,r_smpl_bound_verts_color_weigth, total_len+l_surface_points.shape[0])


        full_faces = np.concatenate ((recover_body_faces, smpl_l_hand_faces, smpl_r_hand_faces,l_surface_face,r_surface_face), axis=0)
        full_points = np.concatenate ((recover_body_points, smpl_l_hand_points, smpl_r_hand_points,l_surface_points,r_surface_points), axis=0)

        J_3d=self.recover_J.copy()
        J_3d[20]=recover_l_hand_curv_out.centroid+recover_l_vector * diff*2
        J_3d[21] = recover_r_hand_curv_out.centroid+recover_r_vector * diff*2
        J_3d[22]=self.smpl_J[22]+smpl_l_transform[:3,3].T
        J_3d[23] = self.smpl_J[23] + smpl_r_transform[:3, 3].T
        # self.show_points ([J_3d,self.smpl_J,self.recover_J])

        # mesh=trimesh.Trimesh(full_points[:,:3],full_faces,vertex_colors=full_points[:,3:6]).show()
        # mesh.show()

        return full_points, full_faces,J_3d

    def trimes_index2points_index(self, trimes_index_out, trimes_index_in, faces, mesh):
        vertex_faces_index_out = mesh.vertex_faces[trimes_index_out]
        vertex_faces_index_out = [np.delete (i, np.where (i == -1)) for i in vertex_faces_index_out]  # 去除-1
        faces_vertex_out = [faces[i] for i in vertex_faces_index_out]
        points_index_out = [reduce (np.intersect1d, (i)) for i in faces_vertex_out]

        vertex_faces_index_in = mesh.vertex_faces[trimes_index_in]
        vertex_faces_index_in = [np.delete (i, np.where (i == -1)) for i in vertex_faces_index_in]  # 去除-1
        faces_vertex_in = [faces[i] for i in vertex_faces_index_in]
        points_index_in = [reduce (np.intersect1d, (i)) for i in faces_vertex_in]

        points_index_out_2points = np.where (np.asarray ([len (index) for index in points_index_out]) > 1)[0]
        points_index_in_2points = np.where (np.asarray ([len (index) for index in points_index_in]) > 1)[0]

        # 去除非边界点
        for i in points_index_out_2points:
            points_index_out[i] = np.setdiff1d (points_index_out[i], points_index_in[i])

        for i in points_index_in_2points:
            points_index_in[i] = np.setdiff1d (points_index_in[i], points_index_out[i])

        # # points_index_out=[np.setdiff1d(points_index_out[i], points_index_in[i]) for i  in points_index_out_2points ]
        # points_index_in = [np.setdiff1d (points_index_in[i], points_index_out[i]) for i in points_index_in_2points]

        return np.asarray ([index[0] for index in points_index_out]), np.asarray (
            [index[0] for index in points_index_in])

    def save_mesh(self, points, faces, out_path):
        output_file = os.path.join (out_path, 'out.ply')
        wigth = np.mean (points[:, 0])
        high = np.mean (points[:, 1])
        fix_p = np.mean (points[:, 2])
        points = points[:, 0:3].copy ()
        points[:, 0:3] = -(points[:, 0:3] - np.array ([[wigth / 2, high / 2, fix_p]])) / (
                (wigth + high) / 4.0)
        # points[:, 0:2] = -(points[:, 0:2] - np.array ([[wigth / 2 , high/ 2 ]]))
        points[:, 0] = -points[:, 0]
        colors = points[:, 3:6].astype (np.uint8)
        # weigths = points[:, 6:]
        mesh = trimesh.Trimesh (points=points, faces=faces, vertex_colors=colors)
        mesh.export (output_file)

    def save2npy(self, path, map):
        np.save (path, map)


if __name__ == '__main__':
    front_depth = np.load ('../data/reconstruct_output/depth_front.npy')
    front_color = cv2.imread ('../data/images/baoluo.png')
    front_color = cv2.cvtColor (front_color, cv2.COLOR_BGR2RGB)
    back_depth = np.load ('../data/reconstruct_output/depth_back.npy')
    back_color = cv2.imread ('../data/images/back_rgb.png')
    back_color = cv2.cvtColor (back_color, cv2.COLOR_BGR2RGB)
    warp_smplh_value = np.load ('../data/baoluo/' + 'warp_and_filled.npy')
    rgb_mask = np.where (front_depth + back_depth == 0, 0, 255).astype ('uint8')
    # D = Depth2Mesh_Bspline (front_depth, front_color, back_depth, back_color, warp_smplh_value[:, :, 6:],
    #                         '../data/baoluo')
    # D()
    # D.stich_mesh ()