#!/usr/bin/python
#-- coding:utf8 --
import argparse
import torch
import numpy as np
import time
import os
import math
import cv2
import trimesh
def get_boundary(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = contours[0].reshape (contours[0].shape[0], 2)
    # img=np.zeros(mask.shape)
    # for pt in contour:
    #     img[pt[1]][pt[0]]=255
    # cv2.imshow('1',img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return contour

def remove_points(fp, bp):
    """
    移除前后不一致的点，将其移除
    :param fp:
    :param bp:
    :return:
    """
    f0 = fp[:, :, 2] - bp[:, :, 2]#坐标差
    f0 = f0 > 0#
    fp[f0, 2] = 0.0
    bp[f0, 2] = 0.0

def getEdgeFaces(mask, fp_idx, bp_idx):
    # _, contours, _ = cv2.findContours (mask.astype (np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    all_boundary_faces_idx = []
    for i in range(len(contours)):
        edges = contours[i][:, 0, :]
        nextedges = np.vstack((edges[1:], edges[0]))
        fp_edge_idx = fp_idx[edges[:, 1], edges[:, 0]].reshape(-1, 1)
        bp_edge_idx = bp_idx[edges[:, 1], edges[:, 0]].reshape(-1, 1)
        bp_nextedge_idx = bp_idx[nextedges[:, 1], nextedges[:, 0]].reshape(-1, 1)
        fp_nextedge_idx = fp_idx[nextedges[:, 1], nextedges[:, 0]].reshape(-1, 1)
        boundary_faces_idx = np.vstack((np.hstack((fp_edge_idx, bp_edge_idx, bp_nextedge_idx)),
                                        np.hstack((fp_edge_idx, bp_nextedge_idx, fp_nextedge_idx))))
        if i == 0:
            all_boundary_faces_idx = boundary_faces_idx
        else:
            all_boundary_faces_idx = np.vstack((all_boundary_faces_idx, boundary_faces_idx))
    return all_boundary_faces_idx


def getbackFaces(mask, p_idx):
    p_valid_idx = p_idx * mask
    p00_idx = p_valid_idx[:-1, :-1].reshape(-1, 1)
    p10_idx = p_valid_idx[1:, :-1].reshape(-1, 1)
    p11_idx = p_valid_idx[1:, 1:].reshape(-1, 1)
    p01_idx = p_valid_idx[:-1, 1:].reshape(-1, 1)
    all_faces = np.vstack((np.hstack((p00_idx, p01_idx, p10_idx)), np.hstack((p01_idx, p11_idx, p10_idx)),
                           np.hstack((p00_idx, p11_idx, p10_idx)), np.hstack((p00_idx, p01_idx, p11_idx))))
    fp_faces = all_faces[np.where(all_faces[:, 0] * all_faces[:, 1] * all_faces[:, 2] > 0)]
    return fp_faces


def getfrontFaces(mask, p_idx):
    p_valid_idx = p_idx * mask
    p00_idx = p_valid_idx[:-1, :-1].reshape(-1, 1)
    p10_idx = p_valid_idx[1:, :-1].reshape(-1, 1)
    p11_idx = p_valid_idx[1:, 1:].reshape(-1, 1)
    p01_idx = p_valid_idx[:-1, 1:].reshape(-1, 1)
    all_faces = np.vstack((np.hstack((p00_idx, p10_idx, p01_idx)), np.hstack((p01_idx, p10_idx, p11_idx)),
                           np.hstack((p00_idx, p10_idx, p11_idx)), np.hstack((p00_idx, p11_idx, p01_idx))))
    fp_faces = all_faces[np.where(all_faces[:, 0] * all_faces[:, 1] * all_faces[:, 2] > 0)]
    return fp_faces

def ply_from_array_color(points, colors, faces, output_file):

    num_points = len(points)
    num_triangles = len(faces)

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
end_header\n'''.format(num_points, num_triangles)

    with open(output_file,'w') as f:
        f.writelines(header)
        index = 0
        for item in points:
            f.write("{0:0.6f} {1:0.6f} {2:0.6f} {3} {4} {5}\n".format(item[0], item[1], item[2],
                                                        colors[index, 0], colors[index, 1], colors[index, 2]))
            index = index + 1

        for item in faces:
            number = len(item)
            row = "{0}".format(number)
            for elem in item:
                row += " {0} ".format(elem)
            row += "\n"
            f.write(row)

def depth2ply(front_depth,front_color,back_depth,back_color,mask,save_dir,save_name='01'):
    high,wigth=front_depth.shape
    fix_p =(np.mean(front_depth)+np.mean(back_depth))/2
    fp_idx = np.zeros ([high, wigth], dtype=np.int)
    bp_idx = np.ones_like (fp_idx) * (high * wigth)
    for hh in range (high):
        for ww in range (wigth):
            fp_idx[hh, ww] = hh * wigth + ww
            bp_idx[hh, ww] += hh * wigth + ww

    # init X, Y coordinate tensors 生成网格
    X, Y = np.meshgrid (np.arange (wigth), np.arange (high))
    X = np.expand_dims(X,axis=2)  # (H,W,1)
    Y = np.expand_dims(Y,axis=2)
    dx=1.0
    dy=1.0
    x_cord = X * dx
    y_cord = Y * dy

   
    # convert the images to 3D mesh
    front_depth=np.expand_dims(front_depth,axis=2)
    back_depth = np.expand_dims (back_depth, axis=2)
    fpc = np.concatenate ((x_cord, y_cord, front_depth, front_color), axis=2)
    bpc = np.concatenate ((x_cord, y_cord, back_depth, back_color), axis=2)

    # remove_points (fpc, bpc)
    # get the edge region for the edge point interpolation
    mask = fpc[:, :, 2] > 0
    mask = mask.astype (np.float32)
    mask = cv2.morphologyEx (mask, cv2.MORPH_CLOSE, np.ones ((int (3), int (3)), np.uint8))  # 闭运算
    cv2.imwrite('mask.png',mask*255)
    kernel = cv2.getStructuringElement (cv2.MORPH_RECT, (3, 3))
    re_mask=mask
    # eroded = cv2.erode (mask, kernel)
    # edge = (mask - eroded).astype (np.bool)
    # # interpolate 2 points for each edge point pairs
    # fpc[edge, 2:6] = (fpc[edge, 2:6] * 2 + bpc[edge, 2:6] * 1) / 3  # 前后各加一点
    # bpc[edge, 2:6] = (fpc[edge, 2:6] * 1 + bpc[edge, 2:6] * 2) / 3
    for i in range(1,6):
        N=2**(i+1)+1

        eroded = cv2.erode (re_mask, kernel)
        edge = (re_mask - eroded).astype (np.bool)
        # interpolate 2 points for each edge point pairs
        fpc[edge, 2:6] = (fpc[edge, 2:6] * (N-1) + bpc[edge, 2:6] * 1) / N  # 前后各加一点
        bpc[edge, 2:6] = (fpc[edge, 2:6] * 1 + bpc[edge, 2:6] * (N-1)) / N
        re_mask=eroded
    fpc = fpc.reshape (-1, 6)
    bpc = bpc.reshape (-1, 6)

    f_faces = getfrontFaces (mask, fp_idx)
    b_faces = getbackFaces (mask, bp_idx)
    edge_faces = getEdgeFaces (mask, fp_idx, bp_idx)
    faces = np.vstack ((f_faces, b_faces, edge_faces))
    points = np.concatenate ((fpc, bpc), axis=0)
    # reset center point and convert mm to m
    points[:, 0:3] = -(points[:, 0:3] - np.array ([[wigth / 2 * dx, high /2 * dy, fix_p]])) / ((wigth+high)/4.0)
    # points[:, 0:2] = -(points[:, 0:2] - np.array ([[wigth / 2 , high/ 2 ]]))
    points[:, 0] = -points[:, 0]
    verts_index,inverse_index = np.unique (np.ravel (faces).astype ('int32'),return_inverse=True)

    re_faces=inverse_index.reshape(-1,3)

    re_points = points[verts_index]

    vertices = re_points[:, 0:3]
    colors = re_points[:, 3:6].astype (np.uint8)
    weigth=re_points[:, 6:]
    mesh = trimesh.Trimesh (vertices=vertices, faces=re_faces, vertex_colors=colors)

    # mkdirs

    if not os.path.isdir (os.path.join (save_dir, 'ply')):
        os.makedirs (os.path.join (save_dir, 'ply'))

    # save the results

    output_ply_name = os.path.join (save_dir, 'ply', save_name + '.ply')
    ply_from_array_color (mesh.vertices, mesh.visual.vertex_colors, mesh.faces, output_ply_name)

def gen_mesh_ply(front_depth,front_color,back_depth,back_color,rgb_mask,out_path):
    rgb_bound = get_boundary (rgb_mask.T)
    # mask = cv2.erode (rgb_mask, np.ones ((int (3), int (3)), np.uint8))
    mask = front_depth> 0
    mask = mask.astype (np.float32)
    mask = cv2.morphologyEx (mask, cv2.MORPH_CLOSE, np.ones ((int (3), int (3)), np.uint8))#闭运算
    front_bound_depth=np.array([front_depth[i,j] for i,j in rgb_bound])
    back_bound_depth=np.array([back_depth[i,j] for i,j in rgb_bound])
    front_bound_depth_mean=np.mean(front_bound_depth)
    back_bound_depth_mean=np.mean(back_bound_depth)

    front_depth_mean=np.mean(front_depth)
    back_depth_mean=np.mean(back_depth)

    bound_difference=back_bound_depth_mean-front_bound_depth_mean
    # total_difference=back_depth_mean-front_depth_mean
    front_difference=front_bound_depth_mean-front_depth_mean

    back_depth=(back_depth-bound_difference+front_difference/3.0)*mask

    depth2ply(front_depth,front_color,back_depth,back_color,mask,out_path,save_name='baoluo')



if __name__ == "__main__":
    front_depth = np.load ('../../data/reconstruct_output/depth_front.npy')
    front_color=cv2.imread('../../data/images/baoluo.png')
    front_color=cv2.cvtColor(front_color, cv2.COLOR_BGR2RGB)
    back_depth = np.load ('../../data/reconstruct_output/depth_back.npy')
    back_color = cv2.imread ('../../data/images/back_rgb.png')
    back_color = cv2.cvtColor (back_color, cv2.COLOR_BGR2RGB)
    # from normal2depth import depth_2_img
    # depth_front_img=depth_2_img(depth_front,'depth_front.png')
    # depth_back_img = depth_2_img (depth_back, 'depth_back.png')
    # cv2.imshow('front',depth_front_img)
    # cv2.imshow('back',depth_back_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # rgb_mask=cv2.imread('data/baoluo_divide/smpl_mask.png',cv2.IMREAD_GRAYSCALE)
    # front_depth_temp=np.int8(front_depth)
    rgb_mask = np.where (front_depth+back_depth == 0, 0, 255).astype ('uint8')
    gen_mesh_ply(front_depth, front_color, back_depth, back_color, rgb_mask, '../../data/baoluo')