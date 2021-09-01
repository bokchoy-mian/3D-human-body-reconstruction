#!/usr/bin/python
#-- coding:utf8 --
import numpy as np
import cv2
import copy
from lib.reconstruct.obj_functions import writeobj
from lib.reconstruct.selectpoints import getinnerpts
from lib.reconstruct.dpcorrespondence import boundary_match

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

# apply bias on contours1, mesh_idx of back should greater than it of front
def mesh_stitch(contours_front, contours_back, mesh_idx1, mesh_idx2, vert_dict, outfile, bias=0):

    len = contours_front.shape[1]

    with open(outfile, 'a+') as output:

        for i in range(0, len):
            contours_front[0, i, 0, 0] += bias

        phi = boundary_match(contours_front, contours_back, 6)

        for i in range(0, len):

            if i == len - 1:
                continue

            mesh_list = [contours_front[0, i, 0, :], contours_front[0, i + 1, 0, :]]

            if phi[i] == phi[i+1]:
                mesh_list.append(contours_back[0, phi[i], 0, :])

            else:
                mesh_list.append(contours_back[0, phi[i + 1], 0, :])
                mesh_list.append(contours_back[0, phi[i], 0, :])

            mesh_list = np.array(mesh_list)

            if mesh_list.shape[0] == 3:
                x1, y1 = mesh_list[0, :]
                a = vert_dict[(x1, y1, mesh_idx1)]
                x2, y2 = mesh_list[1, :]
                b = vert_dict[(x2, y2, mesh_idx1)]
                x3, y3 = mesh_list[2, :]
                c = vert_dict[(x3, y3, mesh_idx2)]

                if mesh_idx1 > mesh_idx2:
                    mesh = 'f %d %d %d\n' % (c, b, a)
                elif mesh_idx1 < mesh_idx2:
                    mesh = 'f %d %d %d\n' % (a, b, c)

                output.write(mesh)

            if mesh_list.shape[0] == 4:
                x1, y1 = mesh_list[0, :]
                a = vert_dict[(x1, y1, mesh_idx1)]
                x2, y2 = mesh_list[1, :]
                b = vert_dict[(x2, y2, mesh_idx1)]
                x3, y3 = mesh_list[2, :]
                c = vert_dict[(x3, y3, mesh_idx2)]
                x4, y4 = mesh_list[3, :]
                d = vert_dict[(x4, y4, mesh_idx2)]

                if mesh_idx1 > mesh_idx2:
                    mesh = 'f %d %d %d %d\n' % (d, c, b, a)
                elif mesh_idx1 < mesh_idx2:
                    mesh = 'f %d %d %d %d\n' % (a, b, c, d)

                output.write(mesh)

            # print('%d/%d stitch mesh' % (i, len))


def Depth2VerTri(depth, mask):
    # mask=depth[:,:]
    ## 1. prepare index for triangle and vertex array
    index_map = np.zeros ((mask.shape[0], mask.shape[1]), dtype=np.int64)
    index = 0
    vertex = []

    for i in range (mask.shape[0]):
        for j in range (mask.shape[1]):

            if (mask[i, j] == 0):
                continue

            index_map[i, j] = index
            index += 1

            vertex.append ([j, -i, depth[i, j]])



    ## 2. create flags for triangle
    mask_bool = np.array (mask, dtype='bool')

    ## flag1
    left_up = mask_bool.copy ()
    left_up[1:mask.shape[0], :] *= mask_bool[0:mask.shape[0] - 1, :]  # multiply up movement
    left_up[:, 1:mask.shape[1]] *= mask_bool[:, 0:mask.shape[1] - 1]  # multiply left movement
    left_up[0, :] = False
    left_up[:, 0] = False

    ## flag2
    right_down = mask_bool.copy ()
    right_down[0:mask.shape[0] - 1, :] *= mask_bool[1:mask.shape[0], :]  # multiply down movement
    right_down[:, 0:mask.shape[1] - 1] *= mask_bool[:, 1:mask.shape[1]]  # multiply right movement
    right_down[mask.shape[0] - 1, :] = False
    right_down[:, mask.shape[1] - 1] = False

    '''
      (i-1, j-1) -----(i-1, j) ------(i-1, j+1)
          |              |               |
          |              |               |
          |              |               |
       (i, j-1) ------ (i, j) ------ (i, j+1)
          |              |               |
          |              |               |
          |              |               |
      (i+1, j-1) ----(i+1, j)------(i+1, j+1)
  
    flag1 means: Δ{ (i, j), (i-1, j), (i, j-1) }
    flag2 means: Δ{ (i, j), (i+1, j), (i, j+1) }
  
    otherwise:
      case1: is not locate on edge(i, j ==0) and exist left up point
      --> Δ{ (i, j), (i-1, j-1), (i, j-1) }
  
      case2: is not locate on edge(i, j ==shape-1) and exist right down
      --> Δ{ (i, j), (i+1, j+1), (i, j+1) }
  
    '''

    ## 3. fill triangle list like above
    triangle = []
    for i in range (mask.shape[0]):
        for j in range (mask.shape[1]):

            ## outside --> ignore
            if (not (mask_bool[i, j])):
                continue

            ## flag1
            if (left_up[i, j]):
                triangle.append ([index_map[i, j], index_map[i - 1, j], index_map[i, j - 1]])

            ## flag2
            if (right_down[i, j]):
                triangle.append ([index_map[i, j], index_map[i + 1, j], index_map[i, j + 1]])

            ## otherwise
            if (not (left_up[i, j]) and not (right_down[i, j])):

                ## case1
                if (i != 0 and j != 0 and mask_bool[i, j - 1] and mask_bool[i - 1, j - 1]):
                    triangle.append ([index_map[i, j], index_map[i - 1, j - 1], index_map[i, j - 1]])

                ## case2
                if (i != mask_bool.shape[0] - 1 and j != mask_bool.shape[1] - 1 and mask_bool[i, j + 1] and mask_bool[
                    i + 1, j + 1]):
                    triangle.append ([index_map[i, j], index_map[i + 1, j + 1], index_map[i, j + 1]])

    return np.array (vertex, dtype=np.float32), np.array (triangle, dtype=np.int64)


def gen_verts_tri(front_depth, back_depth,mask):
    # mask=depth[:,:]
    ## 1. prepare index for triangle and vertex array
    index_map = np.zeros ((2,mask.shape[0], mask.shape[1]), dtype=np.int64)
    index = 0
    vertex = []

    #for front
    for i in range (mask.shape[0]):
        for j in range (mask.shape[1]):

            if (mask[i, j] == 0):
                continue

            index_map[0,i, j] = index
            index += 1

            vertex.append ([j, -i, front_depth[i, j]])

    #for back
    for i in range (mask.shape[0]):
        for j in range (mask.shape[1]):

            if (mask[i, j] == 0):
                continue

            index_map[1,i, j] = index
            index += 1

            vertex.append ([j, -i, back_depth[i, j]])

    ## 2. create flags for triangle
    mask_bool = np.array (mask, dtype='bool')

    ## flag1
    left_up = mask_bool.copy ()
    left_up[1:mask.shape[0], :] *= mask_bool[0:mask.shape[0] - 1, :]  # multiply up movement
    left_up[:, 1:mask.shape[1]] *= mask_bool[:, 0:mask.shape[1] - 1]  # multiply left movement
    left_up[0, :] = False
    left_up[:, 0] = False

    ## flag2
    right_down = mask_bool.copy ()
    right_down[0:mask.shape[0] - 1, :] *= mask_bool[1:mask.shape[0], :]  # multiply down movement
    right_down[:, 0:mask.shape[1] - 1] *= mask_bool[:, 1:mask.shape[1]]  # multiply right movement
    right_down[mask.shape[0] - 1, :] = False
    right_down[:, mask.shape[1] - 1] = False

    '''
      (i-1, j-1) -----(i-1, j) ------(i-1, j+1)
          |              |               |
          |              |               |
          |              |               |
       (i, j-1) ------ (i, j) ------ (i, j+1)
          |              |               |
          |              |               |
          |              |               |
      (i+1, j-1) ----(i+1, j)------(i+1, j+1)

    flag1 means: Δ{ (i, j), (i-1, j), (i, j-1) }
    flag2 means: Δ{ (i, j), (i+1, j), (i, j+1) }
    
    for bound 
    if abs([diff-x,diff-y])==[1,1]：
        if :
            (i-1, j-1) 
                  |                            
                  |                            
                  |                           
               (i, j-1) ------ (i, j) 
           diff-x==diff-y] and  mask_bool[i,j-1]==true
           --> Δ{ (i, j), (i-1, j-1), (i, j-1) }
        else:
            (i-1, j-1) -----(i-1, j) 
                             |            
                             |               
                             |               
                             (i, j)
            diff-x==diff-y and  mask_bool[i-1,j]==true
            --> Δ{ (i, j), (i-1, j-1), (i-1, j) }
        else:
                           (i-1, j)
                            |           
                            |              
                            |               
           (i, j-1) ------ (i, j) 
        diff-x！=diff-y and  mask_bool[i,j]==true
            --> Δ{ (i, j-1), (i-1, j), (i, j) }
        else:
        (i-1, j-1) -----(i-1, j) 
          |             
          |             
          |            
       (i, j-1) 
       diff-x！=diff-y and  mask_bool[i-1,j-1]==true
            --> Δ{ (i, j-1), (i-1, j), (i-1, j-1) }
    '''

    ## 3. fill triangle list like above
    triangle = []

    #for innerpoints
    for i in range (mask.shape[0]):
        for j in range (mask.shape[1]):

            ## outside --> ignore
            if (not (mask_bool[i, j])):
                continue

            ## flag1
            if (left_up[i, j]):
                triangle.append ([index_map[0,i, j], index_map[0,i - 1, j], index_map[0,i, j - 1]])
                triangle.append ([index_map[1, i, j], index_map[1, i - 1, j], index_map[1, i, j - 1]])
            ## flag2
            if (right_down[i, j]):
                triangle.append ([index_map[0,i, j], index_map[0,i + 1, j], index_map[0,i, j + 1]])
                triangle.append ([index_map[1, i, j], index_map[1, i + 1, j], index_map[1, i, j + 1]])

    #for boundpoints
    bound = get_boundary (mask.T)
    for i in range(bound.shape[0]-1):
        diff_x, diff_y = bound[i + 1]-bound[i]
        if abs(diff_x)==1 and abs(diff_y)==1:
            point_mean=(bound[i + 1]+bound[i])/2.0
            if diff_y==diff_x:#同号
                temp_point=(point_mean+[0.5,-0.5]).astype('int32')
                if (mask_bool[temp_point[0],temp_point[1]]):
                    triangle.append([index_map[0, bound[i+1][0], bound[i+1][1]],
                                     index_map[0, bound[i][0], bound[i][1]],
                                     index_map[0,temp_point[0],temp_point[1]]])
                    triangle.append ([index_map[1, bound[i + 1][0], bound[i + 1][1]],
                                      index_map[1, bound[i][0], bound[i][1]],
                                      index_map[1, temp_point[0], temp_point[1]]])
                else:
                    temp_point = (point_mean - [0.5, -0.5]).astype ('int32')
                    triangle.append ([index_map[0, bound[i + 1][0], bound[i + 1][1]],
                                      index_map[0, bound[i][0], bound[i][1]],
                                      index_map[0, temp_point[0], temp_point[1]]])
                    triangle.append ([index_map[1, bound[i + 1][0], bound[i + 1][1]],
                                      index_map[1, bound[i][0], bound[i][1]],
                                      index_map[1, temp_point[0], temp_point[1]]])
            else:#异号
                temp_point = (point_mean + [0.5, 0.5]).astype ('int32')
                if (mask_bool[temp_point[0],temp_point[1]]):
                    triangle.append ([index_map[0, bound[i + 1][0], bound[i + 1][1]],
                                      index_map[0, bound[i][0], bound[i][1]],
                                      index_map[0, temp_point[0], temp_point[1]]])
                    triangle.append ([index_map[1, bound[i + 1][0], bound[i + 1][1]],
                                      index_map[1, bound[i][0], bound[i][1]],
                                      index_map[1, temp_point[0], temp_point[1]]])
                else:
                    temp_point = (point_mean - [0.5, 0.5]).astype ('uint32')
                    triangle.append ([index_map[0, bound[i + 1][0], bound[i + 1][1]],
                                      index_map[0, bound[i][0], bound[i][1]],
                                      index_map[0, temp_point[0], temp_point[1]]])
                    triangle.append ([index_map[1, bound[i + 1][0], bound[i + 1][1]],
                                      index_map[1, bound[i][0], bound[i][1]],
                                      index_map[1, temp_point[0], temp_point[1]]])
            # triangle.append ([index_map[0, x2, y2], index_map[0, x1, y1], index_map[0, x2-1, y2-1]])
            # triangle.append ([index_map[1,i, j], index_map[1,i + 1, j + 1], index_map[1,i, j + 1]])
    #数组尾部的值与头部的值比较
    diff_x, diff_y = bound[-1] - bound[0]
    if abs (diff_x) == 1 and abs (diff_y) == 1:
        point_mean = (bound[-1] + bound[0]) / 2.0
        if diff_y == diff_x:  # 同号
            temp_point = (point_mean + [0.5, -0.5]).astype ('int32')
            if (mask_bool[temp_point[0],temp_point[1]]):
                triangle.append ([index_map[0, bound[-1][0], bound[-1][1]],
                                  index_map[0, bound[0][0], bound[0][1]],
                                  index_map[0, temp_point[0], temp_point[1]]])
                triangle.append ([index_map[1, bound[-1][0], bound[-1][1]],
                                  index_map[1, bound[0][0], bound[0][1]],
                                  index_map[1, temp_point[0], temp_point[1]]])
            else:
                temp_point = (point_mean - [0.5, -0.5]).astype ('int32')
                triangle.append ([index_map[0, bound[-1][0], bound[-1][1]],
                                  index_map[0, bound[0][0], bound[0][1]],
                                  index_map[0, temp_point[0], temp_point[1]]])
                triangle.append ([index_map[1, bound[-1][0], bound[-1][1]],
                                  index_map[1, bound[0][0], bound[0][1]],
                                  index_map[1, temp_point[0], temp_point[1]]])
        else:  # 异号
            temp_point = (point_mean + [0.5, 0.5]).astype ('int32')
            if (mask_bool[temp_point[0],temp_point[1]]):
                triangle.append ([index_map[0, bound[-1][0], bound[-1][1]],
                                  index_map[0, bound[0][0], bound[0][1]],
                                  index_map[0, temp_point[0], temp_point[1]]])
                triangle.append ([index_map[1, bound[-1][0], bound[-1][1]],
                                  index_map[1, bound[0][0], bound[0][1]],
                                  index_map[1, temp_point[0], temp_point[1]]])
            else:
                temp_point = (point_mean - [0.5, 0.5]).astype ('int32')
                triangle.append ([index_map[0, bound[-1][0], bound[-1][1]],
                                  index_map[0, bound[0][0], bound[0][1]],
                                  index_map[0, temp_point[0], temp_point[1]]])
                triangle.append ([index_map[1, bound[-1][0], bound[-1][1]],
                                  index_map[1, bound[0][0], bound[0][1]],
                                  index_map[1, temp_point[0], temp_point[1]]])
        # triangle.append ([index_map[0, x2, y2], index_map[0, x1, y1], index_map[0, x2-1, y2-1]])
        # triangle.append ([index_map[1,i, j], index_map[1,i + 1, j + 1], index_map[1,i, j + 1]])

    #stich
    for i in range(bound.shape[0]-1):
        x1,y1=bound[i]
        x2,y2=bound[i+1]
        triangle.append ([index_map[0, x1, y1], index_map[0, x2, y2], index_map[1, x1, y1]])
        triangle.append ([index_map[1, x1, y1], index_map[1, x2, y2], index_map[0, x2, y2]])
    x1, y1 = bound[0]
    x2, y2 = bound[bound.shape[0]-1]
    triangle.append ([index_map[0, x1, y1], index_map[0, x2, y2], index_map[1, x1, y1]])
    triangle.append ([index_map[1, x1, y1], index_map[1, x2, y2], index_map[0, x2, y2]])
    return np.array (vertex, dtype=np.float32), np.array (triangle, dtype=np.int64)

def stitch_mesh(rgb_mask,front_depth,back_depth,out_path):
    rgb_bound=get_boundary(rgb_mask.T)
    # mask = cv2.erode (rgb_mask, np.ones ((int (3), int (3)), np.uint8))
    mask = cv2.morphologyEx (rgb_mask, cv2.MORPH_CLOSE, np.ones ((int (3), int (3)), np.uint8))#闭运算
    front_bound_depth=np.array([front_depth[i,j] for i,j in rgb_bound])
    back_bound_depth=np.array([back_depth[i,j] for i,j in rgb_bound])
    front_bound_depth_mean=np.mean(front_bound_depth)
    back_bound_depth_mean=np.mean(back_bound_depth)

    front_depth_mean=np.mean(front_depth)
    # back_depth_mean=np.mean(back_depth)

    bound_difference=back_bound_depth_mean-front_bound_depth_mean
    # total_difference=back_depth_mean-front_depth_mean
    front_difference=front_bound_depth_mean-front_depth_mean

    back_depth=back_depth-bound_difference+front_difference/4.0

    #处理穿模
    # diff_flag=back_depth-front_depth
    # index=np.where(diff_flag<=0)

    # innerpoints=getinnerpts(rgb_mask)

    # front_vertex,front_triangle=Depth2VerTri(front_depth,mask)
    # back_vertex, back_triangle = Depth2VerTri (back_depth, mask)
    #
    # writeobj ('data/baoluo_divide/front_recover.obj', front_vertex, front_triangle)
    # writeobj ('data/baoluo_divide/back_recover.obj',back_vertex, back_triangle)

    stich_vertex,stich_triangle=gen_verts_tri(front_depth,back_depth,mask)

    writeobj (out_path, stich_vertex, stich_triangle)


if __name__ == "__main__":

    import numpy as np
    front_depth=np.load('data/baoluo_divide/depth_front.npy')
    back_depth=np.load('data/baoluo_divide/depth_back.npy')
    # from normal2depth import depth_2_img
    # depth_front_img=depth_2_img(depth_front,'depth_front.png')
    # depth_back_img = depth_2_img (depth_back, 'depth_back.png')
    # cv2.imshow('front',depth_front_img)
    # cv2.imshow('back',depth_back_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # rgb_mask=cv2.imread('data/baoluo_divide/smpl_mask.png',cv2.IMREAD_GRAYSCALE)
    # front_depth_temp=np.int8(front_depth)
    rgb_mask =np.where(front_depth==0,0,255).astype('uint8')
    cv2.imwrite ('mask.png', rgb_mask )
    # rgb_mask=np.int8 (rgb_mask)
    stitch_mesh(rgb_mask, front_depth,back_depth,'data/baoluo_divide/stich-1.obj')
