#!/usr/bin/python
#-- coding:utf8 --
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import cv2
from scipy.sparse import diags,vstack,block_diag

class Normal2Depth():
    def __init__(self,mask,normal,out_path):
        """

        :param mask:
        :param normal:正反面normal
        """
        self.out_path=out_path
        self.mask=cv2.erode (mask, np.ones ((3, 3), np.uint8))
        # self.mask=mask
        self.normal=normal*2.0-1.0
        self.normal[mask==0]=np.zeros(normal.shape[2])

    def __call__(self):
        self.front_depth ,self.back_depth=self.normal2depth_2edge_front_back(self.mask,self.normal)
        # self.front_depth, self.back_depth = self.normal2depth_4edge_front_back (self.mask, self.normal)
        # self.front_depth=self.comp_depth_4edge(self.mask,self.normal[:,:,0:3])
        # self.back_depth=self.comp_depth_4edge(self.mask,self.normal[:,:,3:])
        # self.deel_back_depth()
        return self.front_depth,self.back_depth

    def normal2depth_2edge_front_back(self,mask, normal):
        '''
        "arbitrary point p(x, y, Z(x, y)) and Np = (nx, ny, nz)"

        v1 = (x+1, y, Z(x+1, y)) - p
           = (1, 0, Z(x+1, y) - Z(x, y))

        Then, dot(Np, v1) == 0 #right
        0 = Np * v1
          = (nx, ny, nz) * (1, 0, Z(x+1,y) - Z(x, y))
          = nx + nz(Z(x+1,y) - Z(x, y))

        --> Z(x+1,y) - Z(x, y) = -nx/nz = p

        Also dot(Np, v2) is same #up #left #bottom
        --> Z(x,y+1) - Z(x, y) = -ny/nz = q


        Finally, apply least square to find Z(x, y).
        A: round matrix
        x: matrix of Z(x, y)
        b: matrix of p and q
        A*x = b
        (--> 计算下右2个方向)

        '''

        dif = mask.size
        w = mask.shape[1]
        h = mask.shape[0]
        ## 1. fill A
        matrix_1 = sp.lil_matrix ((w, w))
        matrix_1.setdiag (-1, 0)
        matrix_1.setdiag (1, 1)
        matrix_1[-1, -1] = 0
        matrix_1 = block_diag ([matrix_1] * h)

        matrix_2_1 = sp.lil_matrix ((dif - w, dif))
        matrix_2_1.setdiag (-1, 0)
        matrix_2_1.setdiag (1, w)
        matrix_2_2 = sp.lil_matrix ((w, dif))
        matrix_2 = vstack ([matrix_2_1, matrix_2_2])

        A = vstack ([matrix_1, matrix_2])
        # matrix_1 = matrix_1.toarray ()
        # matrix_2 = matrix_2.toarray ()
        # matrix = matrix.toarray ()
        #2
        front_b = np.zeros (A.shape[0], dtype=np.float64)
        back_b = np.zeros (A.shape[0], dtype=np.float64)

        ## 3. set normal
        front_nx = normal[:, :, 0].ravel ()
        front_ny = normal[:, :, 1].ravel ()
        front_nz = normal[:, :, 2].ravel ()

        back_nx = normal[:, :, 3].ravel ()
        back_ny = normal[:, :, 4].ravel ()
        back_nz = normal[:, :, 5].ravel ()

        ## 4. fill b
        ##  --> 0~nx.shape[0] is for v1
        ##  --> .... v2, v3, v4
        #front
        front_b[0:front_nx.shape[0]] = -front_nx / (front_nz + 1e-8)
        front_b[front_nx.shape[0]:2 * front_nx.shape[0]] = -front_ny / (front_nz + 1e-8)
        # front_b[2 * front_nx.shape[0]:3 * front_nx.shape[0]] = -front_nx / (front_nz + 1e-8)
        # front_b[3 * front_nx.shape[0]:front_b.shape[0]] = -front_ny / (front_nz + 1e-8)
        #back
        back_b[0:back_nx.shape[0]] = -back_nx / (back_nz + 1e-8)
        back_b[back_nx.shape[0]:2 * back_nx.shape[0]] = -back_ny / (back_nz + 1e-8)
        # back_b[2 * back_nx.shape[0]:3 * back_nx.shape[0]] = -back_nx / (back_nz + 1e-8)
        # back_b[3 * back_nx.shape[0]:back_b.shape[0]] = -back_ny / (back_nz + 1e-8)

        ## 5. solve Ax = b
        #front
        front_AtA = A.transpose ().dot (A)
        front_Atb = A.transpose ().dot (front_b)
        front_x, front_info = sp.linalg.cg (front_AtA, front_Atb)
        #back
        back_AtA = A.transpose ().dot (A)
        back_Atb = A.transpose ().dot (back_b)
        back_x, back_info = sp.linalg.cg (back_AtA, back_Atb)
        ## 6. create output matrix
        #front
        front_depth = front_x.reshape (mask.shape)
        front_depth -= np.min (front_depth)
        front_depth[mask == 0] = 0.0
        #back
        back_depth = back_x.reshape (mask.shape)
        back_depth -= np.min (back_depth)
        back_depth[mask == 0] = 0.0

        return front_depth,back_depth

    def normal2depth_4edge_front_back(self, mask, normal):
        '''
        "arbitrary point p(x, y, Z(x, y)) and Np = (nx, ny, nz)"

        v1 = (x+1, y, Z(x+1, y)) - p
           = (1, 0, Z(x+1, y) - Z(x, y))

        Then, dot(Np, v1) == 0 #right
        0 = Np * v1
          = (nx, ny, nz) * (1, 0, Z(x+1,y) - Z(x, y))
          = nx + nz(Z(x+1,y) - Z(x, y))

        --> Z(x+1,y) - Z(x, y) = -nx/nz = p

        Also dot(Np, v2) is same #up #left #bottom
        --> Z(x,y+1) - Z(x, y) = -ny/nz = q


        Finally, apply least square to find Z(x, y).
        A: round matrix
        x: matrix of Z(x, y)
        b: matrix of p and q
        A*x = b
        (--> 计算上下左右4个方向)

        '''

        dif = mask.size
        w = mask.shape[1]
        h = mask.shape[0]
        ## 1. fill A
        matrix_1 = sp.lil_matrix ((w, w))
        matrix_1.setdiag (-1, 0)
        matrix_1.setdiag (1, 1)
        matrix_1[-1, -1] = 0
        matrix_1 = block_diag ([matrix_1] * h)

        matrix_2_1 = sp.lil_matrix ((dif - w, dif))
        matrix_2_1.setdiag (-1, 0)
        matrix_2_1.setdiag (1, w)
        matrix_2_2 = sp.lil_matrix ((w, dif))
        matrix_2 = vstack ([matrix_2_1, matrix_2_2])

        matrix_3 = sp.lil_matrix ((w, w))

        matrix_3.setdiag (1, 0)
        matrix_3.setdiag (-1, -1)
        matrix_3[0, 0] = 0
        matrix_3 = block_diag ([matrix_3] * h)

        matrix_4 = vstack ([matrix_2_2, matrix_2_1])

        # matrix = vstack ([matrix_1, matrix_2, matrix_3, matrix_4])

        A = vstack ([matrix_1, matrix_2, matrix_3, matrix_4])
        # matrix_1 = matrix_1.toarray ()
        # matrix_2 = matrix_2.toarray ()
        # matrix = matrix.toarray ()
        # 2
        front_b = np.zeros (A.shape[0], dtype=np.float64)
        back_b = np.zeros (A.shape[0], dtype=np.float64)

        ## 3. set normal
        front_nx = normal[:, :, 0].ravel ()
        front_ny = normal[:, :, 1].ravel ()
        front_nz = normal[:, :, 2].ravel ()

        back_nx = normal[:, :, 3].ravel ()
        back_ny = normal[:, :, 4].ravel ()
        back_nz = normal[:, :, 5].ravel ()

        ## 4. fill b
        ##  --> 0~nx.shape[0] is for v1
        ##  --> .... v2, v3, v4
        # front
        front_b[0:front_nx.shape[0]] = -front_nx / (front_nz + 1e-8)
        front_b[front_nx.shape[0]:2 * front_nx.shape[0]] = -front_ny / (front_nz + 1e-8)
        front_b[2 * front_nx.shape[0]:3 * front_nx.shape[0]] = -front_nx / (front_nz + 1e-8)
        front_b[3 * front_nx.shape[0]:front_b.shape[0]] = -front_ny / (front_nz + 1e-8)
        # back
        back_b[0:back_nx.shape[0]] = -back_nx / (back_nz + 1e-8)
        back_b[back_nx.shape[0]:2 * back_nx.shape[0]] = -back_ny / (back_nz + 1e-8)
        back_b[2 * back_nx.shape[0]:3 * back_nx.shape[0]] = -back_nx / (back_nz + 1e-8)
        back_b[3 * back_nx.shape[0]:back_b.shape[0]] = -back_ny / (back_nz + 1e-8)

        ## 5. solve Ax = b
        # front
        front_AtA = A.transpose ().dot (A)
        front_Atb = A.transpose ().dot (front_b)
        front_x, front_info = sp.linalg.cg (front_AtA, front_Atb)
        # back
        back_AtA = A.transpose ().dot (A)
        back_Atb = A.transpose ().dot (back_b)
        back_x, back_info = sp.linalg.cg (back_AtA, back_Atb)
        ## 6. create output matrix
        # front
        front_depth = front_x.reshape (mask.shape)
        front_depth -= np.min (front_depth)
        front_depth[mask == 0] = 0.0
        # back
        back_depth = back_x.reshape (mask.shape)
        back_depth -= np.min (back_depth)
        back_depth[mask == 0] = 0.0

        return front_depth, back_depth

    def save2npy(self):
        np.save(os.path.join(self.out_path,'depth_front.npy'),self.front_depth)
        np.save (os.path.join(self.out_path,'depth_back.npy'), self.back_depth)

    def save2img(self):
        def depth_2_img(depth):
            MAX = np.max (depth)
            depth_temp = np.where (depth == 0, depth + MAX, depth)
            MIN = np.min (depth_temp)
            depth_normalization = (1.0 - ((depth - MIN) / (MAX - MIN))) * 255

            return depth_normalization.astype (int)

        cv2.imwrite (os.path.join(self.out_path,'front_depth_img.png'), depth_2_img(self.front_depth))
        cv2.imwrite (os.path.join(self.out_path,'back_depth_img.png'), depth_2_img (self.back_depth))

    def comp_depth_4edge_front_back(self,mask, normal):
        '''
        "arbitrary point p(x, y, Z(x, y)) and Np = (nx, ny, nz)"

        v1 = (x+1, y, Z(x+1, y)) - p
           = (1, 0, Z(x+1, y) - Z(x, y))

        Then, dot(Np, v1) == 0 #right
        0 = Np * v1
          = (nx, ny, nz) * (1, 0, Z(x+1,y) - Z(x, y))
          = nx + nz(Z(x+1,y) - Z(x, y))

        --> Z(x+1,y) - Z(x, y) = -nx/nz = p

        Also dot(Np, v2) is same #up #left #bottom
        --> Z(x,y+1) - Z(x, y) = -ny/nz = q


        Finally, apply least square to find Z(x, y).
        A: round matrix
        x: matrix of Z(x, y)
        b: matrix of p and q
        A*x = b
        (--> 计算上下左右四个方向???)

        '''

        ## 1. prepare matrix for least square

        A = sp.lil_matrix ((mask.size * 4, mask.size))
        front_b = np.zeros (A.shape[0], dtype=np.float64)
        back_b = np.zeros (A.shape[0], dtype=np.float64)
        ## 2. set normal
        front_nx = normal[:, :, 0].ravel ()
        front_ny = normal[:, :, 1].ravel ()
        front_nz = normal[:, :, 2].ravel ()

        back_nx = normal[:, :, 3].ravel ()
        back_ny = normal[:, :, 4].ravel ()
        back_nz = normal[:, :, 5].ravel ()

        ## 3. fill b
        ##  --> 0~nx.shape[0] is for v1
        ##  --> .... v2, v3, v4
        #front
        front_b[0:front_nx.shape[0]] = -front_nx / (front_nz + 1e-8)
        front_b[front_nx.shape[0]:2 * front_nx.shape[0]] = -front_ny / (front_nz + 1e-8)
        front_b[2 * front_nx.shape[0]:3 * front_nx.shape[0]] = -front_nx / (front_nz + 1e-8)
        front_b[3 * front_nx.shape[0]:front_b.shape[0]] = -front_ny / (front_nz + 1e-8)
        #back
        back_b[0:back_nx.shape[0]] = -back_nx / (back_nz + 1e-8)
        back_b[back_nx.shape[0]:2 * back_nx.shape[0]] = -back_ny / (back_nz + 1e-8)
        back_b[2 * back_nx.shape[0]:3 * back_nx.shape[0]] = -back_nx / (back_nz + 1e-8)
        back_b[3 * back_nx.shape[0]:back_b.shape[0]] = -back_ny / (back_nz + 1e-8)

        ## 4. fill A
        dif = mask.size
        w = mask.shape[1]
        h = mask.shape[0]
        for i in range (mask.shape[0]):
            # progress_bar (i, mask.shape[0] - 1)

            for j in range (mask.shape[1]):

                ## current pixel om matrix
                pixel = (i * w) + j

                ## for v1(right)
                if j != w - 1:
                    A[pixel, pixel] = -1
                    A[pixel, pixel + 1] = 1

                ## for v2(up)
                if i != h - 1:
                    A[pixel + dif, pixel] = -1
                    A[pixel + dif, pixel + w] = 1

                ## for v3(left)
                if j != 0:
                    A[pixel + (2 * dif), pixel] = 1
                    A[pixel + (2 * dif), pixel - 1] = -1

                ## for v4(bottom)
                if i != 0:
                    A[pixel + (3 * dif), pixel] = 1
                    A[pixel + (3 * dif), pixel - w] = -1

        ## 5. solve Ax = b
        #front
        front_AtA = A.transpose ().dot (A)
        front_Atb = A.transpose ().dot (front_b)
        front_x, front_info = sp.linalg.cg (front_AtA, front_Atb)
        #back
        back_AtA = A.transpose ().dot (A)
        back_Atb = A.transpose ().dot (back_b)
        back_x, back_info = sp.linalg.cg (back_AtA, back_Atb)
        ## 6. create output matrix
        #front
        front_depth = front_x.reshape (mask.shape)
        front_depth -= np.min (front_depth)
        front_depth[mask == 0] = 0.0
        #back
        back_depth = back_x.reshape (mask.shape)
        back_depth -= np.min (back_depth)
        back_depth[mask == 0] = 0.0

        return front_depth,back_depth

    def comp_depth(self,mask, normal):
        '''
        "arbitrary point p(x, y, Z(x, y)) and Np = (nx, ny, nz)"

        v1 = (x+1, y, Z(x+1, y)) - p
           = (1, 0, Z(x+1, y) - Z(x, y))

        Then, dot(Np, v1) == 0 #right
        0 = Np * v1
          = (nx, ny, nz) * (1, 0, Z(x+1,y) - Z(x, y))
          = nx + nz(Z(x+1,y) - Z(x, y))

        --> Z(x+1,y) - Z(x, y) = -nx/nz = p

        Also dot(Np, v2) is same #up
        --> Z(x,y+1) - Z(x, y) = -ny/nz = q


        Finally, apply least square to find Z(x, y).
        A: round matrix
        x: matrix of Z(x, y)
        b: matrix of p and q

        A*x = b


        (--> might be left bottom as well???)

        '''

        ## 1. prepare matrix for least square
        print ("prepare matrix for least square: Ax = b")
        A = sp.lil_matrix ((mask.size * 2, mask.size))
        b = np.zeros (A.shape[0], dtype=np.float32)

        ## 2. set normal
        nx = normal[:, :, 0].ravel ()
        ny = normal[:, :, 1].ravel ()
        nz = normal[:, :, 2].ravel ()

        ## 3. fill b
        ##  --> 0~nx.shape[0] is for v1
        ##  --> .... v2, v3, v4
        b[0:nx.shape[0]] = -nx / (nz + 1e-8)
        b[nx.shape[0]:b.shape[0]] = -ny / (nz + 1e-8)

        ## 4. fill A
        dif = mask.size
        w = mask.shape[1]
        h = mask.shape[0]
        matrix_1 = sp.lil_matrix ((w, w))

        matrix_1.setdiag (-1, 0)
        matrix_1.setdiag (1, 1)
        matrix_1[-1, -1] = 0
        matrix_1 = block_diag ([matrix_1] * h)

        matrix_2_1 = sp.lil_matrix ((dif - w, dif))
        matrix_2_1.setdiag (-1, 0)
        matrix_2_1.setdiag (1, w)
        matrix_2_2 = sp.lil_matrix ((w, dif))
        matrix_2 = vstack ([matrix_2_1, matrix_2_2])

        matrix = vstack ([matrix_1, matrix_2])
        matrix_1 = matrix_1.toarray ()
        matrix_2 = matrix_2.toarray ()
        matrix = matrix.toarray ()
        for i in range (mask.shape[0]):
            # progress_bar (i, mask.shape[0] - 1)

            for j in range (mask.shape[1]):

                ## current pixel om matrix
                pixel = (i * w) + j

                ## for v1(right)
                if j != w - 1:
                    A[pixel, pixel] = -1
                    A[pixel, pixel + 1] = 1

                ## for v2(up)
                if i != h - 1:
                    A[pixel + dif, pixel] = -1
                    A[pixel + dif, pixel + w] = 1

        ## 5. solve Ax = b
        print ("\nsoloving Ax = b ...")
        AtA = A.transpose ().dot (A)
        Atb = A.transpose ().dot (b)
        x, info = sp.linalg.cg (AtA, Atb)

        ## 6. create output matrix
        depth = x.reshape (mask.shape)
        depth -= np.min (depth)
        depth[mask == 0] = 0.0

        return depth

    def comp_depth_4edge(self,mask, normal):
        '''
        "arbitrary point p(x, y, Z(x, y)) and Np = (nx, ny, nz)"

        v1 = (x+1, y, Z(x+1, y)) - p
           = (1, 0, Z(x+1, y) - Z(x, y))

        Then, dot(Np, v1) == 0 #right
        0 = Np * v1
          = (nx, ny, nz) * (1, 0, Z(x+1,y) - Z(x, y))
          = nx + nz(Z(x+1,y) - Z(x, y))

        --> Z(x+1,y) - Z(x, y) = -nx/nz = p

        Also dot(Np, v2) is same #up #left #bottom
        --> Z(x,y+1) - Z(x, y) = -ny/nz = q


        Finally, apply least square to find Z(x, y).
        A: round matrix
        x: matrix of Z(x, y)
        b: matrix of p and q
        A*x = b
        (--> 计算上下左右四个方向???)

        '''

        ## 1. prepare matrix for least square
        print ("prepare matrix for least square: Ax = b")
        A = sp.lil_matrix ((mask.size * 4, mask.size))
        b = np.zeros (A.shape[0], dtype=np.float64)

        ## 2. set normal
        nx = normal[:, :, 0].ravel ()
        ny = normal[:, :, 1].ravel ()
        nz = normal[:, :, 2].ravel ()

        ## 3. fill b
        ##  --> 0~nx.shape[0] is for v1
        ##  --> .... v2, v3, v4
        b[0:nx.shape[0]] = -nx / (nz + 1e-8)
        b[nx.shape[0]:2 * nx.shape[0]] = -ny / (nz + 1e-8)
        b[2 * nx.shape[0]:3 * nx.shape[0]] = -nx / (nz + 1e-8)
        b[3 * nx.shape[0]:b.shape[0]] = -ny / (nz + 1e-8)

        ## 4. fill A
        dif = mask.size
        w = mask.shape[1]
        h = mask.shape[0]
        for i in range (mask.shape[0]):
            # progress_bar (i, mask.shape[0] - 1)

            for j in range (mask.shape[1]):

                ## current pixel om matrix
                pixel = (i * w) + j

                ## for v1(right)
                if j != w - 1:
                    A[pixel, pixel] = -1
                    A[pixel, pixel + 1] = 1

                ## for v2(up)
                if i != h - 1:
                    A[pixel + dif, pixel] = -1
                    A[pixel + dif, pixel + w] = 1

                ## for v3(left)
                if j != 0:
                    A[pixel + (2 * dif), pixel] = 1
                    A[pixel + (2 * dif), pixel - 1] = -1

                ## for v4(bottom)
                if i != 0:
                    A[pixel + (3 * dif), pixel] = 1
                    A[pixel + (3 * dif), pixel - w] = -1

        ## 5. solve Ax = b
        print ("\nsoloving Ax = b ...")
        AtA = A.transpose ().dot (A)
        Atb = A.transpose ().dot (b)
        x, info = sp.linalg.cg (AtA, Atb)

        ## 6. create output matrix
        depth = x.reshape (mask.shape)
        depth -= np.min (depth)
        depth[mask == 0] = 0.0

        return depth

    def deel_back_depth(self):
        mask = np.where (self.front_depth == 0 , 0, 255).astype ('uint8')
        contours, hierarchy = cv2.findContours (mask.T, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        bound = contours[0].reshape (contours[0].shape[0], 2)
        mask = cv2.morphologyEx (mask, cv2.MORPH_CLOSE, np.ones ((int (3), int (3)), np.uint8))  # 闭运算
        front_bound_depth = np.array ([self.front_depth[i, j] for i, j in bound])
        back_bound_depth = np.array ([self.back_depth[i, j] for i, j in bound])
        front_bound_depth_mean = np.mean (front_bound_depth)
        back_bound_depth_mean = np.mean (back_bound_depth)
        front_depth_mean = np.mean (self.front_depth)
        bound_difference = back_bound_depth_mean - front_bound_depth_mean
        front_difference = front_bound_depth_mean - front_depth_mean
        self.back_depth = self.back_depth - bound_difference + front_difference /3.0
        self.back_depth=self.back_depth*mask.astype ('bool')