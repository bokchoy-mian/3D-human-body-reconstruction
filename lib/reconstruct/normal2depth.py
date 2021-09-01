import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import cv2
# import obj_functions as ob
def progress_bar(n, N):

  '''
  print current progress
  '''

  step = 2
  percent = float(n) / float(N) * 100

  ## convert percent to bar
  current = "#" * int(percent//step)
  remain = " " * int(100/step-int(percent//step))
  bar = "|{}{}|".format(current, remain)
  print("\r{}:{:3.0f}[%]".format(bar, percent), end="", flush=True)

def comp_depth(mask, normal):
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
    for i in range (mask.shape[0]):
        progress_bar(i, mask.shape[0] - 1)

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

def comp_depth_4edge(mask, normal):
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
        progress_bar (i, mask.shape[0] - 1)

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

def pre_progress(normal,mask):

    normal_tran=(normal/255.0)*2.0-1.0
    normal_tran[mask == 0] = [0, 0, 0]
    # imtmp_R, imtmp_G, imtmp_B = cv2.split (normal_tran)
    # normal_tran[:,:,2]=np.where(normal_tran[:,:,2]<0,-normal_tran[:,:,2],normal_tran[:,:,2])
    # imtmp_R, imtmp_G, imtmp_B = cv2.split (normal_tran)
    return normal_tran

def pre_progress_normalimg(normal_img_path,mask):
    normal=cv2.imread(normal_img_path)[:,:,::-1]
    normal_tran=(normal/255.0)*2.0-1.0
    normal_tran[mask == 0] = [0, 0, 0]
    # imtmp_R, imtmp_G, imtmp_B = cv2.split (normal_tran)
    # normal_tran[:,:,2]=np.where(normal_tran[:,:,2]<0,-normal_tran[:,:,2],normal_tran[:,:,2])
    # imtmp_R, imtmp_G, imtmp_B = cv2.split (normal_tran)
    return normal_tran

def mask2tiny(mask_path, window):

  # mask
  mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
  eroded = cv2.erode(mask, np.ones((int(window), int(window)), np.uint8)) # 0~1

  return eroded

def depth_2_img(depth,outpath):
    MAX = np.max (depth)
    depth_temp=np.where(depth==0,depth+MAX,depth)
    MIN=np.min(depth_temp)
    depth_normalization=(1.0 - ((depth-MIN) / (MAX-MIN))) * 255
    cv2.imwrite (outpath, depth_normalization)
    return depth_normalization.astype(int)


if __name__ == '__main__':
    # mask_front = mask2tiny ('data/baoluo_1/rgb_mask.png', 3)
    # mask_back=cv2.flip(mask_front,1)
    dr = 'data/baoluo_divide/'
    mask_front=mask2tiny(dr+'rgb_mask.png',3)
    mask_back = mask2tiny(dr+'rgb_mask.png',3)
    normal_front=pre_progress_normalimg(dr+'filled_front.png', mask_front)
    normal_back = pre_progress_normalimg (dr+'filled_back.png', mask_back)

    depth_front=comp_depth_4edge(mask_front,normal_front)
    depth_back = comp_depth_4edge (mask_back, normal_back)

    depth_front_npy=np.load(dr+'depth_front.npy')
    depth_back_npy=np.load(dr+'depth_back.npy')
    depth_2_img(depth_front_npy,dr+'warp_depth_front.png')
    depth_2_img(depth_back_npy,dr+'warp_depth_back.png')
    # np.save('data/baoluo_divide/smpl_initdepth_front.npy',depth_front)
    # np.save ('data/baoluo_divide/smpl_initdepth_back.npy', depth_back)
    # depth=np.load('smpl_depth_back.npy')
    # m, n = depth.shape
    # index = int (depth.argmax ())
    # x = int (index / n)
    # y = index % n
    # print (x, y)
    # vertex_front, triangle_front =ob.Depth2VerTri(depth_front,mask_front)
    # vertex_back, triangle_back = ob.Depth2VerTri (depth_back, mask_back)
    # vertex_z=vertex[:,2]
    # index_z_min=vertex_z.argmin()
    # index_z_max=vertex_z.argmax()
    # ob.writeobj (dr+'front.obj', vertex_front, triangle_front)
    # ob.writeobj (dr+'back.obj', vertex_back, triangle_back)
    # MAX=np.max(depth)
    # depth_temp=np.where(depth==0,depth+MAX,depth)
    # MIN=np.min(depth_temp)
    # depth_normalization=(1.0 - ((depth-MIN) / (MAX-MIN))) * 255
    # cv2.imwrite ('data/baoluo/smpl_normal2depth_back.png', depth_normalization)
    # depth_map=(1.0 - (depth / np.max (depth))) * 255
    # cv2.imwrite ('data/baoluo/smpl_normal_depth_back.png', depth_map)
    # cv2.imshow('1',depth_map)
    # cv2.waitKey()
    # cv2.destroyAllWindows()