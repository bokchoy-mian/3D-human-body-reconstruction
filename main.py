#!/usr/bin/python
#-- coding:utf8 --
import os

import pickle
import time
import cv2
import numpy as np

import PIL.Image as pil_img
# from lib.gen_smplh import gen_smplh
from models.smplh_np import SMPLHModel
from models.smpl_np import SMPLModel

from utils import render_model
from utils.J_render import JRender

from lib.Warp import Wrap
from lib.Normal2Depth import Normal2Depth
from lib.Depth2Mesh import Depth2Mesh
from lib.mesh2smpl_model import RecoverModel
from lib.Depth2Mesh_Bspline import Depth2Mesh_Bspline

# from lib import model2video

os.chdir (os.path.dirname (__file__))

def main(path):
    # path=os.getcwd ()
    out_path = path
    smpl_model_path = 'models/model/smpl/SMPL_MALE.pkl'
    rgb_img_path =  os.path.join (out_path, 'front_rgb.png')
    rgb_mask_path=  os.path.join (out_path, 'mask.png')
    front_color_path = os.path.join (out_path, 'front_rgb.png')
    back_color_path = os.path.join (out_path, 'back_rgb.png')
    # openpose_path=os.path.join (out_path, '0_keypoints.json')

    front_color = cv2.imread (front_color_path)
    front_color = cv2.cvtColor (front_color, cv2.COLOR_BGR2RGB)
    back_color = cv2.imread (back_color_path)
    back_color = cv2.cvtColor (back_color, cv2.COLOR_BGR2RGB)

    smplh_model_path='models/model/smplh/SMPLH_MALE.pkl'
    rgb_img = cv2.imread (rgb_img_path).astype (np.float32)[:, :, ::-1] / 255.0
    rgb_mask = cv2.imread (rgb_mask_path , cv2.IMREAD_GRAYSCALE)
    # rgb_mask = cv2.morphologyEx (rgb_mask, cv2.MORPH_CLOSE, np.ones ((int (5), int (5)), np.uint8))#填充孔洞

    H, W, _ = rgb_img.shape
    # smplh_result,_=gen_smplh(rgb_img_path,openpose_path,out_path)
    with open (os.path.join (out_path, 'smplh.pkl'), 'rb') as f:
        smplh_result=pickle.load (f, encoding='iso-8859-1')

    camera_rotation = smplh_result['camera_rotation'].astype ('float64')
    camera_transl = smplh_result['camera_translation'].astype ('float64')
    camera_center = smplh_result['camera_center'].astype ('float64')


    pose = smplh_result['spmlh_pose'].reshape ([-1, 3]).astype ('float64')#(24,3)
    shape = smplh_result['spmlh_shape'].astype ('float64')

    smplh_model = SMPLHModel (smplh_model_path)
    smplh_model.set_params (beta=shape, pose=pose)

    smpl_model = SMPLModel (smpl_model_path)
    smpl_model.set_params (pose[:24, :], shape)

    #得到关节点在2d图像的像素位置
    camera_intrinsic = {
        "R": camera_rotation,# R，旋转矩阵
        "T": camera_transl,# t，平移向量
        "f": [5000., 5000.], # 焦距，f/dx, f/dy
        "c": camera_center# principal point，主点，主轴与像平面的交点
    }
    J_smpl_3d = smpl_model.gen_J_3d ()
    Jrender = JRender (camera_intrinsic)
    J_2d = (Jrender (J_smpl_3d)).astype ('int16')
    Jrender.save2img (front_color, out_path + '/J_or.png')

    ##render smplh
    render=render_model.Render(smplh_model,rgb_img,smpl_model.weights,camera_center,camera_transl,camera_rotation)
    front_normals_img=render.front_normals_renderer()
    back_normals_img=render.back_normals_renderer()
    smplh_weigth=render.weigth_render()#[h,w,24]
    render.save_normal2npy(os.path.join(out_path,'front_normal.npy'),front_normals_img)
    render.save_normal2npy(os.path.join(out_path,'back_normal.npy'),back_normals_img)
    render.save_normal2npy(os.path.join(out_path,'smplh_weigth.npy'),smplh_weigth)
    render.save_normal2img (os.path.join (out_path, 'front_normal.png'), front_normals_img)
    render.save_normal2img (os.path.join (out_path, 'back_normal.png'), back_normals_img)
    render.save_weigth2img(os.path.join (out_path, 'smplh_weigth.png'), smplh_weigth)
    smplh_mask = np.where (front_normals_img[:,:,0] == 1, 0, 255).astype('uint8')
    cv2.imwrite (os.path.join (out_path, 'smplh_mask.png'), smplh_mask)

    smplh_value=np.concatenate((front_normals_img,back_normals_img,smplh_weigth),axis=2)
    render.save_normal2npy (os.path.join(out_path , 'smplh_value.npy'), smplh_value)
    # smplh_value = np.load (os.path.join(out_path , 'smplh_value.npy'))

    #变换SMPL的value
    warp=Wrap(rgb_mask,smplh_value,out_path)
    warp_smplh_value=warp()
    warp.save2img()
    warp.save2npy()
    # warp.show_img(warp_smplh_value)
    # warp_smplh_value=np.load(os.path.join(out_path ,'warp_and_filled.npy'))

    #根据法向图生成深度图
    normal2depth=Normal2Depth(rgb_mask,warp_smplh_value[:,:,0:6],out_path)
    front_depth, back_depth=normal2depth()
    normal2depth.save2npy()
    normal2depth.save2img()
    # front_depth=np.load(out_path+'depth_front.npy')
    # back_depth = np.load (out_path + 'depth_back.npy')

    #缝合正反面
    gen_mesh=Depth2Mesh_Bspline(front_depth,front_color,back_depth,back_color,warp_smplh_value[:,:,6:],J_2d,out_path)
    points,faces,J_3d=gen_mesh.stich_mesh()
    # gen_mesh.save2npy(os.path.join(out_path,'points.npy'),points)
    # gen_mesh.save2npy(os.path.join(out_path,'faces.npy'),faces)
    # gen_mesh.save2npy (os.path.join(out_path,'J_3d.npy'), J_3d)

    # points=np.load(os.path.join(out_path,'points.npy'))
    # faces =np.load(os.path.join(out_path,'faces.npy'))
    # J_3d=np.load(os.path.join(out_path,'J_3d.npy'))
    verts=points[:,0:3]
    color=points[:,3:6]
    weigth=points[:,6:]

    #将恢复的模型变换到T_pose 标准模型
    recover_model=RecoverModel(smpl_model,verts,color,faces,weigth,pose,shape,J_3d)

    #渲染weigth to img
    # recover_weigth = render.recover_weigth_render (recover_model.or_verts, faces, recover_model.weigths)
    # render.save_weigth2img (os.path.join (out_path, 'recover_weigth.png'), recover_weigth)

    # recover_model.output_or_mesh (os.path.join (out_path, 'or_stich.obj'))
    # recover_model.output_T_posemesh (os.path.join (out_path, 're_Tpose_stich.obj'))
    # recover_model.save_model (os.path.join (out_path, 'or_recover.pkl'))
    # recover_model.replace_hands()
    # recover_model.output_T_posemesh (os.path.join (out_path, 'replace_hand_Tpose_stich.obj'))
    # recover_model.save_model(os.path.join(out_path,'replace_hands_recover.pkl'))


if __name__ == '__main__':
    main('data/test_512/test_12_smpl')
    # for root, dirs, files in os.walk ('data/T_pose_test'):
    #     for dir in dirs:
    #         file_path = os.path.join (root, dir)
    #         main( file_path)

    # out_path = 'data/reconstruct_output/'
    # rgb_mask_path = 'data/images/rgb_mask.png'
    # rgb_mask = cv2.imread (rgb_mask_path, cv2.IMREAD_GRAYSCALE)

    # front_depth=np.load('data/reconstruct_output/depth_front_smplh.npy')
    # front_depth=cv2.medianBlur (front_depth.astype('float32'), 3)
    # back_depth = np.load ('data/reconstruct_output/depth_back_smplh.npy')
    # front_depth = cv2.medianBlur (front_depth.astype('float32'), 3)
    # stitch_mask = cv2.erode (rgb_mask, np.ones ((5, 5), np.uint8))
    # mask = cv2.erode (rgb_mask, np.ones ((3, 3), np.uint8))
    # normal_front= np.load ('data/reconstruct_output/warp_front_normal_filled.npy')
    # normal_back = np.load ('data/reconstruct_output/warp_back_normal_filled.npy')
    # normal_front = cv2.imread ('data/reconstruct_output/filled_front.png', -1)
    # normal_back = cv2.imread ('data/reconstruct_output/filled_back.png', -1)
    # cv2.imshow('_',smplh_mask*255)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # smplh_mask_back = np.where (back_normals_img[:, :, 0] == 1, 0, 255).astype ('uint8')
    # cv2.imwrite (os.path.join (out_path, 'smpl_mask_diff.png'), smplh_mask_back-smplh_mask)
    # cv2.imwrite (os.path.join(out_path,'smpl_mask.png'), smplh_mask)

    # mask = cv2.erode (rgb_mask, np.ones ((3, 3), np.uint8))
    # normal_front = (normal_front[:,:,::-1] / 255.0) * 2.0 - 1.0
    # normal_front = (normal_front) * 2.0 - 1.0
    # normal_front[:, :, 2] = -abs(normal_front[:, :, 2])

    #
    # normal_front[mask == 0] = [0, 0, 0]
    # front_depth = comp_depth_4edge (mask, normal_front)
    # np.save (out_path + 'depth_front_smplh.npy', front_depth)
    # for back
    # normal_back = (normal_back[:,:,::-1] / 255.0) * 2.0 - 1.0
    # normal_back = (normal_back) * 2.0 - 1.0
    # normal_back[:,:,2] = abs(normal_back[:,:,2])
    # normal_back[mask == 0] = [0, 0, 0]
    # back_depth = comp_depth_4edge (mask, normal_back)
    # np.save (out_path + 'depth_back_smplh.npy', back_depth)
    # startime = time.time ()
    # vertex_front, triangle_front = Depth2VerTri (front_depth, mask)
    # writeobj (out_path + 'recover_front_smplh.obj', vertex_front, triangle_front)
    # vertex_back, triangle_back = Depth2VerTri (back_depth, mask)
    # writeobj (out_path + 'recover_back_smplh.obj', vertex_back, triangle_back)
    # print ('gen_obj--time:  %d' % (time.time () - startime))
    # #
    # stitch_mask =cv2.erode (rgb_mask, np.ones ((3, 3), np.uint8))
    # # rgb_mask=np.int8 (rgb_mask)
    # stitch_mesh (stitch_mask, front_depth, back_depth, out_path + 'stich_smplh.obj')
    # front_depth = comp_depth_4edge (rgb_mask,front_normal)
    # back_depth = comp_depth_4edge (rgb_mask, back_normal)

    # stitch_mesh (stitch_mask, front_depth, back_depth, out_path + 'stich_medianBlur_5.obj')