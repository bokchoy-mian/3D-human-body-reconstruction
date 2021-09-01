#!/usr/bin/python
#-- coding:utf8 --
#!/usr/bin/python
#-- coding:utf8 --
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import os.path as osp
# import time
import yaml
import torch

import smplx
try:
    import cPickle as pickle
except ImportError:
    import pickle
from lib.Gen_SMPLH.util import JointMapper
from lib.Gen_SMPLH.smpl_config  import parse_config
from lib.Gen_SMPLH.data_parser import OpenPose_SMPlH
from lib.Gen_SMPLH.fit_single_frame import fit_single_frame

from lib.Gen_SMPLH.camera import create_camera
from lib.Gen_SMPLH.prior import create_prior

torch.backends.cudnn.enabled = False

# from gen_smpl_smplifyx import parse_config

def gen_smplh(img_path=None,keyp_path=None,out_path=None):
    # os.chdir (os.path.dirname (__file__))
    args = parse_config()
    if img_path:
        args['input_img_path']=img_path
    if keyp_path:
        args['input_keyp_path']=keyp_path
    if out_path:
        args['output_folder']=out_path

    output_folder = args.pop ('output_folder')
    output_folder = osp.expandvars (output_folder)

    if not osp.exists (output_folder):
        os.makedirs (output_folder)

    #保存设置
    conf_fn = osp.join (output_folder, 'conf.yaml')
    with open (conf_fn, 'w') as conf_file:
        yaml.dump (args, conf_file)

    use_cuda = args.get ('use_cuda', True)
    if use_cuda and not torch.cuda.is_available ():
        print ('CUDA is not available, exiting!')
        sys.exit (-1)
    input_img_path = args.pop ('input_img_path')
    input_keyp_path = args.pop ('input_keyp_path')
    data_obj =  OpenPose_SMPlH(input_img_path,input_keyp_path, **args)

    input_gender = args.pop ('gender', 'neutral')

    float_dtype = args.get ('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError ('Unknown float type {}, exiting!'.format (float_dtype))

    joint_mapper = JointMapper (data_obj.get_model2data ())

    model_params = dict (model_path=args.get ('model_folder'),
                         joint_mapper=joint_mapper,
                         create_global_orient=True,
                         create_body_pose=not args.get ('use_vposer'),
                         create_betas=True,
                         create_left_hand_pose=True,
                         create_right_hand_pose=True,
                         create_expression=True,
                         create_jaw_pose=True,
                         create_leye_pose=True,
                         create_reye_pose=True,
                         create_transl=False,
                         dtype=dtype,
                         **args)

    smplh_model = smplx.create (gender=input_gender, **model_params)

    # Create the camera object
    focal_length = args.get ('focal_length')
    camera = create_camera (focal_length_x=focal_length,
                            focal_length_y=focal_length,
                            dtype=dtype,
                            **args)

    if hasattr (camera, 'rotation'):
        camera.rotation.requires_grad = False

    use_hands = args.get ('use_hands', True)
    jaw_prior, expr_prior = None, None
    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy ()
        lhand_args['num_gaussians'] = args.get ('num_pca_comps')
        left_hand_prior = create_prior (
            prior_type=args.get ('left_hand_prior_type'),
            dtype=dtype,
            use_left_hand=True,
            **lhand_args)

        rhand_args = args.copy ()
        rhand_args['num_gaussians'] = args.get ('num_pca_comps')
        right_hand_prior = create_prior (
            prior_type=args.get ('right_hand_prior_type'),
            dtype=dtype,
            use_right_hand=True,
            **rhand_args)
    body_pose_prior = create_prior (prior_type=args.get ('body_prior_type'), dtype=dtype, **args)
    shape_prior = create_prior (prior_type=args.get ('shape_prior_type', 'l2'),dtype=dtype, **args)
    angle_prior = create_prior (prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available ():
        device = torch.device ('cuda')
        camera = camera.to (device=device)
        smplh_model = smplh_model.to (device=device)

        body_pose_prior = body_pose_prior.to (device=device)
        angle_prior = angle_prior.to (device=device)
        shape_prior = shape_prior.to (device=device)

        if use_hands:
            left_hand_prior = left_hand_prior.to (device=device)
            right_hand_prior = right_hand_prior.to (device=device)
    else:
        device = torch.device ('cpu')

    # A weight for every joint of the model
    joint_weights = data_obj.get_joint_weights ().to (device=device,dtype=dtype)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_ (dim=0)

############fitting#########
    data=data_obj.read_item()
    img = data['img']
    img_name = data['img_name']
    keypoints = data['keypoints']
    print ('Processing: {}'.format (data['img_path']))


    out_pkl_path = osp.join (output_folder,'{}.pkl'.format ('pre_smplh'))
    smplh_pkl_path = osp.join (output_folder, 'smplh.pkl')
    out_obj_path = osp.join (output_folder,'{}.obj'.format ('smplh'))
    out_rendimg_path = osp.join (output_folder, '{}_rend.png'.format ('smplh2rgb'))

    result,smplh_output=fit_single_frame (img, keypoints[[0]],
                          body_model=smplh_model,
                          camera=camera,
                          joint_weights=joint_weights,
                          dtype=dtype,
                          out_pkl_path =out_pkl_path ,
                          out_obj_path =out_obj_path ,
                          out_rendimg_path=out_rendimg_path,
                          shape_prior=shape_prior,
                          angle_prior=angle_prior,
                          body_pose_prior=body_pose_prior,
                          left_hand_prior=left_hand_prior,
                          right_hand_prior=right_hand_prior,
                          jaw_prior=jaw_prior, expr_prior=expr_prior,
                          **args)
    with open (smplh_pkl_path, 'wb') as result_file:
        pickle.dump (smplh_output, result_file, protocol=2)


    return smplh_output,result

def main(path):
    for root, dirs, files in os.walk (path, topdown=True):
        for dir in dirs:
            file_path = os.path.join (root, dir)
            rgb_img_path = os.path.join (file_path, 'front_rgb.png')
            openpose_path = os.path.join (file_path, '0_keypoints.json')
            smplh_result,_=gen_smplh(rgb_img_path,openpose_path,file_path)

if __name__ == "__main__":

    # main('../data/test_web_512/')
    smplh_output,result=gen_smplh('../data/test_IronMan/front_rgb.png', '../data/test_IronMan/0_keypoints.json', '../data/test_IronMan/')
    # print(smplh_output)

