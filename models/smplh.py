#!/usr/bin/python
#-- coding:utf8 --
import torch
import numpy as np
from smplx import SMPLH as _SMPLH
from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints
import os

import config
import constants

class SMPLH(_SMPLH):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPLH, self).__init__(*args, **kwargs)
        self.seg_index = {}
        self.verts_numpy=self.v_template.numpy()
        self.weigths =self.lbs_weights.numpy()
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPLH, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        self.verts_numpy = output.vertices[0].cpu ().numpy ()
        return output

    def load_index(self,flord='./models/seg_index_np/'):
        pathlist=os.listdir(flord)

        for path in pathlist:
            filename=os.path.basename(path)[:-4]
            index=np.load(flord+path)
            self.seg_index.update({filename:index})
        # temp_index=self.seg_index
        # print('1')
    def segsmpl2part(self):
        if not self.seg_index:
            self.load_index()
        namelist=self.seg_index.keys()
        self.verts_part={}
        for name in namelist:
            # path=out_path+name+'_verts.txt'
            # np.savetxt (path,self.verts_numpy[self.seg_index[name]], fmt='%.5f')
            self.verts_part.update({name:self.verts_numpy[self.seg_index[name]]})

    def gen_color_verts(self,verts=None):
        if verts==None:
            verts=self.verts_numpy
        color_np=np.load(config.VERTS_CORTS_PATH)
        verts_with_color=np.concatenate((verts,color_np),axis=1)
        self.verts_numpy=verts_with_color
        return verts_with_color

    def write_obj(self,outpath,face=None, verts=None,mesh_vt=None, mesh_vn=None, verbose=True):
        """

        :param self.verts_numpy:顶点坐标
        :param self.faces:面
        :param mesh_vt:贴图坐标
        :param mesh_vn:法线
        :param outpath:
        :param verbose:
        :return:
        """
        if verts is None:
            verts=self.verts_numpy
        if face is None:
            face=self.faces
        with open (outpath, 'w') as fp:
            for v in verts:
                if len (v) == 3:
                    fp.write ('v %f %f %f\n' % (v[0], v[1], v[2]))
                elif len (v) == 6:
                    fp.write ('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], v[3], v[4], v[5]))
            if mesh_vt is not None:
                for t in mesh_vt:
                    fp.write ('vt %f %f\n' % (t[0], t[1]))
            if mesh_vn is not None:
                for n in mesh_vn:
                    fp.write ('vn %f %f %f\n' % (n[0], n[1], n[2]))
            for f in face + 1:  # Faces are 1-based, not 0-based in obj files
                if f.shape == (1, 3) or f.shape == (3,) or f.shape == (3, 1):
                    fp.write ('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (f[0], f[0], f[0], f[1], f[1], f[1], f[2], f[2], f[2]))
                elif f.shape == (3, 2):
                    fp.write ('f %d/%d %d/%d %d/%d\n' % (f[0, 0], f[0, 1], f[1, 0], f[1, 1], f[2, 0], f[2, 1]))
                elif f.shape == (3, 3):
                    fp.write ('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (
                    f[0, 0], f[0, 1], f[0, 2], f[1, 0], f[1, 1], f[1, 2], f[2, 0], f[2, 1], f[2, 2]))
                else:
                    print ("strange faces shape!")

        if verbose:
            print ('mesh saved to: ', outpath)

    def divide_face(self, v):
        """smpl模型分离正反面

        :param f:
        :param v:
        :return:
        """
        f=self.faces
        N = f.shape[0]  # N face
        z_vec = [0, 0, 1]
        front_verts_index=[]
        back_verts_index=[]
        front_face = []
        back_face = []
        for i in range (N):
            if i == 0:
                print (f[i][0])
            v0 = v[f[i][0]]
            v1 = v[f[i][1]]
            v2 = v[f[i][2]]

            m = v1 - v0
            n = v2 - v1

            # normal
            # x = m[1] * n[2] - n[1] * m[2]
            # y = n[0] * m[2] - m[0] * n[2]
            z = m[0] * n[1] - n[0] * m[1]

            if z < 0:
                front_temp=[]
                for index in f[i]:
                    if index in front_verts_index:
                        front_temp.append(front_verts_index.index(index))
                    else:
                        front_verts_index.append(index)
                        front_temp.append (front_verts_index.index (index))

                front_face.append (front_temp)
            else:
                back_temp = []
                for index in f[i]:
                    if index in back_verts_index:
                        back_temp.append (back_verts_index.index (index))
                    else:
                        back_verts_index.append (index)
                        back_temp.append (back_verts_index.index (index))
                back_face.append (back_temp)

        front_face = np.array (front_face)
        back_face = np.array (back_face)
        front_verts=v[front_verts_index]
        back_verts=v[back_verts_index]
        return front_face,front_verts, back_face,back_verts

