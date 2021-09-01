#!/usr/bin/python
#-- coding:utf8 --
import numpy as np
import pickle
from lib import Replace_Hands
import trimesh
class SMPLModel():
    '''Simplified SMPL model. All pose-induced transformation is ignored. Also ignore beta.'''
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            params = pickle.load(f,encoding='iso-8859-1')

            self.J_regressor = params['J_regressor']
            self.weigths = params['weights']
            self.v_template = params['v_template']
            # self.v_template[:, 1] = -self.v_template[:, 1]
            # self.v_template[:,2] = -self.v_template[:,2]
            self.faces = params['f']
            self.kintree_table = params['kintree_table']

        id_to_col = {self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])}
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.trans_shape = [3]

        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = self.v_template
        self.J = None
        self.R = None

        # self.update()

    def set_params(self, pose=None, beta=None, trans=None):
        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        self.update()
        return self.verts

    def compute_R_G(self):
        self.J = self.J_regressor.dot(self.v_template)
        pose_cube = self.pose.reshape((-1, 1, 3))
        self.R = self.rodrigues(pose_cube)
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        G[0, :, :] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i, :, :] = G[self.parent[i], :, :].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i], ((self.J[i, :] - self.J[self.parent[i], :]).reshape([3, 1]))]
                    )
                )
            )
        return G

    def do_skinning(self, G):
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
                )
            )
        T = np.tensordot(self.weigths, G, axes=[[1], [0]])
        self.T_inverse=np.linalg.inv(T)
        rest_shape_h = np.hstack((self.v_template, np.ones([self.v_template.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.verts = v + self.trans.reshape([1, 3])

    def update(self):
        G = self.compute_R_G()
        self.do_skinning(G)

    def rodrigues(self, r):
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(np.float64).tiny)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        return np.vstack([x, np.array([[0.0, 0.0, 0.0, 1.0]])])

    def pack(self, x):
        return np.dstack([np.zeros([x.shape[0], 4, 3]), x])

    def gen_J_3d(self):
        return self.J_regressor.dot(self.verts)

    def output_mesh(self, path):
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def inverse(self):
        # temp=self.verts[:,:]
        # self.verts=self.v_template[:,:]
        # self.v_template=temp[:,:]
        # G = self.compute_R_G ()
        rest_shape_h = np.hstack ((self.verts-self.trans.reshape([1, 3]), np.ones ([self.verts.shape[0], 1])))
        v = np.matmul (self.T_inverse, rest_shape_h.reshape ([-1, 4, 1])).reshape ([-1, 4])[:, :3]
        self.verts = v

class RecoverModel():
    '''将重建后的模型与smpl模型绑定'''
    def __init__(self, smpl_model,verts,color,face,weigths,pose,shape,J_3d):
        """

        :param smpl_model_path:
        :param verts: 重建后得模型顶点
        :param face: 重建后的模型面
        :param weigths: 重建后模型的lbs权重
        :param pose: 原始的smpl姿势参数
        :param shape: 原始的smpl形态参数
        """
        self.ignor_J = [13, 14, 22, 23]

        self.smpl=smpl_model
        vertices_aligned, J_aligned = self.mesh_verts_align (self.smpl.verts, verts, self.smpl.J, J_3d)
        self.or_pose=self.smpl.gen_re_pose(J_aligned,pose[:24,:],shape)#将生成的模型的J拟合到标准模型上计算旋转矩阵
        self.or_shape=shape
        self.or_verts = vertices_aligned
        self.or_J =J_aligned

        self.color = color
        self.weigths = weigths/(weigths.sum(axis=1)[:,None])
        self.faces = face.astype(np.int)

        self.smpl.set_params(beta=self.or_shape) #形态要一致
        self.kintree_table = self.smpl.kintree_table
        self.parent = self.smpl.parent
        self.smpl_J=self.smpl.J
        self.smpl_v_template = self.smpl.verts
        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.trans_shape = [3]

        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.v_template= None #T_pose verts
        self.J = None #T_pose J

        self.verts = None
        self.R = None

        # self.to_rest_pose() # 姿态不变只是旋转，只用pose[0]
        self.to_T_pose()
        # self.global_rotation=np.linalg.inv(self.rodrigues(pose[0,:].reshape ((-1, 1, 3)))[0])
        # self.J = self.or_J.dot(self.global_rotation.T)#暂时不转
        # self.v_template=self.or_verts.dot(self.global_rotation.T)#暂时不转

        self.update()

    def to_T_pose(self):
        '''
        根据姿态参数，将smpl拟合到or_pose，从而计算反转矩阵，得到v_template与J
        :return:
        '''

        T_pose=self.or_pose

        self.smpl.set_params (T_pose, self.or_shape)# 将smpl拟合到相同的姿态下，从而得到T的逆
        # self.smpl.output_mesh('data/animate/mesh/smpl_Tpose.obj')
        G=self.smpl.compute_R_G()
        G = G - self.pack (
            np.matmul (
                G,
                np.hstack ([self.smpl.J, np.zeros ([24, 1])]).reshape ([24, 4, 1])
            )
        )
        T = np.tensordot (self.weigths, G, axes=[[1], [0]])
        self.T_inverse = np.linalg.inv (T)#反转矩阵
        # self.smpl.set_params (self.or_pose, self.or_shape, self.or_trans)#将smpl拟合到相同的姿态下，从而得到T的逆
        rest_shape_h = np.hstack ((self.or_verts , np.ones ([self.or_verts.shape[0], 1])))
        self.v_template=np.matmul (self.T_inverse, rest_shape_h.reshape ([-1, 4, 1])).reshape ([-1, 4])[:, :3]

        or_J = np.hstack ((self.or_J, np.ones ([self.or_J.shape[0], 1])))
        self.J = np.matmul (np.linalg.inv (G), or_J.reshape ([-1, 4, 1])).reshape ([-1, 4])[:, :3]

    def replace_hands(self):
        recover_points=np.concatenate((self.v_template,self.color,self.weigths),axis=1)
        recover_faces=self.faces
        recover_J=self.J
        smpl_color=np.ones(self.smpl.v_template.shape)*125
        smpl_points=np.concatenate((self.smpl_v_template,smpl_color,self.smpl.weights),axis=1)
        smpl_faces=self.smpl.faces
        smpl_J=self.smpl_J
        replace=Replace_Hands.Replace_Hands(recover_points,recover_faces, recover_J, smpl_points,smpl_faces,smpl_J)
        full_points, full_faces,J_3d=replace.replace()
        self.faces=full_faces
        self.v_template=full_points[:,:3]
        self.color=full_points[:,3:6]
        self.weigths=full_points[:,6:]
        self.J=J_3d
        return full_points, full_faces, J_3d

    def mesh_verts_align(self, smpl_verts, verts, smpl_J, J_3d, eps=1e-8):
        """
        this function aligns verts to smpl_verts
        smpl_verts, verts are of shape (?,3)
        """
        # finding bounding boxes
        bbox_1_x_min, bbox_1_x_max = np.min (smpl_verts[:, 0]), np.max (smpl_verts[:, 0])
        bbox_1_y_min, bbox_1_y_max = np.min (smpl_verts[:, 1]), np.max (smpl_verts[:, 1])
        # bbox_1_z_min, bbox_1_z_max = np.min (smpl_verts[:, 2]), np.max (smpl_verts[:, 2])
        # H1 = bbox_1_z_max - bbox_1_z_min
        W1 = bbox_1_y_max - bbox_1_y_min
        D1 = bbox_1_x_max - bbox_1_x_min

        bbox_2_x_min, bbox_2_x_max = np.min (verts[:, 0]), np.max (verts[:, 0])
        bbox_2_y_min, bbox_2_y_max = np.min (verts[:, 1]), np.max (verts[:, 1])
        # bbox_2_z_min, bbox_2_z_max = np.min (verts[:, 2]), np.max (verts[:, 2])
        # H2 = bbox_2_z_max - bbox_2_z_min
        W2 = bbox_2_y_max - bbox_2_y_min
        D2 = bbox_2_x_max - bbox_2_x_min

        # get_centers
        # center_1 = 0.5 * np.array ([(bbox_1_x_min + bbox_1_x_max),
        #                             (bbox_1_y_min + bbox_1_y_max),
        #                             (bbox_1_z_min + bbox_1_z_max)])
        #
        # center_2 = 0.5 * np.array ([(bbox_2_x_min + bbox_2_x_max),
        #                             (bbox_2_y_min + bbox_2_y_max),
        #                             (bbox_2_z_min + bbox_2_z_max)])

        verts = verts - J_3d[0]
        J_3d = J_3d - J_3d[0]
        s = ((D1 / D2 + eps) + (W1 / W2 + eps)) / 2.0
        # verts[:, 0] = verts[:, 0] * (D1 / D2 + eps)
        # verts[:, 1] = verts[:, 1] * (W1 / W2 + eps)
        # verts[:, 2] = verts[:, 2] * (H1 / H2 + eps)
        verts = verts * s
        J_3d = J_3d * s

        verts = verts + smpl_J[0]
        J_3d = J_3d + smpl_J[0]
        return verts.astype ('float16'), J_3d.astype ('float16')

    def set_params(self, pose=None, beta=None, trans=None):
        # if pose is not None:
        #     self.pose = pose
        if pose is not None:
            for i in self.ignor_J:
                pose[i] = [0, 0, 0]
            # pose[0]=self.or_pose[0]
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        self.update()
        return self.verts

    def compute_R_G(self):
        # self.J = self.J_regressor.dot(self.v_template)
        pose_cube = self.pose.reshape((-1, 1, 3))
        self.R = self.rodrigues(pose_cube)
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        G[0, :, :] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i, :, :] = G[self.parent[i], :, :].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i], ((self.J[i, :] - self.J[self.parent[i], :]).reshape([3, 1]))]
                    )
                )
            )
        return G

    def do_skinning(self, G):
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
                )
            )
        T = np.tensordot(self.weigths, G, axes=[[1], [0]])
        rest_shape_h = np.hstack((self.v_template, np.ones([self.v_template.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.verts = v + self.trans.reshape([1, 3])

    def update(self):
        G = self.compute_R_G()
        self.do_skinning(G)

    def rodrigues(self, r):
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)#旋转角度
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(np.float64).tiny)
        r_hat = r / theta#旋转轴
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])#斜对称矩阵
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )#单位矩阵
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        return np.vstack([x, np.array([[0.0, 0.0, 0.0, 1.0]])])

    def to_rest_pose(self):
        '''
        姿态不变只是旋转，只用pose[0]
        :return:
        '''
        invers_pose =np.zeros(self.pose.shape)
        invers_pose[0,:]=self.or_pose[0,:]
        pose_cube = invers_pose.reshape ((-1, 1, 3))
        R = self.rodrigues (pose_cube)
        G = np.empty ((self.kintree_table.shape[1], 4, 4))
        G[0, :, :] = self.with_zeros (np.hstack ((R[0], self.smpl_J[0, :].reshape ([3, 1]))))
        for i in range (1, self.kintree_table.shape[1]):
            G[i, :, :] = G[self.parent[i], :, :].dot (
                self.with_zeros (
                    np.hstack (
                        [R[i], ((self.smpl_J[i, :] - self.smpl_J[self.parent[i], :]).reshape ([3, 1]))]
                    )
                )
            )

        G = G - self.pack (
            np.matmul (
                G,
                np.hstack ([self.smpl_J, np.zeros ([24, 1])]).reshape ([24, 4, 1])
            )
        )
        T = np.tensordot (self.weigths, G, axes=[[1], [0]])
        T_inverse = np.linalg.inv (T)
        # self.smpl.set_params (self.or_pose, self.or_shape, self.or_trans)#将smpl拟合到相同的姿态下，从而得到T的逆
        rest_shape_h = np.hstack ((self.or_verts , np.ones ([self.or_verts.shape[0], 1])))
        self.v_template=np.matmul (T_inverse, rest_shape_h.reshape ([-1, 4, 1])).reshape ([-1, 4])[:, :3]
        or_J = np.hstack ((self.or_J, np.ones ([self.or_J.shape[0], 1])))
        self.J = np.matmul (np.linalg.inv (G), or_J.reshape ([-1, 4, 1])).reshape ([-1, 4])[:, :3]

    def pack(self, x):
        return np.dstack([np.zeros([x.shape[0], 4, 3]), x])

    def save_model(self, path):
        params = {'or_pose':self.or_pose,'weights': self.weigths, 'v_template': self.v_template,
                  'color': self.color, 'f': self.faces,
                  'kintree_table': self.kintree_table,
                  'parent': self.parent, 'J': self.J
                  }
        with open (path, 'wb') as f:
            pickle.dump (params,f)
        return params

    def save_mesh(self,vertices,faces,colors,path):
        mesh = trimesh.Trimesh (vertices=vertices, faces=faces, vertex_colors=colors)
        mesh.export (path)
    def output_mesh(self, path):

        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def output_T_posemesh(self, path):
        with open(path, 'w') as fp:
            for v in self.v_template:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def output_or_mesh(self, path):
        with open(path, 'w') as fp:
            for v in self.or_verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def show_mesh(self):
        mesh = trimesh.Trimesh (self.verts, self.faces, vertex_colors=self.color)
        mesh.show ()
    # def __init__(self, model_path):
    #     with open (model_path, 'rb') as f:
    #         params = pickle.load (f, encoding='iso-8859-1')
    #
    #         self.weigths = params['weights']
    #         self.v_template = params['v_template']
    #         self.faces = params['f']
    #         self.kintree_table = params['kintree_table']
    #         self.color=params['color']
    #         self.J=params['J']
    #         self.parent=params['patent']
    #
    #     self.pose_shape = [24, 3]
    #     self.beta_shape = [10]
    #     self.trans_shape = [3]
    #
    #     self.pose = np.zeros (self.pose_shape)
    #     self.beta = np.zeros (self.beta_shape)
    #     self.trans = np.zeros (self.trans_shape)
    #
    #     self.verts = self.v_template
    #
    #     self.R = None
    #
    #     self.update()

if __name__ == '__main__':

    with open ('data/output/smplh_1/baoluo_smpl.pkl', 'rb') as f:
        params = pickle.load (f, encoding='iso-8859-1')



