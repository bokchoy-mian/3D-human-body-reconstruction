import numpy as np
import math
import cv2
import pickle
import transforms3d
import trimesh
import copy
class Joints:
    def __init__(self, idx):
        self.idx = idx
        self.vector = None
        self.coordinate = None
        self.parent = None
        self.children = []
        self.align_R = np.eye (3)
        self.motion_R = None

    def set_motion_R(self, motion):
        self.motion_R = motion[self.idx]
        if self.parent is not None:
            self.motion_R = self.parent.motion_R.dot (self.motion_R)
        for child in self.children:
            child.set_motion_R (motion)

    def init_vector(self):
        if self.parent is not None:
            self.vector = self.coordinate - self.parent.coordinate

    def update_coord(self,r):

        self.coordinate = np.matmul (r,self.coordinate)
        self.init_vector()
        for child in self.children:
            child.update_coord(r)

class SMPLJoints:
    def __init__(self, idx):
        self.idx = idx
        self.to_parent = None
        self.parent = None
        self.coordinate = None
        self.matrix = None
        self.children = []
        self.align_R = np.eye(3)
        self.motion_R = np.eye(3)
        self.vector =None

        self.to_coordinate=None
        self.to_vector=None

    def init_bone(self):
        if self.parent is not None:
            self.to_parent = self.coordinate - self.parent.coordinate

    def init_vector(self):
        if self.parent is not None:
            self.vector =  self.coordinate- self.parent.coordinate

    def set_motion_R(self, motion):
        self.motion_R = motion[self.idx]
        if self.parent is not None:
            self.motion_R = self.parent.motion_R.dot(self.motion_R)
        for child in self.children:
            child.set_motion_R(motion)
    def set_align_R(self, motion):
        self.align_R = self.align_R.dot(motion)
        for child in self.children:
            child.set_align_R(motion)
    # def update_motion_R(self):
    #
    #     if self.parent is not None:
    #         self.motion_R = self.motion_R.dot(self.align_R)
    #
    #     for child in self.children:
    #         child.update_coord()
    def update_coord(self):
        if self.parent is not None:
            absolute_R = self.parent.motion_R.dot(self.parent.align_R)
            self.coordinate = self.parent.coordinate + np.squeeze(absolute_R.dot(np.reshape(self.to_parent, [3,1])))
            self.vector = self.coordinate- self.parent.coordinate
        for child in self.children:
            child.update_coord()

    def to_dict(self):
        ret = {self.idx: self}
        for child in self.children:
            ret.update(child.to_dict())
        return ret

    def export_G(self):
        G = np.zeros([4, 4])
        G[:3,:3] = self.motion_R.dot(self.align_R)
        G[:3,3] = self.coordinate
        G[3,3] = 1
        return G

    def export_theta(self):
        self_relative_G = None
        if self.parent is None:
            self_relative_G = self.export_G()[:3,:3]
        else:
            parent_G = self.parent.export_G()[:3,:3]
            self_G = self.export_G()[:3,:3]
            # parent_G * relative_G = self_G
            self_relative_G = np.linalg.inv(parent_G).dot(self_G)
        ax, rad = transforms3d.axangles.mat2axangle(self_relative_G)
        ax = ax[:3]
        axangle = ax / np.linalg.norm(ax) * rad
        return axangle

    def verctor2mat(self,to_vector,vector):
        if np.linalg.norm (vector - to_vector) < 1e-8:
            mat = np.eye (3)
        else:
            mat = transforms3d.axangles.axangle2mat (np.cross (vector, to_vector), math.acos (
                min (np.dot (vector, to_vector) / (
                        np.linalg.norm (vector) * np.linalg.norm (to_vector)), 1)
            ))
        return mat

class SMPLModel():
    '''Simplified SMPL model. All pose-induced transformation is ignored. Also ignore beta.'''
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            params = pickle.load(f,encoding='iso-8859-1')

            self.J_regressor = params['J_regressor']
            self.weights = params['weights']
            self.shapedirs = params['shapedirs']
            self.posedirs = params['posedirs']
            self.v_template = params['v_template']
            self.faces = params['f']
            self.kintree_table = params['kintree_table']

        id_to_col = {self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])}
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }#{id:parant}
        self.children={

        }

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.trans_shape = [3]

        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = None
        self.J = None
        self.R = None

        self.update()

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
        v_shaped = self.shapedirs.dot (self.beta) + self.v_template
        self.J = self.J_regressor.dot (v_shaped)#初始时J的位置
        pose_cube = self.pose.reshape ((-1, 1, 3))
        self.R = self.rodrigues (pose_cube)
        I_cube = np.broadcast_to (
            np.expand_dims (np.eye (3), axis=0),
            (self.R.shape[0] - 1, 3, 3)
        )
        lrotmin = (self.R[1:] - I_cube).ravel ()
        self.v_posed = v_shaped + self.posedirs.dot (lrotmin)
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
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        self.T_inverse = np.linalg.inv (T)
        rest_shape_h = np.hstack((self.v_posed, np.ones([self.v_posed.shape[0], 1])))
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

    def gen_J_3d(self):
        return self.J_regressor.dot(self.verts)

    def with_zeros(self, x):
        return np.vstack([x, np.array([[0.0, 0.0, 0.0, 1.0]])])

    def pack(self, x):
        return np.dstack([np.zeros([x.shape[0], 4, 3]), x])

    def inverse(self):
        # temp=self.verts[:,:]
        # self.verts=self.v_template[:,:]
        # self.v_template=temp[:,:]
        # G = self.compute_R_G ()
        rest_shape_h = np.hstack ((self.verts-self.trans.reshape([1, 3]), np.ones ([self.verts.shape[0], 1])))
        v = np.matmul (self.T_inverse, rest_shape_h.reshape ([-1, 4, 1])).reshape ([-1, 4])[:, :3]
        self.verts = v

    def gen_invers_pose(self,aligen_J_3d,pose):
        aligen_pose=np.zeros(pose.shape)
        aligen_pose[0,:]= -pose[0,:]
        for i in range (1, self.kintree_table.shape[1]):
            parent=self.parent[i]
            smpl_J = self.J[i, :]
            aligen_J = aligen_J_3d[i, :]
            smpl_J_parent=self.J[parent, :]
            aligen_J_parent=aligen_J_3d[parent,:]

            aligne_vector=aligen_J-aligen_J_parent
            smpl_vector=smpl_J-smpl_J_parent
            W=np.cross(smpl_vector,aligne_vector)#外积
            r_hat=W/np.linalg.norm(W)#旋转轴
            theta=smpl_vector.dot(aligne_vector) / \
                  (np.linalg.norm (aligne_vector) * np.linalg.norm (smpl_vector))#旋转角度
            aligen_pose[i,:]=r_hat/theta#旋转向量

        return aligen_pose

    def setup_joints(self,J):
        joints = {}
        for i in range (24):
            joints[i] = SMPLJoints (i)

        for child, parent in self.parent.items ():
            joints[child].parent = joints[parent]
            joints[parent].children.append (joints[child])

        for j in joints.values ():
            j.coordinate = J[j.idx]*100

        for j in joints.values ():
            j.init_bone ()
            j.init_vector()
        return joints

    def gen_re_pose(self,aligen_J_3d,pose,shape):

        def norm(vec):
            n = np.linalg.norm (vec)
            if n == 0:
                return None
            return vec / n


        self.set_params(pose=np.zeros(pose.shape),beta=shape)
        # self.show_mesh()
        smpl_J3d=self.gen_J_3d()
        smpl_J3d[13:, 2] = 0
        smpl_J=self.setup_joints(smpl_J3d)

        aligen_J_2d = copy.deepcopy (aligen_J_3d)
        # aligen_J_2d[13:, 2] = smpl_J3d[13:, 2]
        aligen_J_2d[13:, 2] = 0

        aligen_J = self.setup_joints (aligen_J_2d)
        pose[12:,:]=0

        motion=self.rodrigues(pose.reshape ((-1, 1, 3)))

        smpl_J[0].set_motion_R(motion)
        smpl_J[0].update_coord()
        # [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        for i in [1,2,4,5]:

            aligne_vector = norm(aligen_J[i].children[0].vector)
            smpl_vector = norm(smpl_J[i].children[0].vector)

            W = np.cross (smpl_vector,aligne_vector)  # 外积
            r_hat = W / np.linalg.norm (W)  # 旋转轴
            theta = math.acos (min ((np.dot  (smpl_vector,aligne_vector)  / \
                                     (np.linalg.norm (aligne_vector) * np.linalg.norm (smpl_vector))), 1))  # 旋转角度

            r = r_hat * theta  # 旋转向量
            r_mat = self.rodrigues (r.reshape ((-1, 1, 3)))[0]

            smpl_J[i].align_R=r_mat
            # smpl_J[i].set_align_R(r_mat)
            # smpl_J[i].update_coord()
        for i in [13, 14, 16, 17,18,19]:

            aligne_vector = norm(aligen_J[i].children[0].vector)
            smpl_vector = norm(smpl_J[i].children[0].vector)

            W = np.cross (aligne_vector,smpl_vector)  # 外积
            r_hat = W / np.linalg.norm (W)  # 旋转轴
            #这边旋转轴一定是垂直aligne_vector向外的啊
            # W = np.cross (aligne_vector, norm(aligen_J[i-1].children[0].vector))  # 外积,法线
            # r_hat = W / np.linalg.norm (W)  # 旋转轴
            # temp_vctor=min ((np.linalg.norm(np.cross (r_hat, smpl_vector)) /(np.linalg.norm (r_hat) * np.linalg.norm (smpl_vector))), 1)*smpl_vector
            # theta = math.acos (min ((np.dot (aligne_vector, temp_vctor) / \
            #                          (np.linalg.norm (aligne_vector) * np.linalg.norm (temp_vctor))), 1))  # 旋转角度

            theta = math.acos (min ((np.dot   (aligne_vector,smpl_vector)  / \
                                     (np.linalg.norm (aligne_vector) * np.linalg.norm (smpl_vector))), 1))  # 旋转角度

            r = r_hat * theta  # 旋转向量
            r_mat = self.rodrigues (r.reshape ((-1, 1, 3)))[0]

            # smpl_J[i].align_R=r_mat
            smpl_J[i].set_align_R (r_mat)
            smpl_J[i].update_coord ()
        return np.array([joint.export_theta() for joint in smpl_J.values()])

    def output_mesh(self, path):
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    def show_mesh(self):
        mesh = trimesh.Trimesh (self.v_template, self.faces)
        # # screen=mesh.scene()
        # # # screen.add_geometry (self.recover_trimesh)
        # # screen.add_geometry(smpl_l_hand_mesh)
        # # screen.add_geometry(smpl_r_hand_mesh)
        # screen.show()
        #
        mesh.show ()
if __name__ == '__main__':
    import pickle
    import cv2
    import PIL.Image as pil_img

    with open('model/smplh/SMPLH_MALE.pkl', 'rb') as f:
        params = pickle.load(f,encoding='iso-8859-1')
        smplh_tem=params['v_template']
    img_path= '../data/input/img/baoluo.png'
    img = cv2.imread (img_path).astype (np.float32)[:, :, ::-1] / 255.0
    H,W,_=img.shape
    smpl = SMPLModel('model/smpl/SMPL_MALE.pkl')
    # smpl.v_template=smplh_tem
    with open ('../data/output/smplh_1/baoluo_smpl.pkl', 'rb') as f:
        params = pickle.load (f, encoding='iso-8859-1')
        pose = params['spml_pose'].reshape([-1,3]).astype('float64')
        beta = params['spml_shape'].astype('float64')
        camera_rotation=params['camera_rotation'].astype('float64')
        camera_transl=params['camera_translation'].astype('float64')
        camera_center=params['camera_center'].astype('float64')
    trans = np.zeros(smpl.trans_shape)
    faces = smpl.faces
    # np.random.seed (9608)
    # pose = (np.random.rand(*smpl.pose_shape) - 0.5) * 0.4
    verts = smpl.set_params( beta=beta,pose=pose, trans=trans)
    outmesh_path = './smpl.obj'
    smpl.output_mesh(outmesh_path)
    out_rendimg_path='./rend_smpl_img.png'
    if out_rendimg_path:
        import os
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        import pyrender
        import trimesh
        material = pyrender.MetallicRoughnessMaterial (
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        out_mesh = trimesh.Trimesh (verts, smpl.faces, process=False)
        rot = trimesh.transformations.rotation_matrix (
            np.radians (180), [1, 0, 0])
        out_mesh.apply_transform (rot)
        # out_mesh.export (out_obj_path)
        mesh = pyrender.Mesh.from_trimesh (
            out_mesh,
            material=material)

        scene = pyrender.Scene (bg_color=[0.0, 0.0, 0.0, 0.0],
                                ambient_light=(0.3, 0.3, 0.3))
        scene.add (mesh, 'mesh')


        # Equivalent to 180 degrees around the y-axis. Transforms the fit to
        # OpenGL compatible coordinate system.
        camera_transl[0] *= -1.0

        camera_pose = np.eye (4)
        camera_pose[:3, 3] = camera_transl

        camera = pyrender.camera.IntrinsicsCamera (
            fx=5000., fy=5000.,
            cx=camera_center[0], cy=camera_center[1])
        scene.add (camera, pose=camera_pose)

        # Get the lights from the viewer
        viewer = pyrender.Viewer (scene, use_raymond_lighting=True,
                                  viewport_size=(W, H),
                                  cull_faces=False,
                                  run_in_thread=True,
                                  registered_keys=dict ())
        light_nodes = viewer._create_raymond_lights ()
        for node in light_nodes:
            scene.add_node (node)

        r = pyrender.OffscreenRenderer (viewport_width=W,
                                        viewport_height=H,
                                        point_size=1.0)
        color, mask = r.render (scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype (np.float32) / 255.0

        valid_mask = (mask > 0)[:, :, np.newaxis]
        input_img = img
        output_img = (color[:, :, :] * valid_mask +
                      (1 - valid_mask) * input_img)

        img = pil_img.fromarray ((output_img * 255).astype (np.uint8))
        img.save (out_rendimg_path)
