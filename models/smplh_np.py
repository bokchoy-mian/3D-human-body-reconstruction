import numpy as np
import pickle


class SMPLHModel():
    '''Simplified SMPLH model. All pose-induced transformation is ignored. Also ignore beta.'''
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            params = pickle.load(f,encoding='iso-8859-1')

            self.J_regressor = params['J_regressor']
            self.weights = params['weights']
            self.v_template = params['v_template']
            self.shapedirs = params['shapedirs']
            self.posedirs = params['posedirs']
            self.faces = params['f']
            self.kintree_table = params['kintree_table']

        id_to_col = {self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])}
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }

        self.pose_shape = [52, 3]
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
        self.J = self.J_regressor.dot(v_shaped)#(52,3)
        pose_cube = self.pose.reshape((-1, 1, 3))#(52,1,3)
        self.R = self.rodrigues(pose_cube)
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
                np.hstack([self.J, np.zeros([52, 1])]).reshape([52, 4, 1])
                )
            )
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        rest_shape_h = np.hstack((self.v_posed, np.ones([self.v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.verts = v + self.trans.reshape([1, 3])

    def update(self):
        G = self.compute_R_G()
        self.do_skinning(G)

    def rodrigues(self, r):
        #r (52,1,3)
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)##(52,1,1)
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

    def divide_face(self):
        """smpl模型分离正反面

        :param f:
        :param self.verts:
        :return:
        """
        f=self.faces
        v=self.verts
        N = f.shape[0]  # N face
        # z_vec = [0, 0, 1]
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
            if z==0:
                print(f[i])
            if z <= 0:
                front_temp=[]
                for index in f[i]:
                    if index in front_verts_index:
                        front_temp.append(front_verts_index.index(index))
                    else:
                        front_verts_index.append(index)
                        front_temp.append (front_verts_index.index (index))

                front_face.append (front_temp)
            elif z>=0:
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

        return front_face,front_verts, front_verts_index,back_face,back_verts,back_verts_index

    def write_obj(self,outpath,faces=None, verts=None):
        """

        :param self.verts_numpy:顶点坐标
        :param self.faces:面
        :param mesh_vn:法线
        :param outpath:
        :param verbose:
        :return:
        """
        with open (outpath, 'w') as fp:
            for v in verts:
                fp.write ('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in faces + 1:
                fp.write ('f %d %d %d\n' % (f[0], f[1], f[2]))


if __name__ == '__main__':
    import pickle
    import cv2
    import PIL.Image as pil_img

    img_path= '../data/input/img/baoluo.png'
    img = cv2.imread (img_path).astype (np.float32)[:, :, ::-1] / 255.0
    H,W,_=img.shape
    smpl = SMPLHModel('model/smplh/SMPLH_MALE.pkl')
    with open ('../data/output/smplh_1/baoluo_smplh.pkl', 'rb') as f:
        params = pickle.load (f, encoding='iso-8859-1')
        pose = params['spmlh_pose'].reshape([-1,3]).astype('float64')
        beta = params['spmlh_shape'].astype('float64')
        camera_rotation=params['camera_rotation'].astype('float64')
        camera_transl=params['camera_translation'].astype('float64')
        camera_center=params['camera_center'].astype('float64')
    trans = np.zeros(smpl.trans_shape)
    faces = smpl.faces
    # np.random.seed (9608)
    # pose = (np.random.rand(*smpl.pose_shape) - 0.5) * 0.4
    verts = smpl.set_params(beta=beta, pose=pose, trans=trans)
    outmesh_path = './smpl.obj'
    smpl.output_mesh(outmesh_path)
    out_rendimg_path='./rend_smplh_img.png'
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
