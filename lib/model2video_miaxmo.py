#!/usr/bin/python
#-- coding:utf8 --

import os
import numpy as np
import cv2
import pickle
import copy
import trimesh
import open3d as o3d

class RecoverModel():
    '''将重建后的模型与smpl模型绑定'''

    def __init__(self, model_path):
        with open (model_path, 'rb') as f:
            params = pickle.load (f, encoding='iso-8859-1')

            self.weigths = params['weights']
            self.v_template = params['v_template']
            self.faces = params['f']
            self.kintree_table = params['kintree_table']
            self.color=params['color']
            self.J=params['J']
            self.parent=params['parent']
            self.or_pose=params['or_pose']
        self.ignor_J=[13,14,22,23]
        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.trans_shape = [3]

        self.pose = np.zeros (self.pose_shape)
        self.beta = np.zeros (self.beta_shape)
        self.trans = np.zeros (self.trans_shape)

        self.verts = self.v_template

        self.R = None

        self.update()

    def set_params(self, pose=None, beta=None, trans=None):
        if pose is not None:
            for i in self.ignor_J:
                pose[i]=[0,0,0]
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

    def pack(self, x):
        return np.dstack([np.zeros([x.shape[0], 4, 3]), x])

    def save_model(self, path):
        params = {'weights': self.weigths, 'v_template': self.v_template,
                  'color': self.color, 'f': self.faces,
                  'kintree_table': self.kintree_table,
                  'parent': self.parent, 'J': self.J
                  }
        with open (path, 'wb') as f:
            pickle.dump (params,f)
        return params

    def output_mesh(self, path):
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

class VideoWriter:
  """
  Write frames to a video.

  Call `write_frame` to write a single frame.
  Call `close` to release resource.

  """
  def __init__(self, path, width, height, fps):
    """
    Parameters
    ----------
    path : str
      Path to the video.
    width : int
      Frame width.
    height : int
      Frame height.
    fps : int
      Video frame rate.
    """
    self.fps = fps
    self.width = width
    self.height = height
    self.video = cv2.VideoWriter(
      path, cv2.VideoWriter_fourcc(*'H264'), fps, (width, height)
    )
    self.frame_idx = 0

  def write_frame(self, frame):
    """
    Write one frame.

    Parameters
    ----------
    frame : np.ndarray
      Frame to write.
    """
    self.video.write(np.flip(frame, axis=-1).copy())
    self.frame_idx += 1

  def close(self):
    """
    Release resource.
    """
    self.video.release()

def create_o3d_mesh(verts, faces,color):
  """
  Create a open3d mesh from vertices and faces.

  Parameters
  ----------
  verts : np.ndarray, shape [v, 3]
    Mesh vertices.
  faces : np.ndarray, shape [f, 3]
    Mesh faces.

  Returns
  -------
  o3d.geometry.TriangleMesh
    Open3d mesh.
  """
  color = color / 255.0
  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(faces)
  mesh.vertices = o3d.utility.Vector3dVector(verts)
  mesh.vertex_colors = o3d.utility.Vector3dVector (color)
  mesh.compute_vertex_normals()
  return mesh

def vis_mesh(verts, faces,color, width=1080, height=1080):
  """
  Visualize mesh with open3d.

  Parameters
  ----------
  verts : np.ndarray, shape [v, 3]
    Mesh vertices.
  faces : np.ndarray, shape [f, 3]
    Mesh faces.
  width : int
    Window width, by default 1080.
  height : int
    Window height, by default 1080.
  """
  mesh = create_o3d_mesh(verts, faces,color)
  viewer = o3d.visualization.Visualizer()
  viewer.create_window(width=width, height=height, visible=True)
  viewer.add_geometry(mesh)

  # viewer.update_renderer()
  viewer.run()

class open3d_render:
    def __init__(self ,bg_img,faces,verts,color,camera_transl,visible=True):

        self.flength = 5000./2
        self.near = 0.1
        self.far = 5000
        self.bg_img = bg_img / 255.0
        self.screen_size = (1024, 1024)
        self.camera_center = (self.screen_size[0] / 2, self.screen_size[1] / 2)
        self.camera_transl = camera_transl
        self.H, self.W = self.screen_size

        self.color = color / 255.0
        self.faces = faces
        self.verts=verts

        # self.bg_img_depth = np.full (bg_img.shape[:2], np.max (verts[:, 2])*0.8).astype (np.float32)
        # self.bg_faces, self.bg_points = self.verts2faces (self.bg_img_depth, self.bg_img[::-1,:,::-1])
        # # mesh=trimesh.Trimesh(vertices=self.bg_points[:,:3],faces=self.bg_faces,vertex_colors=self.bg_points[:,3:])
        # # mesh.show()
        # # mesh = trimesh.Trimesh (vertices=self.verts, faces=self.faces,
        # #                         vertex_colors=self.color)
        # # mesh.show ()
        # self.bg_mesh = o3d.geometry.TriangleMesh ()
        # self.bg_mesh.triangles = o3d.utility.Vector3iVector (self.bg_faces)
        # self.bg_mesh.vertices = o3d.utility.Vector3dVector (self.bg_points[:, :3])
        # self.bg_mesh.vertex_colors = o3d.utility.Vector3dVector (self.bg_points[:, 3:])

        self.mesh = o3d.geometry.TriangleMesh ()
        self.mesh.triangles = o3d.utility.Vector3iVector (self.faces)
        self.mesh.vertices = o3d.utility.Vector3dVector (self.verts)
        self.mesh.vertex_colors=o3d.utility.Vector3dVector (self.color)

        # cam_offset = 1.2
        self.vis = o3d.visualization.Visualizer ()
        self.vis.create_window (width=self.W, height=self.H, visible=visible)
        self.vis.add_geometry (self.mesh)
        # self.vis.add_geometry (self.bg_mesh)
        self.vis.get_render_option ().load_from_json('../data/renderoption.json')
        self.vis.get_render_option ().light_on=False
        self.vis.get_render_option().mesh_show_back_face=True
        # self.vis.get_render_option().background_color
        # self.vis.get_render_option ().point_size=1.0
        # self.vis.get_render_option ().show_coordinate_frame=True
        # self.vis.get_render_option ().point_show_normal=True
        # self.vis.get_render_option ().mesh_color_option = o3d.visualization.MeshColorOption.Color
        # self.vis.get_render_option ().mesh_shade_option=o3d.visualization.MeshShadeOption.Default
        # self.vis.get_render_option ().point_color_option=o3d.visualization.PointColorOption.Color

        self.view_control = self.vis.get_view_control ()
        cam_params = self.view_control.convert_to_pinhole_camera_parameters ()
        # rot = trimesh.transformations.rotation_matrix (np.radians (90), [1, 0, 0])
        # cam_params.extrinsic = np.array ([
        #     [1, 0, 0, self.camera_transl[0]],
        #     [0, 1, 0, self.camera_transl[1]],
        #     [0, 0, 1, self.camera_transl[2]],
        #     [0, 0, 0, 1],
        # ])
        cam_params.extrinsic = np.array ([
            [1, 0, 0, self.camera_transl[0]],
            [0, 0, -1, self.camera_transl[1]],
            [0, 1, 0, self.camera_transl[2]],
            [0, 0, 0, 1],
        ])
        # cam_params.extrinsic = np.array ([
        #     [ 0, 1, 0,self.camera_transl[0]],
        #     [ 0, 0, -1,self.camera_transl[1]],
        #     [ -1, 0, 0,self.camera_transl[2]],
        #     [0, 0, 0, 1],
        # ])

        cam_params.intrinsic=o3d.camera.PinholeCameraIntrinsic(self.W,self.H,self.flength,self.flength,self.camera_center[0]-0.5,self.camera_center[1]-0.5)
        self.view_control.convert_from_pinhole_camera_parameters (cam_params)
        # self.view_control.get_field_of_view ()
        # self.view_control.set_constant_z_near(0.1)
        # self.view_control.set_constant_z_far(100)

        # self.vis.get_view_control ().convert_from_pinhole_camera_parameters (cam_params)
        # view_control = self.vis.get_view_control ()
        # cam_params = view_control.convert_to_pinhole_camera_parameters ()
        # print(cam_params)

    def __call__(self, v):
        self.mesh.vertices = o3d.utility.Vector3dVector (v)
        self.mesh.rotate(self.mesh.get_rotation_matrix_from_xyz((-np.pi/2,0,0)), center=(0, 0, 0))
        # self.mesh.rotate (self.mesh.get_rotation_matrix_from_xyz ((0,-np.pi / 2,  0)), center=(0, 0, 0))
        self.mesh.compute_vertex_normals ()
        self.vis.update_geometry (self.mesh)
        self.vis.poll_events ()
        self.vis.update_renderer ()
        frame = (np.asarray (self.vis.capture_screen_float_buffer ()) * 255).astype (np.uint8)[:, :, ::-1]
        return frame

    def verts2faces(self, depth,color):
        '''
        :param verts_points:
        :return:
        '''
        high, wigth = depth.shape
        idx = np.arange (0, (high * wigth)).reshape ((high, wigth))

        # init X, Y coordinate tensors 生成网格
        X, Y = np.meshgrid (np.arange (wigth), np.arange (high))
        X = np.expand_dims (X, axis=2)  # (H,W,1)
        Y = np.expand_dims (Y, axis=2)
        dx = 1.0
        dy = 1.0
        x_cord = X * dx
        y_cord = Y * dy

        # convert the images to 3D mesh
        depth = np.expand_dims (depth, axis=2)

        points = np.concatenate ((x_cord, y_cord, depth, color), axis=2).reshape (-1, 6)
        points[:, :2] -=[wigth/2,high/2]
        points[:,:2]*=depth[0,0]/70

        p00_idx = idx[:-1, :-1].reshape (-1, 1)
        p10_idx = idx[1:, :-1].reshape (-1, 1)
        p11_idx = idx[1:, 1:].reshape (-1, 1)
        p01_idx = idx[:-1, 1:].reshape (-1, 1)
        faces = np.vstack (
            (np.hstack ((p00_idx, p10_idx, p01_idx)), np.hstack ((p01_idx, p10_idx, p11_idx)),
             np.hstack ((p00_idx, p01_idx, p10_idx)), np.hstack ((p01_idx, p11_idx, p10_idx))))

        return faces,points

    def close(self):
        self.vis.destroy_window ()
        self.vis.close()

class open3d_camera_render:
    def __init__(self ,bg_img,faces,verts,color,camera_transl,visible=True):

        self.flength = 5000./2
        self.bg_img = np.array(bg_img[:, :, ::-1])/255
        self.screen_size = (1024, 1024)
        self.camera_center = (self.screen_size[0] / 2, self.screen_size[1] / 2)
        self.camera_transl = camera_transl
        self.H, self.W = self.screen_size

        self.color = color / 255.0
        self.faces = faces
        self.verts=verts

        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.extrinsic = np.array ([
            [1, 0, 0, self.camera_transl[0]],
            [0, -1, 0, self.camera_transl[1]],
            [0, 0, -1, self.camera_transl[2]],
            [0, 0, 0, 1],
        ])

        # cam_params.intrinsic = o3d.camera.PinholeCameraIntrinsic (self.W, self.H, self.flength, self.flength,
        #                                                           self.camera_center[0] - 0.5,
        #                                                           self.camera_center[1] - 0.5)
        # pcd = o3d.geometry.PointCloud.create_from_rgbd_image (self.bg_mesh_RGBD_img,cam_params.intrinsic)

        # Flip it, otherwise the pointcloud will be upside down
        # pcd.transform ([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.bg_img_depth = np.full (bg_img.shape[:2], np.max(verts[:,2])*1.1).astype (np.float32)
        self.bg_faces,self.bg_points = self.verts2faces (self.bg_img_depth,self.bg_img)
        # mesh=trimesh.Trimesh(vertices=self.bg_points[:,:3],faces=self.bg_faces,vertex_colors=self.bg_points[:,3:])
        # mesh.show()
        # mesh = trimesh.Trimesh (vertices=self.verts, faces=self.faces,
        #                         vertex_colors=self.color)
        # mesh.show ()
        self.bg_mesh=o3d.geometry.TriangleMesh()
        self.bg_mesh.triangles = o3d.utility.Vector3iVector (self.bg_faces)
        self.bg_mesh.vertices=o3d.utility.Vector3dVector (self.bg_points[:,:3])
        self.bg_mesh.vertex_colors=o3d.utility.Vector3dVector (self.bg_points[:,3:])
        # self.bg_mesh.rotate (self.bg_mesh.get_rotation_matrix_from_xyz ((np.pi, 0, 0)), center=(0, 0, 0))

        self.mesh = o3d.geometry.TriangleMesh ()
        self.mesh.triangles = o3d.utility.Vector3iVector (self.faces)
        self.mesh.vertices = o3d.utility.Vector3dVector (self.verts)
        self.mesh.vertex_colors=o3d.utility.Vector3dVector (self.color)
        self.mesh.rotate (self.mesh.get_rotation_matrix_from_xyz (( np.pi ,0, 0)), center=(0, 0, 0))
        # cam_offset = 1.2
        self.vis = o3d.visualization.Visualizer ()
        self.vis.create_window (width=self.W, height=self.H, visible=visible)
        self.vis.add_geometry (self.mesh)
        self.vis.add_geometry(self.bg_mesh)
        self.vis.get_render_option ().load_from_json('../data/renderoption.json')
        self.vis.get_render_option ().light_on=False
        # self.vis.get_render_option ().point_size=1.0
        # self.vis.get_render_option ().show_coordinate_frame=True
        # self.vis.get_render_option ().point_show_normal=True
        # self.vis.get_render_option ().mesh_color_option = o3d.visualization.MeshColorOption.Color
        # self.vis.get_render_option ().mesh_shade_option=o3d.visualization.MeshShadeOption.Default
        # self.vis.get_render_option ().point_color_option=o3d.visualization.PointColorOption.Color

        self.view_control = self.vis.get_view_control ()

    def __call__(self, camera_path="../data/camera/camera_trajectory.json"):

        trajectory = o3d.io.read_pinhole_camera_trajectory (camera_path)
        cam_params = self.view_control.convert_to_pinhole_camera_parameters ()
        # cam_params.extrinsic = np.array ([
        #     [1, 0, 0, self.camera_transl[0]],
        #     [0, -1, 0, self.camera_transl[1]],
        #     [0, 0, -1, self.camera_transl[2]],
        #     [0, 0, 0, 1],
        # ])

        cam_params.intrinsic = o3d.camera.PinholeCameraIntrinsic (self.W, self.H, self.flength, self.flength,
                                                                  self.camera_center[0] - 0.5,
                                                                  self.camera_center[1] - 0.5)
        camera_transl_root=trajectory.parameters[0].extrinsic[:3,3]
        for index in range(len(trajectory.parameters)):
            cam_params_extrinsic = copy.deepcopy(trajectory.parameters[index].extrinsic)
            cam_params_extrinsic[:3,3]=cam_params_extrinsic[:3,3]-camera_transl_root+self.camera_transl
            cam_params.extrinsic=cam_params_extrinsic
            self.view_control.convert_from_pinhole_camera_parameters (cam_params)
            self.vis.poll_events ()
            self.vis.update_renderer ()
            frame = (np.asarray (self.vis.capture_screen_float_buffer ()) * 255).astype (np.uint8)[:, :, ::-1]

        return frame

    def verts2faces(self, depth,color):
        '''
        :param verts_points:
        :return:
        '''
        high, wigth = depth.shape
        idx = np.arange (0, (high * wigth)).reshape ((high, wigth))

        # init X, Y coordinate tensors 生成网格
        X, Y = np.meshgrid (np.arange (wigth), np.arange (high))
        X = np.expand_dims (X, axis=2)  # (H,W,1)
        Y = np.expand_dims (Y, axis=2)
        dx = 1.0
        dy = 1.0
        x_cord = X * dx
        y_cord = Y * dy

        # convert the images to 3D mesh
        depth = np.expand_dims (depth, axis=2)

        points = np.concatenate ((x_cord, y_cord, depth, color), axis=2).reshape (-1, 6)
        points[:, :2] -=[wigth/2,high/2]
        points[:,:2]*=depth[0,0]/70

        p00_idx = idx[:-1, :-1].reshape (-1, 1)
        p10_idx = idx[1:, :-1].reshape (-1, 1)
        p11_idx = idx[1:, 1:].reshape (-1, 1)
        p01_idx = idx[:-1, 1:].reshape (-1, 1)
        faces = np.vstack (
            (np.hstack ((p00_idx, p10_idx, p01_idx)), np.hstack ((p01_idx, p10_idx, p11_idx)),
             np.hstack ((p00_idx, p01_idx, p10_idx)), np.hstack ((p01_idx, p11_idx, p10_idx))))

        return faces,points

    def close(self):
        self.vis.destroy_window ()
        self.vis.close()

class View_mesh:

    def __init__(self,bg_img,expand_rate,camera_translation,out_video_path,mixamo_path,recovermodel=None,recovermodel_path=None):
        self.camera_translation = camera_translation
        self.camera_translation[2]=self.camera_translation[2]*expand_rate
        # self.rate=self.camera_translation[2]/5000.
        self.rate=(1.0 / 0.45)* 2.54 / 100.0
        self.out_video_path=out_video_path
        self.mixamo_path=mixamo_path

        if recovermodel:
            self.recover_model =recovermodel
        else:
            self.recover_model=RecoverModel(recovermodel_path)


        # self.poses,self.root_trans=self.read_amsass(self.amsass_path)
        self.poses, self.root_trans = self.read_mixamo (self.mixamo_path)
        self.num_frames = len (self.poses)

        self.render = open3d_render (bg_img=bg_img, faces=self.recover_model.faces,verts=self.recover_model.verts,
                                       color=self.recover_model.color,camera_transl=self.camera_translation)

    def __call__(self,video_fps):
        # writer = VideoWriter (video_path, width, height, fps)
        video = cv2.VideoWriter (
            self.out_video_path,
            cv2.VideoWriter_fourcc (*'mp4v'),
            video_fps,
            self.render.screen_size
        )

        # verts = self.recover_model.set_params (pose=self.recover_model.or_pose)

        # img = self.render (verts)
        # cv2.imshow ('frame', img)
        # cv2.waitKey (10)
        # video.write (img)

        for frame in range(0,self.num_frames):

            pose = self.poses[frame].reshape([24,3])
            # verts =self.recover_model.set_params (pose=pose,trans=self.root_trans[frame])
            verts = self.recover_model.set_params (pose=pose)
            img=self.render(verts)
            # cv2.imshow ('frame', img)
            # if cv2.waitKey (10) & 0xFF == ord ('q'):
            #     break
            video.write (img)
        video.release ()
        # cv2.destroyAllWindows ()
        self.render.close()

    def read_amsass(self,amsass_path):
        amsass=np.load(amsass_path)
        poses=amsass['poses'][:,:72]#smpl取24个关节点
        root_trans=amsass['trans']-amsass['trans'][0]
        return poses,root_trans

    def read_mixamo(self,mixamo_path):
        with open (mixamo_path, 'rb') as f:
            mixamo = pickle.load (f, encoding='iso-8859-1')
        anim_len = mixamo['anim_len']
        pose_array = mixamo['smpl_array'].reshape (anim_len, -1)
        cam_array = mixamo['cam_array']
        return pose_array, cam_array
        # mixamo=pickle.load()

def main_or(path,mixamo_index):
    os.chdir (os.path.dirname (__file__))
    # path = '../data/T_pose_test/test_30/'
    mixamo_path = os.path.join ('../data/mixamo/' + (mixamo_index), 'result.pkl')
    out_path=os.path.join (path , 'or_'+str (mixamo_index)+'.mp4')
    with open (path + 'smplh.pkl', 'rb') as f:
        smplh_result = pickle.load (f, encoding='iso-8859-1')

    camera_rotation = smplh_result['camera_rotation'].astype ('float64')
    camera_transl = smplh_result['camera_translation'].astype ('float64')
    camera_center = smplh_result['camera_center'].astype ('float64')

    front_color = cv2.imread (path + 'front_rgb.png')

    # recover_model = RecoverModel ('../models/model/recover/recover.pkl')
    # camera_motion=open3d_camera_render(front_color,faces=recover_model.faces,verts=recover_model.verts,
    #                                    color=recover_model.color,camera_transl=camera_transl)
    # camera_motion()
    render2video = View_mesh (front_color, 0.7, camera_transl,
                              out_path,
                              mixamo_path,
                              recovermodel_path=path + 'or_recover.pkl')
    render2video (30)

def main_replace_hands(path,mixamo_index):
    os.chdir (os.path.dirname (__file__))
    # path = '../data/T_pose_test/test_30/'
    out_path = os.path.join (path, 'replace_hands_'+str (mixamo_index) + '.mp4')
    mixamo_path=os.path.join('../data/mixamo/'+str(mixamo_index),'result.pkl')
    with open (path + 'smplh.pkl', 'rb') as f:
        smplh_result = pickle.load (f, encoding='iso-8859-1')

    camera_rotation = smplh_result['camera_rotation'].astype ('float64')
    camera_transl = smplh_result['camera_translation'].astype ('float64')
    camera_center = smplh_result['camera_center'].astype ('float64')

    front_color = cv2.imread (path + 'front_rgb.png')

    # recover_model = RecoverModel ('../models/model/recover/recover.pkl')
    # camera_motion=open3d_camera_render(front_color,faces=recover_model.faces,verts=recover_model.verts,
    #                                    color=recover_model.color,camera_transl=camera_transl)
    # camera_motion()
    render2video = View_mesh (front_color, 0.7, camera_transl,
                              out_path,
                              mixamo_path,
                              recovermodel_path=path + 'replace_hands_recover.pkl')
    render2video (30)

if __name__ == '__main__':

    # path_list=['../data/T_pose_test/test_35/','../data/T_pose_test/test_36/','../data/T_pose_test/test_37/','../data/T_pose_test/test_38/']
    # path_list =['../data/test/test_01/','../data/test/test_02/','../data/test/test_03/','../data/test/test_08/','../data/test/test_11/',
    #             '../data/test/test_12/','../data/test/test_14/','../data/test/test_15/','../data/test/test_17/','../data/test/test_19/']
    path_list =['../data/test_web_512/1/','../data/test_web/4/']
    # mixamo_path_list=['0007','0020','0022','0031','0032','0070','0083','0102','0131','0145']
    mixamo_path_list = ['0007', '0020', '0022', '0031', '0032', '0070', '0083', '0102','0131',  '0145']
    for path in path_list:
        for mixamo_path in mixamo_path_list:
            main_or(path,mixamo_path)
            # main_replace_hands(path,mixamo_path)