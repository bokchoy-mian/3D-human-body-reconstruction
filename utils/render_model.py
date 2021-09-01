#!/usr/bin/python
#-- coding:utf8 --
import os
os.environ['PYOPENGL_PLATFORM'] = 'gel'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
# import pyrender
# import trimesh
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer

from opendr.geometry import VertNormals
from opendr.geometry import TriNormals

class Render():
    def __init__(self,model, img,weigths,camera_center, camera_transl, camera_rotation):
        self.flength = 5000.
        self.near=0.1
        self.far=5000
        self.img=img
        self.verts=model.verts
        self.faces=model.faces
        self.model=model
        self.front_faces,self.front_verts,self.front_verts_index,\
            self.back_faces,self.back_verts,self.back_verts_index=model.divide_face()
        self.weights=weigths
        self.J_point=model.J
        self.H, self.W, _ = img.shape
        # self.camera=pyrender.camera.IntrinsicsCamera (fx=5000., fy=5000.,cx=camera_center[0], cy=camera_center[1])
        self.use_cam=ProjectPoints(f=self.flength * np.ones(2),rt=np.zeros(3),
                                t=camera_transl,k=np.zeros(5),c=camera_center)
        # self.ro_use_cam = ProjectPoints (f=self.flength * np.ones (2), rt=np.zeros (3),
        #                               t=-camera_transl, k=np.zeros (5), c=camera_center)
        self.normals = VertNormals (self.verts, self.faces, True).r.reshape ((-1, 3))
        # self.faces_naormals=TriNormals(self.verts,self.faces).r.reshape ((-1, 3))

    def create_renderer(self):
        rn = ColoredRenderer ()  # 区别

        rn.camera = self.use_cam
        rn.frustum = {'near': self.near, 'far': self.far, 'height': self.H, 'width': self.W}
        
        # if self.img is not None:
        #     rn.background_image =self.img / 255. if self.img.max () > 1 else self.img
        # else:
        #     background_image=np.ones(self.img.shape)
        #     rn.background_image=background_image
        
        return rn

    def normals_renderer(self):

        rn=self.create_renderer()
        rn.set (v=self.verts, f=self.faces, bgcolor=np.ones (3))
        rn.vc = self.normals
        rn.vc = (rn.vc + 1.0) * 0.5
        self.normals_img=rn.r

        return self.normals_img

    def front_normals_renderer(self):

        rn = self.create_renderer ()
        rn.set (v=self.front_verts, f=self.front_faces, bgcolor=np.ones (3))
        # rn.vc = self.normals[self.front_verts_index]
        rn.vc = VertNormals (self.front_verts,self.front_faces, True).r.reshape ((-1, 3))
        rn.vc = (rn.vc + 1.0) * 0.5
        self.front_normals_img = rn.r
        return self.front_normals_img

    def back_normals_renderer(self):

        rn = self.create_renderer ()
        rn.set (v=self.back_verts, f=self.back_faces, bgcolor=np.ones (3))
        # rn.vc = self.normals[self.back_verts_index]
        rn.vc =VertNormals (self.back_verts,self.back_faces, True).r.reshape ((-1, 3))
        rn.vc = (rn.vc + 1.0) * 0.5
        self.back_normals_img = rn.r
        return self.back_normals_img

    def re_back_normals_renderer(self):

        rn = self.create_renderer ()
        verts_z_mean=np.mean(self.verts[:,2])
        verts_z_max = np.max (self.verts[:, 2])
        verts_z_min = np.min (self.verts[:, 2])
        dif=verts_z_max+verts_z_min
        self.re_verts = self.verts[:,:]
        self.re_verts[:,2]*=-1.
        self.re_verts[:, 2]-=verts_z_max
        re_verts_z_mean = np.mean (self.re_verts[:, 2])

        rn.vc = self.normals
        rn.vc = (rn.vc + 1.0) * 0.5
        self.back_normals_img = rn.r

        return self.back_normals_img

    def weigth_render(self):
        rn=self.create_renderer()
        rn.set (v=self.verts, f=self.faces, bgcolor=np.ones (3))
        self.render_weigth=np.zeros([self.H,self.W,24])
        for i in range(8):
            temp_weigth=self.weights[:,i*3:(i+1)*3]
            rn.vc=temp_weigth
            self.render_weigth[:,:,i*3:(i+1)*3]=rn.r

        return self.render_weigth

    def recover_weigth_render(self,verts, faces,weights):
        rn=self.create_renderer()
        rn.set (v=verts, f=faces, bgcolor=np.ones (3))
        render_weigth=np.zeros([self.H,self.W,24])
        for i in range(8):
            temp_weigth=weights[:,i*3:(i+1)*3]
            rn.vc=temp_weigth
            render_weigth[:,:,i*3:(i+1)*3]=rn.r

        return render_weigth

    def J_point_renderer(self):

        rn=self.create_renderer()
        rn.set (v=self.J_point,  bgcolor=np.ones (3))

        rn.vc = np.ones(self.J_point.shape)
        # rn.vc = (rn.vc + 1.0) * 0.5
        self.J_img=rn.r

        return self.J_img

    def save_normal2img(self,save_path,save_img):
        normals_img=(save_img* 255).astype ('uint8')
        cv2.imwrite (save_path, normals_img[:, :, ::-1])
        
    def show_normal2img(self,show_img):
        normals_img = (show_img * 255).astype ('uint8')
        cv2.imshow ('rend_normal_img', normals_img[:, :, ::-1])
        cv2.waitKey()
    def save_weigth2img(self,save_path,weights):
        img=np.zeros((weights.shape[0],weights.shape[1],3))
        colormap = [(0, 0, 0.5),  (0, 0.5,  0.5), (0, 0.75,  0.5), (0, 1,  0.5),
                    (0.5, 0,  0.5),  (0.5, 0.5,  0.5), (0.5, 0.75,  0.5), (0.5, 1,  0.5),
                    (1, 0,  0.5),  (1, 0.5,  0.5), (1, 0.75,  0.5), (1, 1,  0.5),

                    (0, 0, 0), (0, 0.5, 0), (0, 0.75, 0), (0,1,0),
                    (0.5, 0, 0), (0.5, 0.5, 0), (0.5, 0.75, 0), (0.5, 1, 0),
                    (1, 0, 0),  (1, 0.5, 0), (1, 0.75, 0), (1, 1, 0),

                    (0, 0, 1),  (0, 0.5, 1), (0, 0.75, 1), (0, 1, 1),
                    (0.5, 0, 1),  (0.5, 0.5, 1), (0.5, 0.75, 1), (0.5, 1, 1),
                    (1, 0, 1),  (1, 0.5, 1), (1, 0.75, 1), (1, 1, 1)
                    ]
        for i in range(24):
            temp_weigth=weights[:,:,i]
            img += (temp_weigth[:,:,None] *colormap[i]* 255).astype ('uint8')
        cv2.imwrite (save_path, img)

    def save_normal2npy(self,save_path,save_normal):
        np.save(save_path,save_normal)

    def showimg(img, seg_img, outline_img):
        classes = np.array (('Background',  # always index 0
                             'Hat', 'Hair', 'Glove', 'Sunglasses',
                             'UpperClothes', 'Dress', 'Coat', 'Socks',
                             'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
                             'Face', 'Left-arm', 'Right-arm', 'Left-leg',
                             'Right-leg', 'Left-shoe', 'Right-shoe',))
        colormap = [(0, 0, 0),
                    (1, 0.25, 0), (0, 0.25, 0), (0.5, 0, 0.25), (1, 1, 1),
                    (1, 0.75, 0), (0, 0, 0.5), (0.5, 0.25, 0), (0.75, 0, 0.25),
                    (1, 0, 0.25), (0, 0.5, 0), (0.5, 0.5, 0), (0.25, 0, 0.5),
                    (1, 0, 0.75), (0, 0.5, 0.5), (0.25, 0.5, 0.5), (1, 0, 0),
                    (1, 0.25, 0), (0, 0.75, 0), (0.5, 0.75, 0), ]
        cmap = matplotlib.colors.ListedColormap (colormap)
        bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        norm = matplotlib.colors.BoundaryNorm (bounds, cmap.N)
        plt.figure (figsize=(10, 5))  # 设置窗口大小
        plt.suptitle ('Image')  # 图片名称
        plt.subplot (1, 3, 1), plt.title ('image')
        plt.imshow (cv2.cvtColor (img, cv2.COLOR_BGR2RGB)), plt.axis ('off')
        plt.subplot (1, 3, 2), plt.title ('seg_image')
        plt.imshow (seg_img, cmap=cmap, norm=norm), plt.axis ('off')
        plt.subplot (1, 3, 3), plt.title ('outline')
        plt.imshow (cv2.cvtColor (outline_img, cv2.COLOR_RGB2GRAY), cmap="gray"), plt.axis ('off')  # 这里显示灰度图要加cmap
        plt.savefig (fname="result.jpg")

        plt.show ()
        plt.close ()
        # cv2.imshow('',img)