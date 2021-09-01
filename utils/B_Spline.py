#!/usr/bin/python
#-- coding:utf8 --
import numpy as np
from geomdl import fitting
from geomdl import multi
from geomdl import construct
from geomdl import exchange
from geomdl.visualization import VisMPL,VisPlotly,VisVTK

class B_spline_curve(object):
    def __init__(self,points,degree):
        self.points=points
        # self.points.append(self.points[0])
        self.degree=degree
        # self.points_n=points.shape[0]
        self.curve=fitting.interpolate_curve(self.points, degree=self.degree)
    def __call__(self,delta):
        '''

        :param delta: 间隔大小
        :return:
        '''
        self.curve.delta=delta
        # self.curve.evaluate ()
        curve_points = self.curve.evalpts
        return np.asarray(curve_points)
    def show_plot(self):
        self.curve.vis=VisMPL.VisCurve3D()
        self.curve.render()
    def show_html(self):
        self.curve.vis=VisPlotly.VisCurve3D()
        self.curve.render ()
    def show_VTK(self):
        self.curve.vis=VisVTK.VisCurve3D()
        self.curve.render()

class B_spline_curve_multi(object):
    def __init__(self,points,degree):
        """

        :param points: shape(n,4,3)
        :param degree:
        """
        self.points=points
        self.curves_n=points.shape[0]
        self.degree=degree
        self.CurveContainer=multi.CurveContainer()
        self.curves=[fitting.interpolate_curve(points.tolist(), degree=self.degree) for points in self.points]
        self.CurveContainer.append(self.curves)
    def __call__(self,delta):
        '''

        :param delta: 间隔大小
        :return:
        '''
        self.CurveContainer.delta=delta
        # self.curve.evaluate ()
        points = self.CurveContainer.evalpts
        points=np.swapaxes(np.asarray(points).reshape((self.curves_n,-1,3)),0,1)#(m,n,3)
        return points
    def show_plot(self):
        self.CurveContainer.vis=VisMPL.VisCurve3D()
        self.CurveContainer.render()
    def show_html(self):
        self.CurveContainer.vis=VisPlotly.VisCurve3D()
        self.CurveContainer.render ()
    def show_VTK(self):
        self.CurveContainer.vis=VisVTK.VisCurve3D()
        self.CurveContainer.render()

class B_spline_surface():
    def __init__(self, curve_points,size_u ,size_v, degree_u,degree_v):
        '''

        :param curve_points: 包含四条曲线[curve1,curve2,curve3,curve4]
        :param degree_u:
        :param degree_v:
        '''
        self.curve_points = curve_points
        # for i in range(len(self.curve_points)):
        #     self.curve_points[i].append (self.curve_points[i][0])
        self.size_u=size_u #defal=4
        self.size_v=size_v
        self.degree_u = degree_u #,=3
        self.degree_v = degree_v #,=3
        # self.points_n=points.shape[0]
        self.surface= fitting.interpolate_surface (self.curve_points,size_u=self.size_u,size_v=self.size_v,
                                                  degree_u=self.degree_u,degree_v=self.degree_v)

    def __call__(self, delta_v,delta_u):
        self.surface.delta_u=delta_u
        self.surface.delta_v=delta_v
        surface_points = self.surface.evalpts
        surface_points=np.asarray (surface_points).reshape ((self.surface.sample_size_u, self.surface.sample_size_v, -1))
        surface_faces=self.surface.faces
        return surface_points,surface_faces
    def show_plot(self):
        self.surface.vis=VisMPL.VisSurface()
        self.surface.render()
    def show_html(self):
        self.surface.vis=VisPlotly.VisSurface()
        self.surface.render ()
    def show_VTK(self):
        self.surface.vis=VisVTK.VisSurface()
        self.surface.render()

class B_spline_surface_from_curve():
    def __init__(self, curves,degree):
        '''

        :param curve_points: 包含四条曲线[curve1,curve2,curve3,curve4]
        :param degree_u:
        :param degree_v:
        '''
        self.curves = curves
        # for i in range(len(self.curve_points)):
        #     self.curve_points[i].append (self.curve_points[i][0])
        # self.size_u=size_u #defal=4
        # self.size_v=size_v
        # self.degree_u = degree_u #,=3
        # self.degree_v = degree_v #,=3
        # self.points_n=points.shape[0]
        # self.surface= fitting.interpolate_surface (self.curve_points,size_u=self.size_u,size_v=self.size_v,
        #                                           degree_u=self.degree_u,degree_v=self.degree_v)
        self.degree = degree
        self.surface =construct.construct_surface('v',self.curves[0],self.curves[1],
                                                  self.curves[2],self.curves[3],degree=self.degree)
    def __call__(self, delta_v,delta_u):
        self.surface.delta_u=delta_u
        self.surface.delta_v=delta_v
        surface_points = self.surface.evalpts
        surface_faces=self.surface.faces
        return surface_points,surface_faces
    def show_plot(self):
        self.surface.vis=VisMPL.VisSurface()
        self.surface.render()
    def show_html(self):
        self.surface.vis=VisPlotly.VisSurface()
        self.surface.render ()
    def show_VTK(self):
        self.surface.vis=VisVTK.VisSurface()
        self.surface.render()

def main():
    # # points=[[5,10],[15,25],[30,30],[45,5],[55,5],[70,40],[60,60],[35,60],[20,40]]
    # points=np.asarray(((-5, -5, 0), (-2.5, -5, 0), (0, -5, 0), (2.5, -5, 0), (5, -5, 0), (7.5, -5, 0), (10, -5, 0))).tolist()
    # b_curve=B_spline_curve(points,3)
    # re_points=b_curve(0.01)
    # b_curve.show_VTK()

    # points = ((-5, -5, 0), (-2.5, -5, 0), (0, -5, 0), (2.5, -5, 0), (5, -5, 0), (7.5, -5, 0), (10, -5, 0),
    #           (-5, 0, 3), (-2.5, 0, 3), (0, 0, 3), (2.5, 0, 3), (5, 0, 3), (7.5, 0, 3), (10, 0, 3),
    #           (-5, 5, 0), (-2.5, 5, 0), (0, 5, 0), (2.5, 5, 0), (5, 5, 0), (7.5, 5, 0), (10, 5, 0),
    #           (-5, 7.5, -3), (-2.5, 7.5, -3), (0, 7.5, -3), (2.5, 7.5, -3), (5, 7.5, -3), (7.5, 7.5, -3), (10, 7.5, -3),
    #           (-5, 5, -6), (-2.5, 5, -6), (0, 5, -6), (2.5, 5, -6), (5, 5, -6), (7.5, 5, -6), (10, 5, -6),
    #           (-5, 0, -6), (-2.5, 0, -6), (0, 0, -6), (2.5, 0, -6), (5, 0, -6), (7.5, 0, -6), (10, 0, -6),
    #           (-5, -5, 0), (-2.5, -5, 0), (0, -5, 0), (2.5, -5, 0), (5, -5, 0), (7.5, -5, 0), (10, -5, 0)
    #           )
    # points = ((0, -5, 0), (-2.5, -5, -2.5),(-2.5, -5, -2.5), (0, -5, -5), (2.5, -5, -2.5), (0, -5, 0),
    #           (0, -3.5, 1), (-3.5, -3.5, -2.5),(-3.5, -3.5, -2.5), (0, -3.5, -6), (3.5, -3.5, -2.5), (0, -3.5, 1),
    #
    #           (0, 3.5, 1), (-3.5, 3.5, -2.5),(-3.5, 3.5, -2.5), (0, 3.5, -6), (3.5, 3.5, -2.5), (0, 3.5, 1),
    #           (0, 5, 0), (-2.5, 5, -2.5),  (-2.5, 5, -2.5), (0, 5, -5), (2.5, 5, -2.5), (0, 5, 0),
    #           )
    points = ((0, -5, 0),  (-2.5, -5, -2.5), (0, -5, -5), (2.5, -5, -2.5), (0, -5, 0),
              (0, -3.5, 1),  (-3.5, -3.5, -2.5), (0, -3.5, -6), (3.5, -3.5, -2.5), (0, -3.5, 1),

              (0, 3.5, 1),  (-3.5, 3.5, -2.5), (0, 3.5, -6), (3.5, 3.5, -2.5), (0, 3.5, 1),
              (0, 5, 0),  (-2.5, 5, -2.5), (0, 5, -5), (2.5, 5, -2.5), (0, 5, 0),
              )
    size_u = 4
    size_v = 5
    degree_u = 2
    degree_v = 3
    #
    b_surface=B_spline_surface(points,size_u = 4,size_v = 5,degree_u = 2,degree_v = 2)
    b_surf_points,b_surf_faces=b_surface(delta_v=1/20,delta_u=1/15)
    b_surf_points=np.asarray(b_surf_points)
    b_surface.show_VTK()
    b_surface.show_plot()

    points = ((0, -5, 0), (-2.5, -5, -2.5), (0, -5, -5), (2.5, -5, -2.5), (0, -5, 0),
              (3.5, -3.5, -2.5), (0, -3.5, 1), (-3.5, -3.5, -2.5), (0, -3.5, -6), (3.5, -3.5, -2.5),
              (0, 3.5, 1), (-3.5, 3.5, -2.5), (0, 3.5, -6), (3.5, 3.5, -2.5), (0, 3.5, 1),
              (0, 5, 0), (-2.5, 5, -2.5), (0, 5, -5), (2.5, 5, -2.5), (0, 5, 0),
              )
    points=np.asarray (points).reshape (4, 5, 3)
    # points[1,:,:]=np.roll(points[1,:,:],1,axis=0)
    b_curves_multi=B_spline_curve_multi(points,2)
    b_curves_multi.show_VTK()
    b_curves_multi.show_plot()

    b_curve=B_spline_curve(points[0,:4,:],3)
    b_curve.show_VTK()
    b_curve.show_plot()

    b_surface_frome_curves=B_spline_surface_from_curve(b_curves_multi.curves,2)
    b_surf_points, b_surf_faces = b_surface_frome_curves (delta_v=0.1, delta_u=0.1)
    b_surface_frome_curves.show_VTK ()
    b_surface_frome_curves.show_plot ()
if __name__ == "__main__":
    main()
