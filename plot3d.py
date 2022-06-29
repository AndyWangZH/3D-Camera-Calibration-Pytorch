import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import cv2
#from triangulation import triangulation
#from calibration import N,nx,ny


def draw_camera(ax, mtx, h_size=1920, rvec=np.array([0.0,0.0,0.0]), tvec=np.matrix([0.0,0.0,0.0]).T, color='b'):
    fx = mtx[0,0] / h_size * 100
    fy = mtx[1,1] / h_size * 100
    cx = mtx[0,2] / h_size * 100
    cy = mtx[1,2] / h_size * 100
    
    rmtx, _ = cv2.Rodrigues(rvec)
    
    p0 = np.array(rmtx.T * (np.matrix([0,0,0]).T - tvec)).squeeze()
    p1 = np.array(rmtx.T * (np.matrix([cx,cy,fx]).T - tvec)).squeeze()
    p2 = np.array(rmtx.T * (np.matrix([-cx,cy,fx]).T - tvec)).squeeze()
    p3 = np.array(rmtx.T * (np.matrix([-cx,-cy,fx]).T - tvec)).squeeze()
    p4 = np.array(rmtx.T * (np.matrix([cx,-cy,fx]).T - tvec)).squeeze()
    
    ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]], color)
    ax.plot([p0[0],p2[0]], [p0[1],p2[1]], [p0[2],p2[2]], color)
    ax.plot([p0[0],p3[0]], [p0[1],p3[1]], [p0[2],p3[2]], color)
    ax.plot([p0[0],p4[0]], [p0[1],p4[1]], [p0[2],p4[2]], color)
    
    ax.plot([p1[0],p2[0],p3[0],p4[0],p1[0]], [p1[1],p2[1],p3[1],p4[1],p1[1]], [p1[2],p2[2],p3[2],p4[2],p1[2]], color)

def world2cam(world_point, rmtx, tvec):
    cam_point = np.matrix(rmtx) * np.matrix(world_point).T + np.matrix(tvec).T
    cam_point = np.array(cam_point).squeeze()
    return cam_point
    
def draw_board(ax, rvec, tvec, color, nx=10, ny=7, grid_width=20):
    rvec = np.array(rvec)
    tvec = np.array(tvec)
    assert rvec.shape == (3,) 
    assert tvec.shape == (3,) 
    #旋转和平移向量，
    #必须是一个3个元素的1维向量，
    #而不是一个3x1或3x3的二维矩阵
    
    rmtx, _ = cv2.Rodrigues(rvec)
    for x in range(0,nx*grid_width,grid_width):
        p0 = world2cam([x, 0, 0], rmtx, tvec)
        p1 = world2cam([x, (ny-1)*grid_width, 0], rmtx, tvec)
        ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]], color, alpha=0.1)
    for y in range(0,ny*grid_width,grid_width):
        p0 = world2cam([0, y, 0], rmtx, tvec)
        p1 = world2cam([(nx-1)*grid_width, y, 0], rmtx, tvec)
        ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]], color, alpha=0.1)
        
    p0 = world2cam([0, 0, 0], rmtx, tvec)
    p1 = world2cam([0, (ny-1)*grid_width, 0], rmtx, tvec)
    ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]], color, lw=3)
    p0 = world2cam([(nx-1)*grid_width, 0, 0], rmtx, tvec)
    p1 = world2cam([0, 0, 0], rmtx, tvec)
    ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]], color, lw=3)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca(projection='3d')


    cam1_mtx = np.loadtxt('cam1_intrisic_mtx.txt')
    cam2_mtx = np.loadtxt('cam2_intrisic_mtx.txt')
    dist1 = np.loadtxt('cam1_distortion_param.txt')
    dist2 = np.loadtxt('cam2_distortion_param.txt')
    
    rvecs = np.loadtxt('cam1_chessboard_r.txt')
    tvecs = np.loadtxt('cam1_chessboard_t.txt')
    
    cam1_points = np.loadtxt('cam1_points.txt')
    cam2_points = np.loadtxt('cam2_points.txt')

    R = np.loadtxt('extrinsic_r.txt')
    T = np.matrix(np.loadtxt('extrinsic_t.txt')).T

    rvec, _ = cv2.Rodrigues(R)

    draw_camera(ax, cam1_mtx, color='blue')
    draw_camera(ax, cam2_mtx, rvec=rvec, tvec=T, color='red')

    colors = "bgrcmyk"
    for i in range(15):
        color = colors[i%7]
        #print(i, color)
        cam1_point = cam1_points[i].reshape([-1,2])
        cam2_piont = cam2_points[i].reshape([-1,2])
        point3d, _, _ = triangulation(cam1_point, cam2_piont, 
                                      R,T,
                                      cam1_mtx,  dist1,  0,
                                      cam2_mtx, dist2, 0)

        ax.scatter3D(point3d[:,0], point3d[:,1], point3d[:,2], color=color)
        draw_board(ax, rvecs[i], tvecs[i], color)

    set_axes_equal(ax)
    plt.show()


