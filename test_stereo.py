import numpy as np
import cv2

def reproject_error(camera_points_list, project_points_list):

    total_error = 0
    for i in range(camera_points_list.shape[0]):
        error = cv2.norm(camera_points_list[i]-project_points_list[i], cv2.NORM_L2) 
        total_error += error*error
    
    total_error = np.sqrt(total_error / (camera_points_list.shape[0]*camera_points_list.shape[1]))
    print("Current Reprojection Error:", total_error)
    return 0

def Distort_Points(points, dist):

    r2 = points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1]
    # print(r2.shape)
    distort_factor = (1 + dist[0] * r2 + dist[1] * r2 * r2 + dist[2] * r2 * r2 * r2) / (1 + dist[3] * r2 + dist[4] * r2 * r2)
    # print(distort_factor.shape)
    points[:, 0] *= distort_factor
    points[:, 1] *= distort_factor

    return points

def InvTransformRT(R, T):

    R_inv = R.T
    T_inv = -R_inv @ T

    return R_inv, T_inv

def compute_stereo_residual(obj_points_list, camera_points_list, projector_points_list, \
                    camera_rvecs, camera_tvecs, camera_mtx, camera_dist, \
                    projector_rvecs, projector_tvecs, projector_mtx, projector_dist, \
                    R, T):
        
    camera_points_list = np.array(camera_points_list)
    camera_points_list = np.squeeze(camera_points_list, 1)

    projector_points_list = np.array(projector_points_list)
    projector_points_list = np.squeeze(projector_points_list, 1)

    camera2projector_points_2d_list = []
    projector2camera_points_2d_list = []

    R_inv, T_inv = InvTransformRT(R, T)

    for i in range(obj_points_list.shape[0]):
        obj_points = obj_points_list[i]
        
        camera_rvec = camera_rvecs[i]
        camera_rvec, _ = cv2.Rodrigues(camera_rvec)
        camera_tvec = camera_tvecs[i]

        camera_points_3d = camera_rvec@(obj_points.T)
        camera_points_3d = camera_points_3d.T
        camera_points_3d = camera_points_3d + camera_tvec[:, 0]

        camera2projector_points_3d = R@(camera_points_3d.T)
        camera2projector_points_3d = camera2projector_points_3d.T
        camera2projector_points_3d = camera2projector_points_3d + T[:, 0]

        camera2projector_points_2d = np.zeros((camera_points_3d.shape[0], 2))
        camera2projector_points_2d[:, 0] = camera2projector_points_3d[:, 0] / camera2projector_points_3d[:, 2] 
        camera2projector_points_2d[:, 1] = camera2projector_points_3d[:, 1] / camera2projector_points_3d[:, 2] 
        
        camera2projector_points_2d = Distort_Points(camera2projector_points_2d, projector_dist)
        camera2projector_points_2d[:, 0] = camera2projector_points_2d[:, 0] * projector_mtx[0, 0] + projector_mtx[0, 2]
        camera2projector_points_2d[:, 1] = camera2projector_points_2d[:, 1] * projector_mtx[1, 1] + projector_mtx[1, 2]
        camera2projector_points_2d_list.append(camera2projector_points_2d)

        projector_rvec = projector_rvecs[i]
        projector_rvec, _ = cv2.Rodrigues(projector_rvec)
        projector_tvec = projector_tvecs[i]

        projector_points_3d = projector_rvec@(obj_points.T)
        projector_points_3d = projector_points_3d.T
        projector_points_3d = projector_points_3d + projector_tvec[:, 0]

        projector2camera_points_3d = R_inv@(projector_points_3d.T)
        projector2camera_points_3d = projector2camera_points_3d.T
        projector2camera_points_3d = projector2camera_points_3d + T_inv[:, 0]

        projector2camera_points_2d = np.zeros((projector_points_3d.shape[0], 2))
        projector2camera_points_2d[:, 0] = projector2camera_points_3d[:, 0] / projector2camera_points_3d[:, 2]
        projector2camera_points_2d[:, 1] = projector2camera_points_3d[:, 1] / projector2camera_points_3d[:, 2]

        projector2camera_points_2d = Distort_Points(projector2camera_points_2d, camera_dist)
        projector2camera_points_2d[:, 0] = projector2camera_points_2d[:, 0] * camera_mtx[0, 0] + camera_mtx[0, 2]
        projector2camera_points_2d[:, 1] = projector2camera_points_2d[:, 1] * camera_mtx[1, 1] + camera_mtx[1, 2]
        projector2camera_points_2d_list.append(projector2camera_points_2d)

        # print("Camera Points:", camera_points_list[i])
        # print("Projector Points:", projector_points_list[i])
        # print("Distort Camera 2 Projector Points:", camera2projector_points_2d)
        # print(camera_points_3d.shape)
        
        # camera_points_2d = np.zeros((camera_points_3d.shape[0], 2))

        # camera_points_2d[:, 0] = camera_points_3d[:, 0] / camera_points_3d[:, 2] * camera_mtx[0,0] + camera_mtx[0,2]
        # camera_points_2d[:, 1] = camera_points_3d[:, 1] / camera_points_3d[:, 2] * camera_mtx[1,1] + camera_mtx[1,2]

        # print(camera_points_2d)
        
    camera2projector_points_2d_list = np.array(camera2projector_points_2d_list)
    projector2camera_points_2d_list = np.array(projector2camera_points_2d_list)
    print("Projector -> Camera")
    reproject_error(projector_points_list, camera2projector_points_2d_list)
    print("Camera -> Projector")
    reproject_error(camera_points_list, projector2camera_points_2d_list)

    return 0