import numpy as np
import cv2
from scipy import optimize


def compute_residuals(obj_points_list, camera_points_list, camera_mtx, camera_dist, camera_rvecs, camera_tvecs):

    num_params = 4 + 5 + obj_points_list.shape[0] * 6
    init_params = np.zeros(num_params)
    init_params[0] = camera_mtx[0, 0]
    init_params[1] = camera_mtx[0, 2]
    init_params[2] = camera_mtx[1, 1]
    init_params[3] = camera_mtx[1, 2]

    init_params[4:9] = camera_dist

    for i in range(obj_points_list.shape[0]):
        # print(np.vstack((camera_rvecs[i], camera_tvecs[i]))[:, 0].shape)
        init_params[9+i*6:9+(i+1)*6] = np.vstack((camera_rvecs[i], camera_tvecs[i]))[:, 0] 


    # camera_mtx = np.zeros((3,3))
    # camera_mtx[0, 0] = init_params[0]
    # camera_mtx[0, 2] = init_params[1]
    # camera_mtx[1, 1] = init_params[2]
    # camera_mtx[1, 2] = init_params[3]
    # camera_mtx[2 ,2] = 1
    # camera_dist = init_params[4:9]

    camera_points_list = np.array(camera_points_list)
    camera_points_list = np.squeeze(camera_points_list, 1)
    print(camera_points_list.shape)
    project_points_list = []
    for i in range(obj_points_list.shape[0]):
        obj_points = obj_points_list[i]
        # camera_points = camera_points_list[i]
        camera_rvec = camera_rvecs[i]
        camera_tvec = camera_tvecs[i]
        # camera_rvec = init_params[9+i*6:9+(i+1)*6][:3]
        # camera_tvec = init_params[9+i*6:9+(i+1)*6][3:]
        project_points, _ = cv2.projectPoints(obj_points, camera_rvec, camera_tvec, camera_mtx, camera_dist)
        project_points = np.squeeze(project_points, 1)
        # print("Camera Points:", camera_points.shape)
        # print("Project Points:", project_points.shape)
        project_points_list.append(project_points)

    project_points_list = np.array(project_points_list)
    print(project_points_list.shape)

    reproject_error(camera_points_list, project_points_list)

    residual = camera_points_list - project_points_list

    # print(residual[0])
    # print(np.linalg.norm(camera_points_list-project_points_list)/(16*77))

    return residual

def reproject_error(camera_points_list, project_points_list):

    total_error = 0
    for i in range(camera_points_list.shape[0]):
        error = cv2.norm(camera_points_list[i]-project_points_list[i], cv2.NORM_L2) 
        total_error += error*error
    
    total_error = np.sqrt(total_error / (camera_points_list.shape[0]*camera_points_list.shape[1]))
    print("Reprojection Error:", total_error)
    return 0

def LM_refine(residual, camera_mtx, camera_dist, camera_rvecs, camera_tvecs):



    return 0