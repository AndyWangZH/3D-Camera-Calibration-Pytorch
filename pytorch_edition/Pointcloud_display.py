import cv2
import numpy as np
import os
import math
from utils import *
import torch
from RefineModule import PlaneRefine
from tqdm import trange


def generate_inial_params_from_opencv(capture_folder):
    
    obj_points_list, camera_points_list, projector_points_list = find_point_pairs(capture_folder)
    ret, camera_mtx, camera_dist, camera_rvecs, camera_tvecs \
    = cv2.calibrateCamera(obj_points_list, camera_points_list, (1920,1200), None, None)
    camera_dist = np.squeeze(camera_dist)
    print("----------------------------------------")
    print('camera params')
    print('ret', ret)
    print('mtx', camera_mtx)
    print('dist', camera_dist)
    print("----------------------------------------")
    ret, projector_mtx, projector_dist, projector_rvecs, projector_tvecs \
    = cv2.calibrateCamera(obj_points_list, projector_points_list, (1280,720), None, None)
    projector_dist = np.squeeze(projector_dist)
    print("----------------------------------------")
    print('projector params')
    print('ret', ret)
    print('mtx', projector_mtx)
    print('dist', projector_dist)
    print("----------------------------------------")
    stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 500, 1e-15)
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(obj_points_list, camera_points_list, projector_points_list, camera_mtx, camera_dist, projector_mtx, projector_dist, imageSize=(1920,1200), criteria=stereocalib_criteria)
    print("----------------------------------------")
    print('stereo params')
    print('ret',ret)
    print('M1',M1)
    print('d1',d1)
    print('M2',M2)
    print('d2',d2)
    print('R',R)
    print('T',T)
    print("----------------------------------------")

    return camera_mtx, camera_dist, projector_mtx, projector_dist, R, T

def Generate_Scene_PointCloud(sample_folder, R, T, camera_mtx, camera_dist, projector_mtx, projector_dist, pcd_save_name):
    
    dmd_x, confidence_x = extract_dmd_coordinate(sample_folder, start_idx=0)
    dmd_y, confidence_y = extract_dmd_coordinate(sample_folder, start_idx=18)

    # Can these also in CUDA?
    camera_point_list = []
    projector_point_list = []

    for iy in range(1200):
        for ix in range(1920):
            if confidence_x[iy, ix] >=0.2 and confidence_y[iy, ix] >=0.2:
                camera_point = np.float64(np.array([ix, iy]))
                projector_point = np.array([dmd_x[iy, ix], dmd_y[iy, ix]])
                camera_point_list.append(camera_point)
                projector_point_list.append(projector_point)
    
    camera_point_list = np.array(camera_point_list)
    projector_point_list = np.array(projector_point_list)
    camera_point_list = torch.from_numpy(camera_point_list).cuda()
    projector_point_list = torch.from_numpy(projector_point_list).cuda()

    print("Triangulation")
    points_3d, _, _ = triangulation_pytorch(camera_point_list, projector_point_list, R, T, camera_mtx, camera_dist, projector_mtx, projector_dist)
    points_3d_display = points_3d.detach().cpu().numpy()
    np.savetxt(pcd_save_name, points_3d_display)

    return 0

if __name__ == "__main__":

    #Original Stereo Params
    camera_mtx, camera_dist, projector_mtx, projector_dist, R, T = generate_inial_params_from_opencv("F:/Company/Camera_Calibration_2022/3d_camera_calibration_2022/3d_camera_calibration_test_data")
    camera_mtx = torch.from_numpy(camera_mtx).cuda()
    camera_dist = torch.from_numpy(camera_dist).cuda()
    projector_mtx = torch.from_numpy(projector_mtx).cuda()
    projector_dist = torch.from_numpy(projector_dist).cuda()
    R = torch.from_numpy(R).cuda()
    T = torch.from_numpy(T).cuda()

    #Refine Stereo Params

    Generate_Scene_PointCloud("F:/Company/Camera_Calibration_2022/3d_camera_calibration_2022/3d_camera_calibration_test_data/02", R, T, camera_mtx, camera_dist, projector_mtx, projector_dist, "origin_02.xyz")