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

def generate_data_for_optimize(sample_folder, R, T, camera_mtx, camera_dist, projector_mtx, projector_dist):

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
    points_3d, _, error_3d = triangulation_pytorch(camera_point_list, projector_point_list, R, T, camera_mtx, camera_dist, projector_mtx, projector_dist)
    points_3d_test = points_3d.detach().cpu().numpy()
    np.savetxt("torch_original_whole_02.xyz", points_3d_test)

    # pts = np.array(points_3d.cpu())
    # np.savetxt("torch_test.xyz", pts)
    select_index = (error_3d < 8) & (points_3d[:,2] > 900) & (points_3d[:,2] < 1100)

    camera_point_list_filtered = camera_point_list[select_index].clone()
    projector_point_list_filtered = projector_point_list[select_index].clone()
    points_3d_filtered = points_3d[select_index].clone()

    points_3d_filtered_test = points_3d_filtered.detach().cpu()
    points_3d_filtered_test = np.array(points_3d_filtered_test)
    np.savetxt("torch_original_plane_02.xyz", points_3d_filtered_test)

    plane_params = plane_fitting_torch(points_3d_filtered)

    plane_fit = torch.zeros_like(points_3d_filtered)
    plane_fit[:, 0:2] = points_3d_filtered[:, 0:2]
    plane_fit[:, 2] = points_3d_filtered[:, 0] * plane_params[0, 0] + points_3d_filtered[:, 1] * plane_params[1, 0] + plane_params[2, 0]
    plane_fit = plane_fit.detach().cpu().numpy()
    np.savetxt("torch_fitting_plane_02.xyz", plane_fit)

    return camera_point_list_filtered, projector_point_list_filtered, plane_params, camera_point_list, projector_point_list


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)

    epoch_num = 500

    camera_mtx, camera_dist, projector_mtx, projector_dist, R, T = generate_inial_params_from_opencv("F:/Company/Camera_Calibration_2022/3d_camera_calibration_2022/3d_camera_calibration_test_data")
    camera_mtx = torch.from_numpy(camera_mtx).cuda()
    camera_dist = torch.from_numpy(camera_dist).cuda()
    projector_mtx = torch.from_numpy(projector_mtx).cuda()
    projector_dist = torch.from_numpy(projector_dist).cuda()
    R = torch.from_numpy(R).cuda()
    T = torch.from_numpy(T).cuda()

    camera_point_list_filtered, projector_point_list_filtered, plane_params, camera_point_list, projector_point_list = generate_data_for_optimize("F:/Company/Camera_Calibration_2022/3d_camera_calibration_2022/3d_camera_calibration_test_data/02", R, T, camera_mtx, camera_dist, projector_mtx, projector_dist)

    model = PlaneRefine(R, T, camera_mtx, camera_dist, projector_mtx, projector_dist).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    for epoch in trange(epoch_num):

        random_index = torch.randint(0, camera_point_list_filtered.shape[0], (10000, 1)).view(10000).cuda()
        
        raw_pts, _ = model(camera_point_list_filtered[random_index], projector_point_list_filtered[random_index])
        clean_pts = torch.zeros_like(raw_pts)
        clean_pts[:, 0] = raw_pts[:, 0]
        clean_pts[:, 1] = raw_pts[:, 1]
        clean_pts[:, 2] = raw_pts[:, 0] * plane_params[0, 0] + raw_pts[:, 1] * plane_params[1, 0] + plane_params[2, 0]

        loss = criterion(raw_pts, clean_pts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch:", epoch, "Loss:", loss)
    
    params_rf = {}
    for name, parameters in model.named_parameters():
        params_rf[name] = parameters

    R_rf = params_rf['R']
    print("R refined:", R_rf)
    T_rf = params_rf['T']
    print("T refined:", T_rf)
    camera_mtx_rf = params_rf['camera_mtx']
    print("camera_mtx refined:", camera_mtx_rf)
    camera_dist_rf = params_rf['camera_dist']
    print("camera_mtx refined:", camera_dist_rf)
    projector_mtx_rf = params_rf['projector_mtx']
    print("projector_mtx refined:", projector_mtx_rf)
    projector_dist_rf = params_rf['projector_dist']
    print("projector_mtx refined:", projector_dist_rf)

    points_3d_plane_rf, _, _ = triangulation_pytorch(camera_point_list_filtered, projector_point_list_filtered, R_rf, T_rf, camera_mtx_rf, camera_dist_rf, projector_mtx_rf, projector_dist_rf)
    points_3d_plane_rf = points_3d_plane_rf.detach().cpu().numpy()
    np.savetxt("torch_refine_plane_02.xyz", points_3d_plane_rf)

    points_3d_rf, _, _ = triangulation_pytorch(camera_point_list, projector_point_list, R_rf, T_rf, camera_mtx_rf, camera_dist_rf, projector_mtx_rf, projector_dist_rf)
    points_3d_rf = points_3d_rf.detach().cpu().numpy()
    np.savetxt("torch_refine_whole_02.xyz", points_3d_rf)