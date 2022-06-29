import cv2
import numpy as np
import os
import math
from utils import *
import torch
from RefineModule import PlaneRefine, PlaneRefineBatch
from tqdm import trange
from PointCloudDataset import CalibrationDataset
from torch.utils.data import Dataset, DataLoader
from Pointcloud_display import *

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


if __name__ == "__main__":

    epoch_num = 50

    camera_mtx, camera_dist, projector_mtx, projector_dist, R, T = generate_inial_params_from_opencv("F:/Company/Camera_Calibration_2022/3d_camera_calibration_2022/3d_camera_calibration_test_data")
    camera_mtx = torch.from_numpy(camera_mtx).cuda()
    camera_dist = torch.from_numpy(camera_dist).cuda()
    projector_mtx = torch.from_numpy(projector_mtx).cuda()
    projector_dist = torch.from_numpy(projector_dist).cuda()
    R = torch.from_numpy(R).cuda()
    T = torch.from_numpy(T).cuda()

    R_initial = R.clone().detach()
    T_initial = T.clone().detach()
    camera_mtx_initial = camera_mtx.clone().detach()
    camera_dist_initial = camera_dist.clone().detach()
    projector_mtx_initial = projector_mtx.clone().detach()
    projector_dist_initial = projector_dist.clone().detach()

    # Generate_Scene_PointCloud("F:/Company/Camera_Calibration_2022/3d_camera_calibration_2022/3d_camera_calibration_test_data/01", R_initial, T_initial, camera_mtx_initial, camera_dist_initial, projector_mtx_initial, projector_dist_initial, "test_pcd_pytorch_batch_origin.xyz")

    folder_path = "F:/Company/Camera_Calibration_2022/3d_camera_calibration_2022/3d_camera_calibration_test_data"
    train_dataset = CalibrationDataset(folder_path, camera_mtx_initial, camera_dist_initial, projector_mtx_initial, projector_dist_initial, R_initial, T_initial)
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=False)

    model = PlaneRefineBatch(R, T, camera_mtx, camera_dist, projector_mtx, projector_dist).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    criterion = torch.nn.MSELoss()

    for epoch in range(epoch_num):
        # epoch_loss = 0.0
        for i, data in enumerate(train_loader):
            # print(i)
            camera_points_list_filtered, projector_points_list_filtered, plane_params = data
            raw_pts, tri_error = model(camera_points_list_filtered, projector_points_list_filtered)
            plane_params = plane_params.repeat(1, 1, 10000).permute(0, 2, 1)
            tri_error = tri_error.mean()
            # print(tri_error)
            # print(raw_pts.shape)
            # print(plane_params[:, 0, :])
            # raw_pts_test = raw_pts.detach().cpu().numpy()
            # np.savetxt("raw_"+str(epoch)+"_"+str(i)+".xyz", raw_pts_test[0])

            clean_pts = torch.zeros_like(raw_pts)
            clean_pts[:, :, 0] = raw_pts[:, :, 0]
            clean_pts[:, :, 1] = raw_pts[:, :, 1]
            clean_pts[:, :, 2] = raw_pts[:, :, 0] * plane_params[:, :, 0] + raw_pts[:, :, 1] * plane_params[:, :, 1] + plane_params[:, :, 2]

            # clean_pts_test = clean_pts.detach().cpu().numpy()
            # np.savetxt("clean_train_"+str(epoch)+"_"+str(i)+".xyz", clean_pts_test[0])

            plane_loss = criterion(raw_pts, clean_pts)
            # print(plane_loss)

            loss = plane_loss + tri_error

            # epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch:", epoch, "Batch:", i, "Total Loss:", loss.item(), "Plane Loss:", plane_loss.item(), "Triangulation Loss:", tri_error.item())
    

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

    Generate_Scene_PointCloud("F:/Company/Camera_Calibration_2022/3d_camera_calibration_2022/3d_camera_calibration_test_data/01", R_rf, T_rf, camera_mtx_rf, camera_dist_rf, projector_mtx_rf, projector_dist_rf, "test_pcd_pytorch_batch_refine_3.xyz")