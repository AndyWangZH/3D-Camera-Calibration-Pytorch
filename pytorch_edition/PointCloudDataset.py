import cv2
import numpy as np
import os
import math
from utils import *
import torch
from RefineModule import PlaneRefine
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
import glob
import os

def generate_inial_params_from_opencv(capture_folder):
    
    obj_points_list, camera_points_list, projector_points_list = find_point_pairs(capture_folder)
    ret, camera_mtx, camera_dist, camera_rvecs, camera_tvecs \
    = cv2.calibrateCamera(obj_points_list, camera_points_list, (1920,1200), None, None)
    camera_dist = np.squeeze(camera_dist)
    # print("----------------------------------------")
    # print('camera params')
    # print('ret', ret)
    # print('mtx', camera_mtx)
    # print('dist', camera_dist)
    # print("----------------------------------------")
    ret, projector_mtx, projector_dist, projector_rvecs, projector_tvecs \
    = cv2.calibrateCamera(obj_points_list, projector_points_list, (1280,720), None, None)
    projector_dist = np.squeeze(projector_dist)
    # print("----------------------------------------")
    # print('projector params')
    # print('ret', ret)
    # print('mtx', projector_mtx)
    # print('dist', projector_dist)
    # print("----------------------------------------")
    stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 500, 1e-15)
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(obj_points_list, camera_points_list, projector_points_list, camera_mtx, camera_dist, projector_mtx, projector_dist, imageSize=(1920,1200), criteria=stereocalib_criteria)
    # print("----------------------------------------")
    # print('stereo params')
    # print('ret',ret)
    # print('M1',M1)
    # print('d1',d1)
    # print('M2',M2)
    # print('d2',d2)
    # print('R',R)
    # print('T',T)
    # print("----------------------------------------")

    return camera_mtx, camera_dist, projector_mtx, projector_dist, R, T


class CalibrationDataset(Dataset):

    def __init__(self, folder_path, camera_mtx, camera_dist, projector_mtx, projector_dist, R, T):
        self.folder_path = folder_path

        sample_folder_path = [
        x
        for x in glob.glob(os.path.join(folder_path, "*"))
        ]
        sample_folder_path = sorted(sample_folder_path)

        self.sample_folder_path = sample_folder_path
        self.camera_mtx = camera_mtx
        self.camera_dist = camera_dist
        self.projector_mtx = projector_mtx
        self.projector_dist = projector_dist
        self.R = R
        self.T = T

    def generate_data(self, sample_folder, camera_mtx, camera_dist, projector_mtx, projector_dist, R, T):

        dmd_x, confidence_x = extract_dmd_coordinate(sample_folder, start_idx=0)
        dmd_y, confidence_y = extract_dmd_coordinate(sample_folder, start_idx=18)

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

        points_3d, _, error_3d = triangulation_pytorch(camera_point_list, projector_point_list, R, T, camera_mtx, camera_dist, projector_mtx, projector_dist)

        select_index = (error_3d < 8) & (points_3d[:,2] > 900) & (points_3d[:,2] < 1100)

        camera_point_list_filtered = camera_point_list[select_index].clone()
        projector_point_list_filtered = projector_point_list[select_index].clone()
        points_3d_filtered = points_3d[select_index].clone()

        plane_params = plane_fitting_torch(points_3d_filtered)

        return camera_point_list_filtered, projector_point_list_filtered, plane_params

    def __getitem__(self, index):

        sample_folder = self.sample_folder_path[index]
        camera_point_list_filtered, projector_point_list_filtered, plane_params = self.generate_data(sample_folder, self.camera_mtx, self.camera_dist, self.projector_mtx, self.projector_dist, self.R, self.T)
        random_index = torch.randint(0, camera_point_list_filtered.shape[0], (10000, 1)).view(10000).cuda()

        return camera_point_list_filtered[random_index], projector_point_list_filtered[random_index], plane_params

    def __len__(self):

        return len(self.sample_folder_path)

if __name__ == "__main__":


    camera_mtx, camera_dist, projector_mtx, projector_dist, R, T = generate_inial_params_from_opencv("F:/Company/Camera_Calibration_2022/3d_camera_calibration_2022/3d_camera_calibration_test_data")
    camera_mtx = torch.from_numpy(camera_mtx).cuda()
    camera_dist = torch.from_numpy(camera_dist).cuda()
    projector_mtx = torch.from_numpy(projector_mtx).cuda()
    projector_dist = torch.from_numpy(projector_dist).cuda()
    R = torch.from_numpy(R).cuda()
    T = torch.from_numpy(T).cuda()

    # folder_path = "F:/Company/Camera_Calibration_2022/3d_camera_calibration_2022/3d_camera_calibration_test_data"

    # train_dataset = CalibrationDataset(folder_path, camera_mtx, camera_dist, projector_mtx, projector_dist, R, T)

    # train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

    # for batch_cp, batch_pp, batch_pa in train_loader:
    #     # print(batch_cp.shape)
    #     # print(batch_pp.shape)
    #     # print(batch_pa)


