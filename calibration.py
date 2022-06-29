import cv2
import os
import math
import numpy as np
from scipy.interpolate import interp2d
from pseudo_color import save_pseudo_color
from ellipse_center_correction import ellipse_center_correction
from plot3d import draw_camera, set_axes_equal, draw_board
from triangulation import triangulation, undistortPoints
import matplotlib.pyplot as plt
from scipy import optimize
import test
from tqdm import trange

def InvTransformRT(R, T):

    R_inv = R.T
    T_inv = -R_inv @ T

    return R_inv, T_inv

def compute_3d_residual(init_stereo_params, obj_points_list, camera_points_list, projector_points_list, \
                    camera_rvecs, camera_tvecs, camera_mtx, camera_dist, \
                    projector_rvecs, projector_tvecs, projector_mtx, projector_dist, re_error=False):

    # camera_points_3d_list = []
    points_3d_list = []

    R = init_stereo_params[:3]
    R, _ = cv2.Rodrigues(R)
    T = init_stereo_params[3:].reshape(3, 1)

    # print(obj_points_list.shape)

    for i in range(obj_points_list.shape[0]):
        obj_points = obj_points_list[i]
        camera_rvec = camera_rvecs[i]
        camera_tvec = camera_tvecs[i]
        projector_rvec = projector_rvecs[i]
        projector_tvec = projector_tvecs[i]
        camera_project_points, _ = cv2.projectPoints(obj_points, camera_rvec, camera_tvec, camera_mtx, camera_dist)
        camera_project_points = np.array(camera_project_points).reshape([-1,2])
        projector_project_points, _ = cv2.projectPoints(obj_points, projector_rvec, projector_tvec, projector_mtx, projector_dist)
        projector_project_points = np.array(projector_project_points).reshape([-1,2])
        points_3d, _, _ = triangulation(camera_project_points, projector_project_points, R, T, camera_mtx, camera_dist, 0, projector_mtx, projector_dist, 0)
        # points_3d_list.append(points_3d)
        camera_rvec, _ = cv2.Rodrigues(camera_rvec)
        camera_rvec_inv, camera_tvec_inv = InvTransformRT(camera_rvec, camera_tvec)
        points_3d = camera_rvec_inv@(points_3d.T)
        points_3d = points_3d.T 
        points_3d = points_3d + camera_tvec_inv[:, 0]
        points_3d_list.append(points_3d)
        # camera_points_3d = camera_rvec@(obj_points.T)
        # camera_points_3d = camera_points_3d.T
        # camera_points_3d = camera_points_3d + camera_tvec[:, 0]
        # camera_points_3d_list.append(camera_points_3d)
    
    # camera_points_3d_list = np.array(camera_points_3d_list)
    points_3d_list = np.array(points_3d_list)
    obj_points_list = np.float64(obj_points_list)
    # residual_3d = camera_points_3d_list.ravel() - points_3d_list.ravel()
    residual_z = points_3d_list[:,:,2].ravel()
    residual_pts = obj_points_list.ravel() - points_3d_list.ravel()
    # residual_3d = obj_points_list.ravel() - points_3d_list.ravel()
    residual_3d = np.hstack((residual_pts, residual_z))
    # print(residual_3d.shape)

    return residual_3d

def Reconstruction_Error(obj_points_list, camera_rvecs, camera_tvecs, camera_mtx, camera_dist,\
                        projector_rvecs, projector_tvecs, projector_mtx, projector_dist, R, T):

    total_error_3d = 0 

    for i in range(obj_points_list.shape[0]):
        obj_points = obj_points_list[i]
        camera_rvec = camera_rvecs[i]
        camera_tvec = camera_tvecs[i]
        projector_rvec = projector_rvecs[i]
        projector_tvec = projector_tvecs[i]
        camera_project_points, _ = cv2.projectPoints(obj_points, camera_rvec, camera_tvec, camera_mtx, camera_dist)
        camera_project_points = np.array(camera_project_points).reshape([-1,2])
        projector_project_points, _ = cv2.projectPoints(obj_points, projector_rvec, projector_tvec, projector_mtx, projector_dist)
        projector_project_points = np.array(projector_project_points).reshape([-1,2])
        points_3d, _, _ = triangulation(camera_project_points, projector_project_points, R, T, camera_mtx, camera_dist, 0, projector_mtx, projector_dist, 0)
        
        camera_rvec, _ = cv2.Rodrigues(camera_rvec)
        # camera_points_3d = camera_rvec@(obj_points.T)
        # camera_points_3d = camera_points_3d.T
        # camera_points_3d = camera_points_3d + camera_tvec[:, 0]
        camera_rvec_inv, camera_tvec_inv = InvTransformRT(camera_rvec, camera_tvec)
        points_3d = camera_rvec_inv@(points_3d.T)
        points_3d = points_3d.T 
        points_3d = points_3d + camera_tvec_inv[:, 0]
        # print("Reproject Points 3D:", points_3d)
        # print("Object Points:", obj_points)
        # print("Camera Points 3D:", camera_points_3d)
        # error_3d = cv2.norm(camera_points_3d, points_3d, cv2.NORM_L2)
        error_3d = cv2.norm(np.float64(obj_points), points_3d, cv2.NORM_L2)
        total_error_3d += error_3d*error_3d
    
    total_error_3d = np.sqrt(total_error_3d / (obj_points_list.shape[0]*obj_points_list.shape[1]))
    print("Reconstruction Error:", total_error_3d)

    return 0

def Distort_Points(points, dist):

    r2 = points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1]
    distort_factor = (1 + dist[0] * r2 + dist[1] * r2 * r2 + dist[2] * r2 * r2 * r2) / (1 + dist[3] * r2 + dist[4] * r2 * r2)
    points[:, 0] *= distort_factor
    points[:, 1] *= distort_factor

    return points

def create_stereo_inial_params(R, T):

    init_stereo_params = np.zeros(6)
    R, _ = cv2.Rodrigues(R)
    init_stereo_params[:3] = R[:, 0]
    init_stereo_params[3:] = T[:, 0]

    return init_stereo_params

def compute_stereo_residual(init_stereo_params, obj_points_list, camera_points_list, projector_points_list, \
                    camera_rvecs, camera_tvecs, camera_mtx, camera_dist, \
                    projector_rvecs, projector_tvecs, projector_mtx, projector_dist, re_error=False):
    
    R = init_stereo_params[:3]
    R, _ = cv2.Rodrigues(R)
    T = init_stereo_params[3:].reshape(3, 1)

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
        
    camera2projector_points_2d_list = np.array(camera2projector_points_2d_list)
    projector2camera_points_2d_list = np.array(projector2camera_points_2d_list)
    if re_error:
        print("Camera -> Projector")
        reproject_error(projector_points_list, camera2projector_points_2d_list)
        print("Projector -> Camera")
        reproject_error(camera_points_list, projector2camera_points_2d_list)

    #stereo_residual = camera_points_list.ravel() - projector2camera_points_2d_list.ravel()
    stereo_residual_camera2projector = projector_points_list.ravel() - camera2projector_points_2d_list.ravel()
    stereo_residual_projector2camera = camera_points_list.ravel() - projector2camera_points_2d_list.ravel()

    stereo_residual_bi = np.hstack((stereo_residual_camera2projector, stereo_residual_projector2camera))
    # print(stereo_residual_bi.shape)

    return stereo_residual_bi

def LM_stereo_refine(residual_function, init_stereo_params, obj_points_list, camera_points_list, projector_points_list, \
                    camera_rvecs, camera_tvecs, camera_mtx, camera_dist, \
                    projector_rvecs, projector_tvecs, projector_mtx, projector_dist):

    sol = optimize.least_squares(residual_function, init_stereo_params, args=(obj_points_list, camera_points_list, projector_points_list, \
                    camera_rvecs, camera_tvecs, camera_mtx, camera_dist, \
                    projector_rvecs, projector_tvecs, projector_mtx, projector_dist), kwargs={'re_error':False},method='lm', xtol=1e-15, ftol=1e-15)
        
    R_f = sol.x[:3]
    R_f, _ = cv2.Rodrigues(R_f)
    T_f = sol.x[3:].reshape(3,1)

    return R_f, T_f

def reproject_error(camera_points_list, project_points_list):

    total_error = 0
    for i in range(camera_points_list.shape[0]):
        error = cv2.norm(camera_points_list[i]-project_points_list[i], cv2.NORM_L2) 
        total_error += error*error
    
    total_error = np.sqrt(total_error / (camera_points_list.shape[0]*camera_points_list.shape[1]))
    print("Reprojection Error:", total_error)
    return 0

def compute_residuals(init_params, obj_points_list, camera_points_list, re_error=False):

    camera_mtx = np.zeros((3,3))
    camera_mtx[0, 0] = init_params[0]
    camera_mtx[0, 2] = init_params[1]
    camera_mtx[1, 1] = init_params[2]
    camera_mtx[1, 2] = init_params[3]
    camera_mtx[2 ,2] = 1
    camera_dist = init_params[4:9]

    camera_points_list = np.array(camera_points_list)
    camera_points_list = np.squeeze(camera_points_list, 1)
    project_points_list = []
    for i in range(obj_points_list.shape[0]):
        obj_points = obj_points_list[i]
        camera_rvec = init_params[9+i*6:9+(i+1)*6][:3]
        camera_tvec = init_params[9+i*6:9+(i+1)*6][3:]
        project_points, _ = cv2.projectPoints(obj_points, camera_rvec, camera_tvec, camera_mtx, camera_dist)
        project_points = np.squeeze(project_points, 1)
        project_points_list.append(project_points)

    project_points_list = np.array(project_points_list)

    if re_error:
        reproject_error(camera_points_list, project_points_list)
    
    # Not quite sure about here
    # residual = np.linalg.norm(camera_points_list.reshape(-1, 2)-project_points_list.reshape(-1, 2), axis=1)
    residual = camera_points_list.ravel() - project_points_list.ravel()

    return residual

def compute_residuals_3d_mono(init_params, obj_points_list, camera_points_list, re_error=False):
    
    camera_mtx = np.zeros((3,3))
    camera_mtx[0, 0] = init_params[0]
    camera_mtx[0, 2] = init_params[1]
    camera_mtx[1, 1] = init_params[2]
    camera_mtx[1, 2] = init_params[3]
    camera_mtx[2 ,2] = 1
    camera_dist = init_params[4:9]

    camera_points_list = np.array(camera_points_list)
    camera_points_list = np.squeeze(camera_points_list, 1)
    reproject_3dpoints_list = []
    for i in range(obj_points_list.shape[0]):
        obj_points = obj_points_list[i]
        camera_rvec = init_params[9+i*6:9+(i+1)*6][:3]
        camera_rvec, _ = cv2.Rodrigues(camera_rvec)
        camera_tvec = init_params[9+i*6:9+(i+1)*6][3:].reshape(3, 1)
        camera_rvec_inv, camera_tvec_inv = InvTransformRT(camera_rvec, camera_tvec)
        # print(camera_rvec_inv)
        # print(camera_tvec_inv)
        camera_points = camera_points_list[i]
        # camera_points_n = undistortPoints(camera_points, camera_mtx, camera_dist)
        camera_points_n = cv2.undistortPoints(camera_points, camera_mtx, camera_dist)
        # print("Camera Points N:", camera_points_n.shape)
        camera_points_3d = np.zeros((obj_points_list.shape[1], 3))
        camera_points_3d[:, 0] = camera_points_n[:, 0, 0] * 12
        camera_points_3d[:, 1] = camera_points_n[:, 0, 1] * 12
        camera_points_3d[:, 2] = 12
        reproject_3dpoints = []
        for i in range(camera_points_3d.shape[0]):
            # print(i)
            camera_point_3d = camera_points_3d[i].reshape(3, 1)
            camera_ray = camera_point_3d / np.sqrt((camera_point_3d[0,0]*camera_point_3d[0,0] + camera_point_3d[1,0]*camera_point_3d[1,0] + camera_point_3d[2,0]*camera_point_3d[2,0]))
            # print("camera_ray:", camera_ray)
            L = np.array([0,0,1]).reshape(3, 1)
            lambda_r = -((L.T)@camera_tvec_inv)/((L.T)@camera_rvec_inv@camera_ray)
            # print("lambda_r:", lambda_r)
            reproject_3dpoint = (camera_tvec_inv + lambda_r*(camera_rvec_inv@camera_ray)).reshape(-1, 3)
            # print("reproject_3dpoint:", reproject_3dpoint)
            # print("obj_point:", obj_points[0])
            reproject_3dpoints.append(reproject_3dpoint)
        reproject_3dpoints = np.array(reproject_3dpoints).squeeze(1)
        # reproject_error(obj_points, reproject_3dpoints)
        # print("Object Points:", obj_points)
        # print("Reproject 3D Points:", reproject_3dpoints)
        reproject_3dpoints_list.append(reproject_3dpoints)
    reproject_3dpoints_list = np.array(reproject_3dpoints_list)

    # print(obj_points_list.shape)
    # print(reproject_3dpoints_list.shape)
    if re_error:
        reproject_error(obj_points_list, reproject_3dpoints_list)

    residual = obj_points_list.ravel() - reproject_3dpoints_list.ravel()

    return residual

def create_inial_params(mtx, dist, rvecs, tvecs, img_num):

    num_params = 4 + 5 + img_num * 6
    init_params = np.zeros(num_params)
    init_params[0] = mtx[0, 0] 
    init_params[1] = mtx[0, 2]
    init_params[2] = mtx[1, 1] 
    init_params[3] = mtx[1, 2]
    init_params[4:9] = dist

    for i in range(img_num):
        # print(np.vstack((camera_rvecs[i], camera_tvecs[i]))[:, 0].shape)
        init_params[9+i*6:9+(i+1)*6] = np.vstack((rvecs[i], tvecs[i]))[:, 0] 

    return init_params

def LM_refine(residual_function, init_params, obj_points_list, camera_points_list, re_error):
    
    sol = optimize.least_squares(residual_function, init_params, args=(obj_points_list, camera_points_list), kwargs={'re_error':re_error}, method='lm', xtol=1e-15, ftol=1e-15)
    
    mtx_f = np.zeros((3,3))
    mtx_f[0, 0] = sol.x[0]
    mtx_f[0, 2] = sol.x[1]
    mtx_f[1, 1] = sol.x[2]
    mtx_f[1, 2] = sol.x[3]
    mtx_f[2, 2] = 1
    dist_f = sol.x[4:9]

    rvecs_f = []
    tvecs_f = []

    for i in range(obj_points_list.shape[0]):
        RT_f = sol.x[9+i*6:9+(i+1)*6]
        rvecs_f.append(RT_f[:3].reshape(3, 1))
        tvecs_f.append(RT_f[3:].reshape(3, 1))

    rvecs_f = np.array(rvecs_f)
    tvecs_f = np.array(tvecs_f)

    return mtx_f, dist_f, rvecs_f, tvecs_f


def create_board_points():
    points = []
    for i in range(11):
        if i%2==0:
            x0 = 15.0
        else:
            x0 = 25.0
        y = 15.0+i*10.0
        for j in range(7):
            points.append([x0+j*20.0, y, 0.0])
    return points
            
    
def read_phases(sample_folder, start_idx, n_phases):
    phases = []
    for i in range(start_idx, start_idx+n_phases):
        phase = cv2.imread(os.path.join(sample_folder, 'phase%02d.bmp'%i), 0)
        phases.append(phase)
    return phases
    
def unwarp_phase(last_unwrapped_phase, new_phase, multiply_rate):
    k = np.int32(0.5 + (multiply_rate * last_unwrapped_phase - new_phase) / math.pi);
    new_unwrapped_phase = math.pi * k + new_phase;
    return new_unwrapped_phase
    
def extract_dmd_coordinate(sample_folder, start_idx):
    phases = read_phases(sample_folder, start_idx+0, 4)
    phi_1, confidence_1 = extract_4_phase_shift(*phases)
    #save_pseudo_color('phi_1.png', phi_1, gray_max=2*math.pi)
    
    phases = read_phases(sample_folder, start_idx+4, 4)
    phi_2, confidence_2 = extract_4_phase_shift(*phases)
    phi_12 = unwarp_phase(phi_1, phi_2, 8)
    #save_pseudo_color('phi_12.png', phi_12, gray_max=16*math.pi)

    phases = read_phases(sample_folder, start_idx+8, 4)
    phi_3, confidence_3 = extract_4_phase_shift(*phases)
    phi_123 = unwarp_phase(phi_12, phi_3, 4)
    #save_pseudo_color('phi_123.png', phi_123, gray_max=64*math.pi)
    
    phases = read_phases(sample_folder, start_idx+12, 6)
    phi_4, confidence_4 = extract_6_phase_shift(*phases)
    phi_1234 = unwarp_phase(phi_123, phi_4, 4)
    #save_pseudo_color('phi_1234.png', phi_1234, gray_max=256*math.pi)
    
    coordinate = phi_1234 / (256*math.pi) * 1280
    
    return coordinate, confidence_4


    
def extract_4_phase_shift(phase0, phase1, phase2, phase3):
    a = np.float64(phase3) - np.float64(phase1)
    b = np.float64(phase0) - np.float64(phase2)
    r = np.sqrt(a*a + b*b) + 0.5
    confidence = r / 255.0
    phi = math.pi + np.arctan2(a, b)

    return phi, confidence
    
def extract_6_phase_shift(phase0, phase1, phase2, phase3, phase4, phase5):
    a = np.float64(phase3) * math.cos(0 * math.pi / 6.0) + \
        np.float64(phase4) * math.cos(2 * math.pi / 6.0) + \
        np.float64(phase5) * math.cos(4 * math.pi / 6.0) + \
        np.float64(phase0) * math.cos(6 * math.pi / 6.0) + \
        np.float64(phase1) * math.cos(8 * math.pi / 6.0) + \
        np.float64(phase2) * math.cos(10 * math.pi / 6.0)
        
    b = np.float64(phase3) * math.sin(0 * math.pi / 6.0) + \
        np.float64(phase4) * math.sin(2 * math.pi / 6.0) + \
        np.float64(phase5) * math.sin(4 * math.pi / 6.0) + \
        np.float64(phase0) * math.sin(6 * math.pi / 6.0) + \
        np.float64(phase1) * math.sin(8 * math.pi / 6.0) + \
        np.float64(phase2) * math.sin(10 * math.pi / 6.0)
        
    r = np.sqrt(a*a + b*b) + 0.5
    confidence = r / 255.0
    phi = math.pi + np.arctan2(a, b)

    return phi, confidence
    
def show_float32_image(window_name, image, confidence):
    _max = np.max(image)
    _min = np.min(image)
    _image = np.uint8((image - _min) / (_max - _min) * 255)
    _image[confidence<0.3] = 0
    #print(_image)
    cv2.namedWindow(window_name, 0)
    cv2.imshow(window_name, _image)
    cv2.waitKey(1)
    return _image
    
def interp2d(image, point):
    x = point[0]
    y = point[1]
    
    x0 = int(x)
    x1 = int(x+1)
    y0 = int(y)
    y1 = int(y+1)
    
    x0_weight = x-x0
    x1_weight = x1-x
    y0_weight = y-y0
    y1_weight = y1-y
    
    value = image[y0,x0]*y1_weight*x1_weight +\
            image[y1,x0]*y0_weight*x1_weight +\
            image[y1,x1]*y0_weight*x0_weight +\
            image[y0,x1]*y1_weight*x0_weight
    return value
    

def find_point_pairs(capture_folder):
    capture_list = os.listdir(capture_folder)
    obj_points = create_board_points()
    obj_points_list = []
    camera_points_list = []
    projector_points_list = []
    
    for sample in capture_list:
        sample_folder = os.path.join(capture_folder, sample)
        image_path = os.path.join(sample_folder, 'phase36.bmp')
        print(sample_folder)
        img = -cv2.imread(image_path, 0) + 255
        ret, corners = cv2.findCirclesGrid(img, (7,11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        # print(ret)
        corners = ellipse_center_correction(img, corners)
        camera_points_list.append(corners)
        camera_points = corners.squeeze()
        
        #print(camera_points)
        dmd_x, confidence = extract_dmd_coordinate(sample_folder, start_idx=0)
        dmd_y, confidence = extract_dmd_coordinate(sample_folder, start_idx=18)
        
        projector_points = []
        for camera_point in camera_points:
            projector_point = [interp2d(dmd_x, camera_point), interp2d(dmd_y, camera_point)]
            projector_points.append(projector_point)
        projector_points = np.array(projector_points)
        #print(projector_points)
        
        obj_points_list.append(obj_points)
        projector_points_list.append(projector_points)
    # print(projector_points_list)
        
    return np.float32(obj_points_list), camera_points_list, np.float32(projector_points_list)


def generate_scene_pointclouds_original(sample_folder, R, T, camera_mtx, camera_dist, projector_mtx, projector_dist):

    dmd_x, confidence_x = extract_dmd_coordinate(sample_folder, start_idx=0)
    # print(np.max(confidence_x))
    dmd_y, confidence_y = extract_dmd_coordinate(sample_folder, start_idx=18)
    # print(confidence_y.shape)

    camera_point_list = []
    projector_point_list = []

    for iy in trange(1200):
        for ix in range(1920):
            if confidence_x[iy, ix] >=0.2 and confidence_y[iy, ix] >=0.2:
                camera_point = np.array([ix, iy])
                projector_point = np.array([dmd_x[iy, ix], dmd_y[iy, ix]])
                camera_point_list.append(camera_point)
                projector_point_list.append(projector_point)
    
    camera_point_list = np.array(camera_point_list)
    projector_point_list = np.array(projector_point_list)

    points_3d, _, error_3d = triangulation(camera_point_list, projector_point_list, R, T, camera_mtx, camera_dist, 0, projector_mtx, projector_dist, 0)
  
    points_3d_filtered = []

    for i in trange(points_3d.shape[0]):
        if error_3d[i] < 8:
            if points_3d[i, 2] > 900 and points_3d[i, 2] < 1100:
                points_3d_filtered.append(points_3d[i])
             

    points_3d_filtered = np.array(points_3d_filtered).squeeze(1)

    np.savetxt("after_refine_2.xyz", points_3d_filtered)

    return 0

def plane_fitting(plane):

    A = np.ones((plane.shape[0], 3))
    A[:, 0] = plane[:, 0]
    A[:, 1] = plane[:, 1]
    B = np.zeros((plane.shape[0], 1))
    B = plane[:, 2]
    A_T = A.T
    A1 = np.dot(A_T,A)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2,A_T)
    X = np.dot(A3, B)

    return X

def generate_scene_pointclouds(sample_folder, R, T, camera_mtx, camera_dist, projector_mtx, projector_dist):

    dmd_x, confidence_x = extract_dmd_coordinate(sample_folder, start_idx=0)
    # print(np.max(confidence_x))
    dmd_y, confidence_y = extract_dmd_coordinate(sample_folder, start_idx=18)
    # print(confidence_y.shape)

    camera_point_list = []
    projector_point_list = []

    for iy in trange(1200):
        for ix in range(1920):
            if confidence_x[iy, ix] >=0.2 and confidence_y[iy, ix] >=0.2:
                camera_point = np.array([ix, iy])
                projector_point = np.array([dmd_x[iy, ix], dmd_y[iy, ix]])
                camera_point_list.append(camera_point)
                projector_point_list.append(projector_point)
    
    camera_point_list = np.array(camera_point_list)
    projector_point_list = np.array(projector_point_list)

    points_3d, _, error_3d = triangulation(np.float64(camera_point_list), projector_point_list, R, T, camera_mtx, camera_dist, 0, projector_mtx, projector_dist, 0)
    # print(points_3d.shape)
    # print(error_3d.shape)
    # print(np.max(error_3d))

    points_3d_filtered = []
    camera_point_list_filtered = []
    projector_point_list_filtered = []

    for i in trange(points_3d.shape[0]):
        if error_3d[i] < 8:
            if points_3d[i, 2] > 900 and points_3d[i, 2] < 1100:
                points_3d_filtered.append(points_3d[i])
                camera_point_list_filtered.append(camera_point_list[i])
                projector_point_list_filtered.append(projector_point_list[i])


    points_3d_filtered = np.array(points_3d_filtered).squeeze(1)
    camera_point_list_filtered = np.array(camera_point_list_filtered)
    projector_point_list_filtered = np.array(projector_point_list_filtered)
    # print(camera_point_list_filtered.shape)
    # print(points_3d_filtered.shape)
    X = plane_fitting(points_3d_filtered)
    np.savetxt("text_plane.xyz", points_3d_filtered)

    return camera_point_list_filtered, projector_point_list_filtered, X
    
def create_plane_inial_params(R, T, camera_mtx, camera_dist, projector_mtx, projector_dist):

    params = np.vstack((camera_mtx.reshape(-1,1), camera_dist.reshape(-1,1)))
    params = np.vstack((params, projector_mtx.reshape(-1,1)))
    params = np.vstack((params, projector_dist.reshape(-1,1)))
    params = np.vstack((params, R.reshape(-1,1)))
    params = np.vstack((params, T.reshape(-1,1)))
    params = params.squeeze(1)
    # print(params.shape)

    return params

def compute_plane_resiudal(params, camera_point_list_filtered, projector_point_list_filtered, X):

    random_index = np.random.randint(0, camera_point_list_filtered.shape[0], 50000)
    camera_mtx = np.array(params[:9]).reshape(3, 3)
    camera_dist = np.array(params[9:14])
    projector_mtx = np.array(params[14:23]).reshape(3, 3)
    projector_dist = np.array(params[23:28])
    R = np.array(params[28:37]).reshape(3, 3)
    T = np.array(params[37:]).reshape(3, 1)
    plane, _, error_3d = triangulation(camera_point_list_filtered[random_index], projector_point_list_filtered[random_index], R, T, camera_mtx, camera_dist, 0, projector_mtx, projector_dist, 0)
    plane = np.array(plane)
    # print(plane.shape)
    fitting = []
    for i in range(plane.shape[0]):
        pts = np.array([plane[i, 0], plane[i, 1], X[0] * plane[i, 0] + X[1] * plane[i, 1] + X[2]])
        fitting.append(pts)

    fitting = np.array(fitting)
    print("Plane Loss:", cv2.norm(fitting, plane, cv2.NORM_L2))
    plane_residual = plane.ravel() - fitting.ravel()

    return plane_residual

def LM_plane_refine(residual_function, init_plane_params, camera_point_list_filtered, projector_point_list_filtered, X):

    sol = optimize.least_squares(residual_function, init_plane_params, args=(camera_point_list_filtered, projector_point_list_filtered, X), xtol=1e-15, ftol=1e-15)
    camera_mtx = np.array(sol.x[:9]).reshape(3, 3)
    camera_dist = np.array(sol.x[9:14])
    projector_mtx = np.array(sol.x[14:23]).reshape(3, 3)
    projector_dist = np.array(sol.x[23:28])
    R = np.array(sol.x[28:37]).reshape(3, 3)
    T = np.array(sol.x[37:]).reshape(3, 1)

    return camera_mtx, camera_dist, projector_mtx, projector_dist, R, T


def stereo_calibration(capture_folder):
    obj_points_list, camera_points_list, projector_points_list = find_point_pairs(capture_folder)
    print("Object points list shape:", obj_points_list.shape)
    # print("Camera points list shape:", camera_points_list.shape)
    # print("Projector points list shape:", projector_points_list.shape)

    ret, camera_mtx, camera_dist, camera_rvecs, camera_tvecs \
    = cv2.calibrateCamera(obj_points_list, camera_points_list, (1920,1200), None, None)
    
    camera_dist = np.squeeze(camera_dist)
    print("----------------------------------------")
    print('camera params')
    print('ret', ret)
    print('mtx', camera_mtx)
    print('dist', camera_dist)
    # print('rvecs shape',np.array(camera_rvecs).shape)
    # print('tvecs shape',np.array(camera_tvecs).shape)
    
    # camera_init_params = create_inial_params(camera_mtx, camera_dist, camera_rvecs, camera_tvecs, obj_points_list.shape[0])
    # # compute_residuals(camera_init_params, obj_points_list, camera_points_list, re_error=True)
    # # compute_residuals_3d_mono(camera_init_params, obj_points_list, camera_points_list, re_error=False)
    # camera_mtx_f, camera_dist_f, camera_rvecs_f, camera_tvecs_f = LM_refine(compute_residuals_3d_mono, camera_init_params, obj_points_list, camera_points_list, re_error=False)

    # print('----------After LM Refine-----------')
    # print('camera params')
    # print('mtx_f', camera_mtx_f)
    # print('dist_f', camera_dist_f)
    # print('rvecs_f shape',np.array(camera_rvecs_f).shape)
    # print('tvecs_f shape',np.array(camera_tvecs_f).shape)
    # compute_residuals(create_inial_params(camera_mtx_f, camera_dist_f, camera_rvecs_f, camera_tvecs_f, obj_points_list.shape[0]), obj_points_list, camera_points_list, re_error=True)
    # compute_residuals_3d_mono(create_inial_params(camera_mtx_f, camera_dist_f, camera_rvecs_f, camera_tvecs_f, obj_points_list.shape[0]), obj_points_list, camera_points_list, re_error=False)
    print("----------------------------------------")

    ret, projector_mtx, projector_dist, projector_rvecs, projector_tvecs \
    = cv2.calibrateCamera(obj_points_list, projector_points_list, (1280,720), None, None)
    
    projector_dist = np.squeeze(projector_dist)

    print('projector params')
    print('ret', ret)
    print('mtx', projector_mtx)
    print('dist', projector_dist)
    # print('rvecs shape',np.array(projector_rvecs).shape)
    # print('tvecs shape',np.array(projector_tvecs).shape)

    # projector_init_params = create_inial_params(projector_mtx, projector_dist, projector_rvecs, projector_tvecs, obj_points_list.shape[0])
    # # compute_residuals_3d_mono(projector_init_params, obj_points_list, np.expand_dims(projector_points_list, axis=1), re_error=False)
    # # compute_residuals(projector_init_params, obj_points_list, np.expand_dims(projector_points_list, axis=1), re_error=True)
    # projector_mtx_f, projector_dist_f, projector_rvecs_f, projector_tvecs_f = LM_refine(compute_residuals_3d_mono, projector_init_params, obj_points_list, np.expand_dims(projector_points_list, axis=1), re_error=False)
    # print('----------After LM Refine-----------')
    # print('projector params')
    # print('mtx_f', projector_mtx_f)
    # print('dist_f', projector_dist_f)
    # print('rvecs_f shape',np.array(projector_rvecs_f).shape)
    # print('tvecs_f shape',np.array(projector_tvecs_f).shape)
    # compute_residuals(create_inial_params(projector_mtx_f, projector_dist_f, projector_rvecs_f, projector_tvecs_f, obj_points_list.shape[0]), obj_points_list, np.expand_dims(projector_points_list, axis=1), re_error=True)
    # compute_residuals_3d_mono(create_inial_params(projector_mtx_f, projector_dist_f, projector_rvecs_f, projector_tvecs_f, obj_points_list.shape[0]), obj_points_list, np.expand_dims(projector_points_list, axis=1), re_error=False)
    print("----------------------------------------")

    stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 500, 1e-15)
    # ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(obj_points_list, camera_points_list, projector_points_list, camera_mtx, camera_dist, projector_mtx, projector_dist, imageSize=(1920,1200), criteria=stereocalib_criteria)
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(obj_points_list, camera_points_list, projector_points_list, camera_mtx, camera_dist, projector_mtx, projector_dist, imageSize=(1920,1200), criteria=stereocalib_criteria)

    print('stereo params')
    print('ret',ret)
    print('M1',M1)
    print('d1',d1)
    print('M2',M2)
    print('d2',d2)
    print('R',R)
    print('T',T)
    
    # stereo_initial_params = create_stereo_inial_params(R, T)
    # # compute_stereo_residual(stereo_initial_params, obj_points_list, camera_points_list, np.expand_dims(projector_points_list, axis=1), \
    # #     camera_rvecs_f, camera_tvecs_f, camera_mtx_f, camera_dist_f, \
    # #     projector_rvecs_f, projector_tvecs_f, projector_mtx_f, projector_dist_f, re_error=True)
    Reconstruction_Error(obj_points_list, camera_rvecs, camera_tvecs, camera_mtx, camera_dist, projector_rvecs, projector_tvecs, projector_mtx, projector_dist, R, T)

    # R_f, T_f = LM_stereo_refine(compute_3d_residual, stereo_initial_params, obj_points_list, camera_points_list, np.expand_dims(projector_points_list, axis=1), \
    #     camera_rvecs_f, camera_tvecs_f, camera_mtx_f, camera_dist_f, \
    #     projector_rvecs_f, projector_tvecs_f, projector_mtx_f, projector_dist_f)

    # print("----------After LM Refine----------")
    # print('R_f:', R_f)
    # print('T_f:', T_f)
    # compute_stereo_residual(create_stereo_inial_params(R_f, T_f), obj_points_list, camera_points_list, np.expand_dims(projector_points_list, axis=1), \
    #     camera_rvecs, camera_tvecs, camera_mtx, camera_dist, \
    #     projector_rvecs, projector_tvecs, projector_mtx, projector_dist, re_error=True)
    # Reconstruction_Error(obj_points_list, camera_rvecs_f, camera_tvecs_f, camera_mtx_f, camera_dist_f, projector_rvecs_f, projector_tvecs_f, projector_mtx_f, projector_dist_f, R_f, T_f)
    print("----------------------------------------")

    
    camera_point_list_filtered, projector_point_list_filtered, X = generate_scene_pointclouds("3d_camera_calibration_test_data/01", R, T, camera_mtx, camera_dist, projector_mtx, projector_dist)
    plane_inial_params = create_plane_inial_params(R, T, camera_mtx, camera_dist, projector_mtx, projector_dist)
    # compute_plane_resiudal(plane_inial_params, camera_point_list_filtered, projector_point_list_filtered)
    camera_mtx_p, camera_dist_p, projector_mtx_p, projector_dist_p, R_p, T_p = LM_plane_refine(compute_plane_resiudal, plane_inial_params, camera_point_list_filtered, projector_point_list_filtered, X)
    print("camera_mtx_p:", camera_mtx_p)
    print("camera_dist_p:", camera_dist_p)
    print("projector_mtx_p:", projector_mtx_p)
    print("projector_dist_p:", projector_dist_p)
    print("R_p:", R_p)
    print("T_p:", T_p)
    Reconstruction_Error(obj_points_list, camera_rvecs, camera_tvecs, camera_mtx_p, camera_dist_p, projector_rvecs, projector_tvecs, projector_mtx_p, projector_dist_p, R_p, T_p)
    generate_scene_pointclouds_original("3d_camera_calibration_test_data/01", R_p, T_p, camera_mtx_p, camera_dist_p, projector_mtx_p, projector_dist_p)

    # return ret, \
    #        camera_mtx_f, camera_dist_f, camera_rvecs_f, camera_tvecs_f, \
    #        projector_mtx_f, projector_dist_f, projector_rvecs_f, projector_tvecs_f, \
    #        R_f, T_f, \
    #        obj_points_list, camera_points_list, projector_points_list

    return ret, \
           camera_mtx, camera_dist, camera_rvecs, camera_tvecs, \
           projector_mtx, projector_dist, projector_rvecs, projector_tvecs, \
           R, T, \
           obj_points_list, camera_points_list, projector_points_list

if __name__=='__main__':

    ret, \
    camera_mtx, camera_dist, camera_rvecs, camera_tvecs, \
    projector_mtx, projector_dist, projector_rvecs, projector_tvecs, \
    R, T, \
    obj_points_list, camera_points_list, projector_points_list \
    = stereo_calibration("3d_camera_calibration_test_data/")
    
    # params = np.vstack((camera_mtx.reshape(-1,1), camera_dist.reshape(-1,1)))
    # params = np.vstack((params, projector_mtx.reshape(-1,1)))
    # params = np.vstack((params, projector_dist.reshape(-1,1)))
    # params = np.vstack((params, R.reshape(-1,1)))
    # params = np.vstack((params, T.reshape(-1,1)))
    # print(params.shape)
    # np.savetxt("param_o_20220118.txt", params)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # draw_camera(ax, camera_mtx, color='blue', h_size=1920)
    
    # rvec, _ = cv2.Rodrigues(R)
    # draw_camera(ax, projector_mtx, rvec=rvec, tvec=T, color='red', h_size=1280)
    
    # colors = "bgrcmyk"
    # for i in range(16):
    #     color = colors[i%7]
    #     camera_point = camera_points_list[i].reshape([-1,2])
    #     projector_piont = projector_points_list[i].reshape([-1,2])
    #     point3d, _, _ = triangulation(camera_point, projector_piont, 
    #                                   R,T,
    #                                   camera_mtx, camera_dist, 0,
    #                                   projector_mtx, projector_dist, 0)

    #     ax.scatter3D(point3d[:,0], point3d[:,1], point3d[:,2], color=color)
    #     rvecs = np.squeeze(camera_rvecs[i])
    #     tvecs = np.squeeze(camera_tvecs[i])
        
    #     draw_board(ax, rvecs, tvecs, color)
    
    # set_axes_equal(ax)
    # plt.show()