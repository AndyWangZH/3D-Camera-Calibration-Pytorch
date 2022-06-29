import cv2
import numpy as np
import os
import math
import torch

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
    
def ellipse_center_correction(img, corners):
    #print(corners)
    p1 = np.squeeze(corners[0])
    p2 = np.squeeze(corners[6])
    p3 = np.squeeze(corners[70])
    p4 = np.squeeze(corners[76])
    
    p1_ = (200, 200)
    p2_ = (800, 200)
    p3_ = (200, 700)
    p4_ = (800, 700)
    
    P = np.float32([p1, p2, p3, p4])
    P_ = np.float32([p1_, p2_, p3_, p4_])
    
    #print(P,P_)
    
    M = cv2.getPerspectiveTransform(P, P_)
    M_ = cv2.getPerspectiveTransform(P_, P)
    warp = cv2.warpPerspective(img, M, (1920,1400))
    
    ret, warp_corners = cv2.findCirclesGrid(warp, (7,11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    warp = cv2.drawChessboardCorners(warp, (7,11), warp_corners,ret)
    warp_corners = np.squeeze(warp_corners)
    
    warp_p1 = warp_corners[0]
    warp_p2 = warp_corners[6]
    warp_p3 = warp_corners[70]
    warp_p4 = warp_corners[76]
    warp_P = np.float32([[warp_p1, warp_p2, warp_p3, warp_p4]])
    #print(warp_P)
    #print(M)
    #print(warp_corners.shape)
    
    ellipse_centers = cv2.perspectiveTransform(np.float32([warp_corners]), M_)
    #print('ellipse_centers', ellipse_centers)
    
    #cv2.namedWindow('warp', 0)
    #cv2.imshow('warp', warp)
    
    return ellipse_centers

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

def undistortPoints_pytorch_batch(points, intrinsic, distortion):

    assert points.shape[2] == 2

    cc_x = intrinsic[0,2]
    cc_y = intrinsic[1,2]
    fc_x = intrinsic[0,0]
    fc_y = intrinsic[1,1]
    k1 = distortion[0]
    k2 = distortion[1]
    k3 = distortion[4]
    p1 = distortion[2]
    p2 = distortion[3]

    points_distort = torch.zeros_like(points)
    points_distort[:, :, 0] = (points[:, :, 0] - cc_x) / fc_x
    points_distort[:, :, 1] = (points[:, :, 1] - cc_y) / fc_y

    points_iter = points_distort.clone()

    for _ in range(8):
        r_2 = (points_iter[:, :, 0].clone())**2 + (points_iter[:, :, 1].clone())**2
        # print(r_2.shape)
        k_radial =  1 + k1 * r_2.clone() + k2 * (r_2.clone())**2 + k3 * (r_2.clone())**3
        points_delta = torch.zeros_like(points_iter)
        points_delta[:, :, 0] = 2 * p1 * points_iter[:, :, 0].clone() * points_iter[:, :, 1].clone() + p2 * (r_2[:, :].clone() + 2 * (points_iter[:, :, 0].clone())**2)
        points_delta[:, :, 1] = p1 * (r_2[:, :].clone() + 2 * (points_iter[:, :, 1].clone())**2) + 2 * p2 * points_iter[:, :, 0].clone() * points_iter[:, :, 1].clone()
        points_iter[:, :, 0] = (points_distort[:, :, 0].clone() - points_delta[:, :, 0].clone())/ k_radial.clone()
        points_iter[:, :, 1] = (points_distort[:, :, 1].clone() - points_delta[:, :, 1].clone()) / k_radial.clone()

    return points_iter




def undistortPoints_pytorch(points, intrinsic, distortion):
    
    assert points.shape[1]==2
    cc_x = intrinsic[0,2]
    cc_y = intrinsic[1,2]
    fc_x = intrinsic[0,0]
    fc_y = intrinsic[1,1]
    k1 = distortion[0]
    k2 = distortion[1]
    k3 = distortion[4]
    p1 = distortion[2]
    p2 = distortion[3]

    points_distort = torch.zeros_like(points)
    points_distort[:, 0] = (points[:, 0] - cc_x) / fc_x
    points_distort[:, 1] = (points[:, 1] - cc_y) / fc_y
    # print("Torch distort:", points_distort[0, 0], points_distort[0, 1])

    points_iter = points_distort.clone()

    for _ in range(8):
        r_2 = (points_iter[:, 0].clone())**2 + (points_iter[:, 1].clone())**2
        # print(r_2.shape)
        k_radial =  1 + k1 * r_2.clone() + k2 * (r_2.clone())**2 + k3 * (r_2.clone())**3
        points_delta = torch.zeros_like(points_iter)
        points_delta[:, 0] = 2 * p1 * points_iter[:, 0].clone() * points_iter[:, 1].clone() + p2 * (r_2[:].clone() + 2 * (points_iter[:, 0].clone())**2)
        points_delta[:, 1] = p1 * (r_2[:].clone() + 2 * (points_iter[:, 1].clone())**2) + 2 * p2 * points_iter[:, 0].clone() * points_iter[:, 1].clone()
        points_iter[:, 0] = (points_distort[:, 0].clone() - points_delta[:, 0].clone())/ k_radial.clone()
        points_iter[:, 1] = (points_distort[:, 1].clone() - points_delta[:, 1].clone()) / k_radial.clone()
    # points_undistort = torch.zeros_like(points_iter)
    # points_undistort[:, 0] = (points_distort[:, 0] - points_delta[:, 0])/ k_radial
    # points_undistort[:, 1] = (points_distort[:, 1] - points_delta[:, 1]) / k_radial
    
    return points_iter

def triangulation_pytorch_batch(xL, xR, R, T, left_intrinsic, kc_left, right_intrinsic, kc_right):

    assert xL.shape == xR.shape
    assert xL.shape[2] == 2

    B = xL.shape[0]
    N = xL.shape[1]

    xt_u = undistortPoints_pytorch_batch(xL, left_intrinsic, kc_left)
    xtt_u = undistortPoints_pytorch_batch(xR, right_intrinsic, kc_right)
    # print(torch.cat((xt_u, torch.ones([B, N, 1], dtype=torch.double).cuda()), dim=2).T.shape)
    xt = torch.cat((xt_u, torch.ones([B, N, 1], dtype=torch.double).cuda()), dim=2).T.permute(2, 0, 1)
    xtt = torch.cat((xtt_u, torch.ones([B, N, 1], dtype=torch.double).cuda()), dim=2).T.permute(2, 0, 1)
    u = torch.bmm(R.unsqueeze(0).repeat(B, 1, 1), xt)

    n_xt2 = torch.sum(torch.pow(xt, 2), 1)
    n_xtt2 = torch.sum(torch.pow(xtt, 2), 1)

    DD = torch.mul(n_xt2, n_xtt2) - torch.pow(torch.sum(torch.mul(u, xtt), 1), 2)

    dot_uT = torch.sum(torch.mul(u, T), 1)
    dot_xttT = torch.sum(torch.mul(xtt, T), 1)
    dot_xttu = torch.sum(torch.mul(u, xtt), 1)
    
    NN1 = torch.mul(dot_xttu,dot_xttT) - torch.mul(n_xtt2,dot_uT)
    NN2 = torch.mul(n_xt2,dot_xttT) - torch.mul(dot_uT,dot_xttu)

    Zt = torch.div(NN1,DD)
    Ztt = torch.div(NN2,DD)
    # print(Zt)
    
    # print(xt.shape)
    # print(Zt.shape)
    X1 = torch.mul(xt, Zt.unsqueeze(1).repeat(1, 3, 1))
    # print(X1)
    X2 = torch.bmm(R.T.unsqueeze(0).repeat(B, 1, 1), (torch.mul(xtt,Ztt.unsqueeze(1).repeat(1, 3, 1)) - T))
    # print(X2)

    XL = (X1 + X2) / 2.0
    XR = torch.bmm(R.unsqueeze(0).repeat(B, 1, 1), XL) + T
    error = torch.sqrt(torch.sum(torch.pow(X1-X2, 2), 1))
    # print(error.shape)
    
    return XL.transpose(1,2), error.T


def triangulation_pytorch(xL, xR, R, T, left_intrinsic, kc_left, right_intrinsic, kc_right):

    assert xL.shape == xR.shape
    assert xL.shape[1] == 2

    N = xL.shape[0] 

    xt_u = undistortPoints_pytorch(xL, left_intrinsic, kc_left)
    # print("Original:", xt_u[0, :])
    xtt_u = undistortPoints_pytorch(xR, right_intrinsic, kc_right)

    xt = torch.cat((xt_u, torch.ones([N, 1], dtype=torch.double).cuda()), dim=1).T
    xtt = torch.cat((xtt_u, torch.ones([N, 1], dtype=torch.double).cuda()), dim=1).T
    # print(xt[:, 0])
    u = torch.mm(R, xt)
    # print(u)

    n_xt2 = torch.sum(torch.pow(xt, 2), 0)
    n_xtt2 = torch.sum(torch.pow(xtt, 2), 0)
    # print(n_xt2)

    DD = torch.mul(n_xt2, n_xtt2) - torch.pow(torch.sum(torch.mul(u, xtt), 0), 2)

    dot_uT = torch.sum(torch.mul(u, T), 0)
    dot_xttT = torch.sum(torch.mul(xtt, T), 0)
    dot_xttu = torch.sum(torch.mul(u, xtt), 0)
    # print(dot_xttu)

    NN1 = torch.mul(dot_xttu,dot_xttT) - torch.mul(n_xtt2,dot_uT)
    NN2 = torch.mul(n_xt2,dot_xttT) - torch.mul(dot_uT,dot_xttu)

    Zt = torch.div(NN1,DD)
    Ztt = torch.div(NN2,DD)
    # print(Zt)
    # print(xt.shape)
    # print(Zt.shape)
    X1 = torch.mul(xt, Zt)
    # print(X1)
    X2 = torch.mm(R.T, (torch.mul(xtt,Ztt) - T))
    # print(X2)

    XL = (X1 + X2) / 2.0
    XR = torch.mm(R, XL) + T

    error = torch.sqrt(torch.sum(torch.pow(X1-X2, 2), 0))
    # print(error)

    return XL.T, XR.T, error.T

def plane_fitting_torch(points):

    A = torch.ones_like(points, dtype=torch.double).cuda()
    A[:, 0] = points[:, 0]
    A[:, 1] = points[:, 1]
    B = torch.zeros((points.shape[0], 1), dtype=torch.double).cuda()
    B[:, 0] = points[:, 2]
    A_T = A.T
    A1 = torch.mm(A_T, A)
    # print(A1)
    A2 = torch.inverse(A1)
    A3 = torch.mm(A2, A_T)
    X = torch.mm(A3, B)

    return X

if __name__ == "__main__":

    torch.set_printoptions(precision=8)
    
    xL = torch.from_numpy(np.array([[510.6646, 357.2370], [510.6646, 357.2370]])).cuda()
    xR = torch.from_numpy(np.array([[629.7592, 169.4815], [629.7592, 169.4815]])).cuda()
    
    fc_left = np.array([967.5319, 968.5227])
    cc_left = np.array([646.8668, 348.3653])
    left_intrinsic = torch.from_numpy(np.array(  [[fc_left[0], 0,          cc_left[0]],
                                [0,          fc_left[1], cc_left[1]],
                                [0,          0,          1         ]])).cuda()

    #                     k1       k2       p1       p2      k3(not used)
    kc_left = torch.from_numpy(np.array([0.1604, -0.5732, -0.0015, -0.0006, 0.7387])).cuda()
    
    fc_right = np.array([930.4610, 930.2560])
    cc_right = np.array([628.7786, 320.8216])
    right_intrinsic = torch.from_numpy(np.array(  [[fc_right[0], 0,           cc_right[0]],
                                 [0,            fc_right[1], cc_right[1]],
                                 [0,            0,           1         ]])).cuda()

    kc_right = torch.from_numpy(np.array([-0.4524, 0.2999, 0.0003, -0.0004, -0.1418])).cuda() 

    om = np.array([[0.0925],
                   [0.5269],
                   [0.0275]])              
    R, _ = cv2.Rodrigues(om)
    R = torch.from_numpy(R).cuda()
    T =  torch.from_numpy(np.array( [[-230.7309],
                    [-52.8380],
                    [46.5424]])).cuda()
    
    points_3d, _, error = triangulation_pytorch(xL, xR, R, T, left_intrinsic, kc_left, right_intrinsic, kc_right)
    # print(points_3d)

    points_3d_btach, error_batch = triangulation_pytorch_batch(xL.unsqueeze(0).repeat(2,1,1), xR.unsqueeze(0).repeat(2,1,1), R, T, left_intrinsic, kc_left, right_intrinsic, kc_right)
    print(points_3d_btach.shape)
    print(error_batch.shape)