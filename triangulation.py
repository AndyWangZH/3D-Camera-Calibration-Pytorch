#coding=utf-8
import numpy as np
import cv2

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

    points_distort = np.zeros_like(points)
    points_distort[:, 0] = (points[:, 0] - cc_x) / fc_x
    points_distort[:, 1] = (points[:, 1] - cc_y) / fc_y
    # print("Torch distort:", points_distort[0, 0], points_distort[0, 1])

    points_iter = points_distort.copy()

    for _ in range(20):
        r_2 = points_iter[:, 0]**2 + points_iter[:, 1]**2
        k_radial =  1 + k1 * r_2 + k2 * r_2**2 + k3 * r_2**3
        points_delta = np.zeros_like(points_iter)
        points_delta[:, 0] = 2 * p1 * points_iter[:, 0] * points_iter[:, 1] + p2 * (r_2[:] + 2 * points_iter[:, 0]**2)
        points_delta[:, 1] = p1 * (r_2[:] + 2 * points_iter[:, 1]**2) + 2 * p2 * points_iter[:, 0] * points_iter[:, 1]
        points_iter[:, 0] = (points_distort[:, 0] - points_delta[:, 0])/ k_radial
        points_iter[:, 1] = (points_distort[:, 1] - points_delta[:, 1]) / k_radial
    
    # print("Pytorch:", points_iter)
    
    return points_iter




def undistortPoints(points, intrinsic, distortion):
    assert points.shape[1]==2
    n = points.shape[0]
    cc_x = intrinsic[0,2]
    cc_y = intrinsic[1,2]
    fc_x = intrinsic[0,0]
    fc_y = intrinsic[1,1]
    k1 = distortion[0]
    k2 = distortion[1]
    k3 = distortion[4]
    p1 = distortion[2]
    p2 = distortion[3]

    undistort_points = []
    for i in range(n):
        x,y = points[i]
        x_distort = (x-cc_x)/fc_x
        y_distort = (y-cc_y)/fc_y
        
        x_iter = x_distort
        y_iter = y_distort
        for j in range(20):
            r_2 = x_iter**2 + y_iter**2
            k_radial =  1 + k1 * r_2 + k2 * r_2**2 + k3 * r_2**3
            delta_x = 2*p1*x_iter*y_iter + p2*(r_2 + 2*x_iter**2)
            delta_y = p1 * (r_2 + 2*y_iter**2) + 2*p2*x_iter*y_iter
            x_iter = (x_distort - delta_x)/k_radial
            y_iter = (y_distort - delta_y)/k_radial
        undistort_points.append([x_iter, y_iter])
    return np.array(undistort_points)
    
def triangulation(xL, xR, 
                         R,T,
                         left_intrinsic,  kc_left,  alpha_c_left,
                         right_intrinsic, kc_right, alpha_c_right):
                         
    assert xL.shape == xR.shape
    assert xL.shape[1] == 2
                         
    N = xL.shape[0] 

    #Question found: Too slow for manual undistortPoints

    xt = cv2.undistortPoints(xL, left_intrinsic, kc_left).reshape([-1,2])
    # print("CV:", xt_cv)
    # xt_2 = undistortPoints(xL, left_intrinsic, kc_left).reshape([-1,2])
    # print("Manual:", xt_2)
    # xt = undistortPoints_pytorch(xL, left_intrinsic, kc_left)
    # print("Torch:", xt)
    
    xtt = cv2.undistortPoints(xR, right_intrinsic, kc_right).reshape([-1,2])
    # xtt = undistortPoints(xR, right_intrinsic, kc_right).reshape([-1,2])
    
    # 转换为齐次坐标
    xt = np.hstack((xt,np.ones([N, 1]))).T
    xtt = np.hstack((xtt,np.ones([N, 1]))).T
    # print(xt)

    u = np.matrix(R) * np.matrix(xt)
    # print(u)

    n_xt2 = np.sum(np.power(xt,2), 0)
    # print(n_xt2)
    n_xtt2 = np.sum(np.power(xtt,2), 0)
    
    DD = np.multiply(n_xt2,n_xtt2) - np.power(np.sum(np.multiply(u,xtt),0),2)
    # print(DD)

    dot_uT = np.sum(np.multiply(u,T),0)
    dot_xttT = np.sum(np.multiply(xtt,T),0)
    dot_xttu = np.sum(np.multiply(u,xtt),0)
    # print(dot_xttu)

    NN1 = np.multiply(dot_xttu,dot_xttT) - np.multiply(n_xtt2,dot_uT)
    NN2 = np.multiply(n_xt2,dot_xttT) - np.multiply(dot_uT,dot_xttu)
    
    Zt = np.divide(NN1,DD)
    Ztt = np.divide(NN2,DD)
    # print(Zt)

    X1 = np.multiply(xt, Zt)
    # print(X1)
    X2 = R.T * (np.multiply(xtt,Ztt) - T)
    # print(X2)
    XL = (X1 + X2) / 2.0
    
    XR = R*XL + T #这一句解释了左右摄像机之间的关系

    # Error = np.mean(np.sqrt(np.sum(np.power(X1-X2, 2),0)))
    Error = np.sqrt(np.sum(np.power(X1-X2, 2),0))
    
    return XL.T, XR.T, Error.T


if __name__ == '__main__':
    xL = np.array([[510.6646, 357.2370], [510.6646, 357.2370]])
    xR = np.array([[629.7592, 169.4815], [629.7592, 169.4815]])
    
    om = np.array([[0.0925],
                   [0.5269],
                   [0.0275]])
                   
    R, _ = cv2.Rodrigues(om)
                   
    T = np.array( [[-230.7309],
                    [-52.8380],
                    [46.5424]])
                    
    fc_left = np.array([967.5319, 968.5227])

    cc_left = np.array([646.8668, 348.3653])
    
    left_intrinsic = np.array(  [[fc_left[0], 0,          cc_left[0]],
                                [0,          fc_left[1], cc_left[1]],
                                [0,          0,          1         ]])

    #                     k1       k2       p1       p2      k3(not used)
    kc_left = np.array([0.1604, -0.5732, -0.0015, -0.0006, 0.7387])
    
    alpha_c_left = 0

    fc_right = np.array([930.4610, 930.2560])

    cc_right = np.array([628.7786, 320.8216])
    
    right_intrinsic = np.array(  [[fc_right[0], 0,           cc_right[0]],
                                 [0,            fc_right[1], cc_right[1]],
                                 [0,            0,           1         ]])

    kc_right = np.array([-0.4524, 0.2999, 0.0003, -0.0004, -0.1418])
    
    alpha_c_right = 0
                    
    XL_target = np.array([[-85.0024, 5.5794, 605.6999],[-85.0024, 5.5794, 605.6999]])
    
    XL, XR, ERR = triangulation( xL, xR, 
                                        R,T,
                                        left_intrinsic,  kc_left,  alpha_c_left,
                                        right_intrinsic, kc_right, alpha_c_right)

    print(XL)
    error = np.linalg.norm(XL-XL_target)
    assert error< 0.05
    print('Unit test passed!')
    