import cv2
import os
import math
import numpy as np

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
    
    

if __name__ == '__main__':
    capture_folder = 'ellipse_center_correction_test_data'
    capture_list = os.listdir(capture_folder)
    for sample in capture_list:
        image_path = os.path.join(capture_folder, sample)
        #image_path = os.path.join(sample_folder, 'phase36.bmp')
        print(image_path)
        
        img = -cv2.imread(image_path, 0) + 255
        ret, corners = cv2.findCirclesGrid(img, (7,11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        corners = np.squeeze(corners)
        print(ret)

        ellipse_corners = ellipse_center_correction(img, corners)
        ellipse_corners = np.squeeze(ellipse_corners)
        error = ellipse_corners - corners
        
        print(error)
        
        for i in range(len(corners)):
            corner = tuple(np.int32(corners[i]))
            ellipse = tuple(np.int32(corners[i]+error[i]*1000))
            #print(ellipse)
            
            cv2.line(img, corner, ellipse, (0), 1)
        
        p1 = tuple(np.int32(np.squeeze(corners[0])))
        p2 = tuple(np.int32(np.squeeze(corners[6])))
        p3 = tuple(np.int32(np.squeeze(corners[70])))
        p4 = tuple(np.int32(np.squeeze(corners[76])))
        
        print(p1)
        
        cv2.circle(img, p1, 20, (0), 2)
        cv2.circle(img, p2, 20, (0), 2)
        cv2.circle(img, p3, 20, (0), 2)
        cv2.circle(img, p4, 20, (0), 2)
        
        print(p1)
        img = cv2.drawChessboardCorners(img, (7,11), corners,ret)
        cv2.namedWindow('show', 0)
        cv2.imshow('show', img)
        cv2.waitKey()