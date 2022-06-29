import cv2
import numpy as np


def gray2rgb(gray, gray_min, gray_max):
    n = (gray-gray_min) / (gray_max-gray_min)
    if n<0: return (255,0,0)
    elif n>1: return (0,0,255)
    else:
        b = (0 if n>0.5 else (1-2*n)*256) - 0.0001
        g = (2*n*256 if n<0.5 else ((1-n)*2) * 256) - 0.4999
        r = (0 if n<0.5 else (2*n-1) * 256) - 0.0001
        return (int(b), int(g), int(r))
        
def gray2rgb2(gray, gray_min, gray_max):
    n = (gray-gray_min) / (gray_max-gray_min)
    if n<0: n=0
    if n>1: n=1
    color_vector = [-255/4.0, -255/4.0, 255/2.0]
    rotation_vector = [.57, .57, .57]
    
    
def save_pseudo_color(filename, gray_img, gray_min=0.0, gray_max=1.0):
    img_size = gray_img.shape
    assert np.array(img_size).shape[0] == 2
    pseudo_color = np.zeros([img_size[0],img_size[1],3], dtype=float)
    for y in range(img_size[0]):
        for x in range(img_size[1]):
            pseudo_color[y,x,:] = gray2rgb(gray_img[y,x], gray_min, gray_max)
    cv2.imwrite(filename, pseudo_color)
    
if __name__ == '__main__':
    img = np.zeros([100,2048], dtype=float)
    for i in range(2048):
        a = i/2048
        img[:, i] = a
        
    img_gray = img*256
    cv2.imwrite('gray.bmp', img_gray)
    save_pseudo_color('pseudo_color.bmp', img, gray_min=0.0, gray_max=1.0)