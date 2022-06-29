import cv2
import numpy as np
import torch
from tqdm import trange, tqdm

def FarthestPointSampleTorch(xyz, nsamples):   # cuda needed
    """
    Input:
        xyz: pointcloud data, [N, D],
        nsamples: number of samples
    Return:
        sampled points
    """
    print("Farthest Point Sampling By Torch")
    print("Desired Sample Nums:", nsamples)
    xyz = torch.from_numpy(xyz[:,:3]).type(torch.float32).cuda('cuda:0')
    N, _ = xyz.shape 
    centroids = torch.zeros((nsamples,), dtype=torch.long).cuda()
    distance = torch.ones((N,), dtype=torch.float32).cuda() * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long).cuda()
    for i in trange(nsamples):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, axis=-1)[1]
    pcd = np.array(xyz[centroids,:].cpu())
    del xyz
    del distance
    del farthest
    return pcd  



if __name__ == "__main__":

    plane = np.loadtxt("text_plane.xyz")
    # plane_fps = FarthestPointSampleTorch(plane, 5000)
    # plane_fps = np.loadtxt("plane_fps.xyz")
    # np.savetxt("plane_fps.xyz", plane_fps)

    A = np.ones((plane.shape[0], 3))
    # print(plane[:, 0].shape)
    A[:, 0] = plane[:, 0]
    A[:, 1] = plane[:, 1]

    B = np.zeros((plane.shape[0], 1))
    B = plane[:, 2]

    A_T = A.T
    A1 = np.dot(A_T,A)
    print(A1)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2,A_T)
    # print(A3.shape)
    # print(B.shape)
    X = np.dot(A3, B)
    
    print(X)
 
    #计算方差
    # R=0
    # pts_list = []
    # for i in trange(plane.shape[0]):
    #     # R=R+(X[0, 0] * plane_fps[i, 0] + X[1, 0] * plane_fps[i, 1] + X[2, 0] - plane_fps[i, 2])**2
    #     pts = np.array([plane[i, 0], plane[i, 1], X[0] * plane[i, 0] + X[1] * plane[i, 1] + X[2]])
    #     # print("fitting:", pts)
    #     # print("original:", plane_fps[i])
    #     pts_list.append(pts)

    # pts_list = np.array(pts_list)
    # np.savetxt("plane_fitting_nofps.xyz", pts_list)
    # print ('方差为：%.*f'%(3,R))
