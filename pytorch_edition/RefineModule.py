import torch
import torch.nn as nn
from utils import triangulation_pytorch, undistortPoints_pytorch, triangulation_pytorch_batch, undistortPoints_pytorch_batch

class PlaneRefine(nn.Module):
    def __init__(self, R, T, camera_mtx, camera_dist, projector_mtx, projector_dist):
        super(PlaneRefine, self).__init__()
        self.R = nn.Parameter(R)
        self.T = nn.Parameter(T)
        self.camera_mtx = nn.Parameter(camera_mtx)
        self.camera_dist = nn.Parameter(camera_dist)
        self.projector_mtx = nn.Parameter(projector_mtx)
        self.projector_dist = nn.Parameter(projector_dist)

    def forward(self, camera_points, projector_points):
        pts, _, pts_error = triangulation_pytorch(camera_points, projector_points, self.R, self.T, self.camera_mtx, self.camera_dist, self.projector_mtx, self.projector_dist)

        return pts, pts_error

class PlaneRefineBatch(nn.Module):
    def __init__(self, R, T, camera_mtx, camera_dist, projector_mtx, projector_dist):
        super(PlaneRefineBatch, self).__init__()
        self.R = nn.Parameter(R)
        self.T = nn.Parameter(T)
        self.camera_mtx = nn.Parameter(camera_mtx)
        self.camera_dist = nn.Parameter(camera_dist)
        self.projector_mtx = nn.Parameter(projector_mtx)
        self.projector_dist = nn.Parameter(projector_dist)

    def forward(self, camera_points, projector_points):
        pts, error = triangulation_pytorch_batch(camera_points, projector_points, self.R, self.T, self.camera_mtx, self.camera_dist, self.projector_mtx, self.projector_dist)

        return pts, error
