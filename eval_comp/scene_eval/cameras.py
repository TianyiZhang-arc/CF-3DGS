#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np

from eval_comp.eval_utils.graphics_utils import getWorld2View, getProjectionMatrix, getIntrinsicMatrix
from eval_comp.eval_utils.utils_poses.lie_group_helper import rotation2quat, quat2rotation
from eval_comp.eval_utils.pose_utils import rotation_6d_to_matrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, R_gt, T_gt, K,
                 FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.uid = uid
        self.colmap_id = colmap_id
        # optimize variables
        self.pose_delta = torch.zeros(9).float().to(self.data_device) # dx, drot (6-dim)
        self.identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).float().to(self.data_device)
        self.c2w = torch.tensor(np.linalg.inv(getWorld2View(R, T))).to(self.data_device) # variable to save pose update

        self.R_gt = torch.tensor(R_gt) if R_gt is not None else R_gt
        self.T_gt = torch.tensor(T_gt) if T_gt is not None else T_gt
        self.K = K
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        # self.trans = trans
        # self.scale = scale

    def update_pose(self, update_rot=False):
        """Adjust camera pose based on deltas."""
        dx, drot = self.pose_delta[:3], 0.1 * self.pose_delta[3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.to(drot.device)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=self.pose_delta.device)
        if update_rot:
            transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        T_c2w = torch.eye(4, device=self.pose_delta.device)
        T_c2w[0:3, 0:3] = self.c2w[:3, :3].detach()
        T_c2w[0:3, 3] = self.c2w[:3, 3].detach()
        self.pose_delta.data.fill_(0)
        return torch.matmul(T_c2w, transform)
        
    
    @property
    def R(self):
        return self.c2w[:3, :3].transpose(0, 1)
        
    
    @property
    def T(self):
        return - self.c2w[:3, :3].transpose(0, 1) @ self.c2w[:3, 3]
      
    @property
    def world_view_transform(self):
        # return getWorld2View(self.R, self.T, self.trans, self.scale).transpose(0, 1).to(self.data_device)
        return getWorld2View(self.R, self.T).transpose(0, 1).to(self.data_device)
    
    @property
    def projection_matrix(self):
        return getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device)

    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
    
    @property
    def camera_center(self):
        return self.c2w[:3, 3]
    
    @property
    def intrinsic_matrix(self):
        return torch.tensor(self.K).float().to(self.data_device)
        # return torch.tensor(getIntrinsicMatrix(self.FoVx, self.FoVy, self.image_height, self.image_width)).to(self.data_device)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

class PseudoCamera(nn.Module):
    def __init__(self, quat:torch.Tensor, T:torch.Tensor, FoVx, FoVy, width, height,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(PseudoCamera, self).__init__()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.quat = quat.to(self.data_device) # 1x4
        self.T = T.to(self.data_device)
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

    @property
    def R(self):
        return quat2rotation(self.quat.unsqueeze(0)).squeeze()
      
    @property
    def world_view_transform(self):
        # return getWorld2View(self.R, self.T, self.trans, self.scale).transpose(0, 1).to(self.data_device)
        return getWorld2View(self.R, self.T).transpose(0, 1).to(self.data_device)
    
    @property
    def projection_matrix(self):
        return getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device)

    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
    
    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    @property
    def intrinsic_matrix(self):
        return torch.tensor(getIntrinsicMatrix(self.FoVx, self.FoVy, self.image_height, self.image_width)).to(self.data_device)