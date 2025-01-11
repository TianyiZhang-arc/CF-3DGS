import numpy as np
import torch
from scipy.spatial.transform import Rotation as RotLib

from eval_utils.utils_poses.align_utils import alignTrajectory
from eval_utils.utils_poses.lie_group_helper import convert3x4_4x4, inv_3x4
from eval_utils.utils_poses.comp_ate import rotation_distance, translation_distance, evaluate_camera_alignment


def _pts_dist_max(pts):
    """
    :param pts:  (N, 3) torch or np
    :return:     scalar
    """
    if torch.is_tensor(pts):
        dist = pts.unsqueeze(0) - pts.unsqueeze(1)  # (1, N, 3) - (N, 1, 3) -> (N, N, 3)
        dist = dist[0]  # (N, 3)
        dist = dist.norm(dim=1)  # (N, )
        max_dist = dist.max()
    else:
        dist = pts[None, :, :] - pts[:, None, :]  # (1, N, 3) - (N, 1, 3) -> (N, N, 3)
        dist = dist[0]  # (N, 3)
        dist = np.linalg.norm(dist, axis=1)  # (N, )
        max_dist = dist.max()
    return max_dist

def align_ate_c2b_use_a2b(traj_a, traj_b, traj_c=None):
    """Align c to b using the sim3 from a to b.
    :param traj_a:  (N0, 3/4, 4) torch tensor
    :param traj_b:  (N0, 3/4, 4) torch tensor
    :param traj_c:  None or (N1, 3/4, 4) torch tensor
    :return:        (N1, 4,   4) torch tensor
    """
    assert traj_a.shape[0] == traj_b.shape[0]
    n_views = traj_a.shape[0]
    if n_views < 20:
        small_system = True
    else:
        small_system = False
    
        
    traj_a = traj_a.float()
    traj_b = traj_b.float()
    traj_c = traj_c.float()
    device = traj_a.device
    if not small_system:
        if traj_c is None:
            traj_c = traj_a.clone()

        traj_a = traj_a.cpu().numpy()
        traj_b = traj_b.cpu().numpy()
        traj_c = traj_c.cpu().numpy()

        R_a = traj_a[:, :3, :3]  # (N0, 3, 3)
        t_a = traj_a[:, :3, 3]  # (N0, 3)
        quat_a = RotLib.from_matrix(R_a).as_quat()  # (N0, 4)

        R_b = traj_b[:, :3, :3]  # (N0, 3, 3)
        t_b = traj_b[:, :3, 3]  # (N0, 3)
        quat_b = RotLib.from_matrix(R_b).as_quat()  # (N0, 4)

        # This function works in quaternion.
        # scalar, (3, 3), (3, ) gt = R * s * est + t.
        s, R, t = alignTrajectory(t_a, t_b, quat_a, quat_b, method='sim3')

        # reshape tensors
        # R = R[None, :, :].astype(np.float32)  # (1, 3, 3)
        # t = t[None, :, None].astype(np.float32)  # (1, 3, 1)
        R = R[None, :, :]
        t = t[None, :, None]
        s = float(s)

        R_c = traj_c[:, :3, :3]  # (N1, 3, 3)
        t_c = traj_c[:, :3, 3:4]  # (N1, 3, 1)

        R_c_aligned = R @ R_c  # (N1, 3, 3)
        t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)
        traj_c_aligned = np.concatenate([R_c_aligned, t_c_aligned], axis=2)  # (N1, 3, 4)

        # append the last row
        traj_c_aligned = convert3x4_4x4(traj_c_aligned)  # (N1, 4, 4)

        traj_c_aligned = torch.from_numpy(traj_c_aligned).to(device)
        return traj_c_aligned  # (N1, 4, 4)
    else:
        # match each view and find the transformation with minimal error
        pose_w2c = torch.linalg.inv(traj_a)[:, :3, :4]
        pose_GT_w2c = torch.linalg.inv(traj_b)[:, :3, :4]
        if traj_c is None:
            traj_c = traj_a.clone()
        # traj_c = traj_c.float().cpu().numpy()
        traj_c = traj_c.cpu().numpy()
        pose_aligned_w2c, ssim_est_gt_c2w = prealign_w2c_small_camera_systems(pose_w2c, pose_GT_w2c)
        # reshape tensors
        R = ssim_est_gt_c2w['R'].float().cpu().numpy()
        t = ssim_est_gt_c2w['t'].float().cpu().numpy()
        s = ssim_est_gt_c2w['s'].float().cpu().numpy()
        R_c = traj_c[:, :3, :3]  # (N1, 3, 3)
        t_c = traj_c[:, :3, 3:4]  # (N1, 3, 1)

        R_c_aligned = R @ R_c  # (N1, 3, 3)
        t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)
        traj_c_aligned = np.concatenate([R_c_aligned, t_c_aligned], axis=2)  # (N1, 3, 4)

        # append the last row
        traj_c_aligned = convert3x4_4x4(traj_c_aligned)  # (N1, 4, 4)
        traj_c_aligned = torch.from_numpy(traj_c_aligned).to(device)
        return traj_c_aligned  # (N1, 4, 4)
    
        # use the first view to align
        # if traj_c is None:
        #     traj_c = traj_a.clone()
        # traj_a_scaled = traj_a.clone().cpu().numpy()
        # traj_a = traj_a.cpu().numpy()
        # traj_b = traj_b.cpu().numpy()
        # traj_c = traj_c.cpu().numpy()
        # traj_c_scaled, scale_a2b = align_scale_c2b_use_a2b(traj_a, traj_b, traj_c)
        # traj_a_scaled[:, :3, 3] *= scale_a2b
        # transformation_from_to = traj_b[0] @ np.linalg.inv(traj_a_scaled[0])
        # traj_c_aligned = transformation_from_to @ traj_c_scaled
        # traj_c_aligned = torch.from_numpy(traj_c_aligned).to(device)     
        # return traj_c_aligned  # (N1, 4, 4)



def align_scale_c2b_use_a2b(traj_a, traj_b, traj_c=None):
    '''Scale c to b using the scale from a to b.
    :param traj_a:      (N0, 3/4, 4) torch tensor
    :param traj_b:      (N0, 3/4, 4) torch tensor
    :param traj_c:      None or (N1, 3/4, 4) torch tensor
    :return:
        scaled_traj_c   (N1, 4, 4)   torch tensor
        scale           scalar
    '''
    if traj_c is None:
        traj_c = traj_a.clone()

    t_a = traj_a[:, :3, 3]  # (N, 3)
    t_b = traj_b[:, :3, 3]  # (N, 3)

    # scale estimated poses to colmap scale
    # s_a2b: a*s ~ b
    scale_a2b = _pts_dist_max(t_b) / _pts_dist_max(t_a)

    traj_c[:, :3, 3] *= scale_a2b

    if traj_c.shape[1] == 3:
        traj_c = convert3x4_4x4(traj_c)  # (N, 4, 4)

    return traj_c, scale_a2b  # (N, 4, 4)

def prealign_w2c_small_camera_systems(pose_w2c: torch.Tensor, pose_GT_w2c: torch.Tensor):
    """Compute the transformation from pose_w2c to pose_GT_w2c by aligning the each pair of pose_w2c 
    to the corresponding pair of pose_GT_w2c and computing the scaling. This is more robust than the
    technique above for small number of input views/poses (<10). Save the inverse 
    transformation for the evaluation, where the test poses must be transformed to the coordinate 
    system of the optimized poses. 

    Args:
        opt (edict): settings
        pose_w2c (torch.Tensor): Shape is (B, 3, 4)
        pose_GT_w2c (torch.Tensor): Shape is (B, 3, 4)
    """
    def alignment_function(poses_c2w_from_padded: torch.Tensor, 
                            poses_c2w_to_padded: torch.Tensor, idx_a: int, idx_b: int):
        """Args: FInd alignment function between two poses at indixes ix_a and idx_n

            poses_c2w_from_padded: Shape is (B, 4, 4)
            poses_c2w_to_padded: Shape is (B, 4, 4)
            idx_a:
            idx_b:

        Returns:
        """
        # We take a copy to keep the original poses unchanged.
        poses_c2w_from_padded = poses_c2w_from_padded.clone()
        # We use the distance between the same two poses in both set to obtain
        # scale misalgnment.
        dist_from = torch.norm(
            poses_c2w_from_padded[idx_a, :3, 3] - poses_c2w_from_padded[idx_b, :3, 3]
        )
        dist_to = torch.norm(
            poses_c2w_to_padded[idx_a, :3, 3] - poses_c2w_to_padded[idx_b, :3, 3])
        scale = dist_to / dist_from

        # alternative for scale
        # dist_from = poses_w2c_from_padded[idx_a, :3, 3] @ poses_c2w_from_padded[idx_b, :3, 3]
        # dist_to = poses_w2c_to_padded[idx_a, :3, 3] @ poses_c2w_to_padded[idx_b, :3, 3]
        # scale = onp.abs(dist_to /dist_from).mean()

        # We bring the first set of poses in the same scale as the second set.
        poses_c2w_from_padded[:, :3, 3] = poses_c2w_from_padded[:, :3, 3] * scale

        # Now we simply apply the transformation that aligns the first pose of the
        # first set with first pose of the second set.
        transformation_from_to = poses_c2w_to_padded[idx_a] @ torch.linalg.inv(poses_c2w_from_padded[idx_a])
        poses_aligned_c2w = transformation_from_to[None] @ poses_c2w_from_padded

        poses_aligned_w2c = torch.linalg.inv(poses_aligned_c2w)
        ssim_est_gt_c2w = {'R':transformation_from_to[:3, :3].unsqueeze(0), 
                           'type':'traj_align', 
                           't':transformation_from_to[:3, 3].reshape(1, 3, 1), 
                           's':scale}

        return poses_aligned_w2c[:, :3], ssim_est_gt_c2w
    
    def pad_poses(p: torch.Tensor) -> torch.Tensor:
        """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
        bottom = torch.broadcast_to(torch.tensor([0, 0, 0, 1.0], device=p.device, 
                                    dtype=p.dtype), 
                                    p[..., :1, :4].shape)
        return torch.cat((p[..., :3, :4], bottom), dim=-2)


    pose_c2w = inv_3x4(pose_w2c)
    pose_GT_c2w = inv_3x4(pose_GT_w2c)
    B = pose_c2w.shape[0]

    n_first_fixed_poses = 0
    if n_first_fixed_poses > 1:
        # the trajectory should be consistent with the first poses 
        ssim_est_gt_c2w = {'R':torch.eye(3).unsqueeze(0), 't':torch.zeros(1,3,1), 's':1.}
        pose_aligned_w2c = pose_w2c
    else:
        # try every combination of pairs and get the rotation/translation
        # take the one with the smallest error
        # this is because for small number of views, the procrustes alignement with SVD is not robust. 
        pose_aligned_w2c_list = []
        ssim_est_gt_c2w_list = []
        error_R_list = []
        error_t_list = []
        full_error = []
        for pair_id_0 in range(B):  # to avoid that it is too long
            for pair_id_1 in range(B):
                if pair_id_0 == pair_id_1:
                    continue
                
                pose_aligned_w2c_, ssim_est_gt_c2w_ = alignment_function\
                    (pad_poses(pose_c2w), pad_poses(pose_GT_c2w),
                        pair_id_0, pair_id_1)
                pose_aligned_w2c_list.append(pose_aligned_w2c_)
                ssim_est_gt_c2w_list.append(ssim_est_gt_c2w_ )

                R_error, t_error = evaluate_camera_alignment(pose_aligned_w2c_, pose_GT_w2c)
                error_R_list.append(R_error.mean().item())
                error_t_list.append(t_error.mean().item())
                full_error.append(t_error.mean().item() * (R_error.mean().item()))

        ind_best = np.argmin(full_error)
        # print(np.argmin(error_R_list), np.argmin(error_t_list), ind_best)
        pose_aligned_w2c = pose_aligned_w2c_list[ind_best]
        ssim_est_gt_c2w = ssim_est_gt_c2w_list[ind_best]

    return pose_aligned_w2c, ssim_est_gt_c2w


def get_align_transformation(traj_est, traj_gt):
    """
    Args: 
        traj_est: c2w, (N, 4, 4)
        traj_gt: c2w, (N, 4, 4)
    """
    traj_est_scaled = traj_est.clone()
    t_est = traj_est[:, :3, 3]  # (N, 3)
    t_gt = traj_gt[:, :3, 3]  # (N, 3)

    # scale estimated poses to colmap scale
    s = _pts_dist_max(t_gt) / _pts_dist_max(t_est)
    traj_est_scaled[:, :3, 3] *= s
    T = traj_gt[0] @ (np.linalg.inv(traj_est_scaled[0]) if \
                      isinstance(traj_est_scaled, np.ndarray) else torch.linalg.inv(traj_est_scaled[0])) 
    # transformation: 
    # traj_est[:, :3, 3] *= s
    # T @ traj_est      
    return s, T 