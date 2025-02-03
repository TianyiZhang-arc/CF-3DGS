
import numpy as np
import torch
import cv2
from typing import List, Optional

##### RTE #####
def rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5*(a+b+c-1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error

def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2+dy**2+dz**2)
    return trans_error

def compute_rpe(gt, pred):
    """
    Args:
        gt: c2w
        pred: c2w
    """
    trans_errors = []
    rot_errors = []
    for i in range(len(gt)-1):
        gt1 = gt[i]
        gt2 = gt[i+1]
        gt_rel = np.linalg.inv(gt1) @ gt2

        pred1 = pred[i]
        pred2 = pred[i+1]
        pred_rel = np.linalg.inv(pred1) @ pred2
        rel_err = np.linalg.inv(gt_rel) @ pred_rel
        
        trans_errors.append(translation_error(rel_err))
        rot_errors.append(rotation_error(rel_err))
    rpe_trans = np.mean(np.asarray(trans_errors))
    rpe_rot = np.mean(np.asarray(rot_errors))
    return rpe_trans * 100, rpe_rot * 180 / np.pi

##### ATE #####
def compute_ATE(gt, pred):
    """Compute RMSE of ATE
    Args:
        gt: c2w
        pred: c2w
    """
    errors = []

    for i in range(len(pred)):
        # cur_gt = np.linalg.inv(gt_0) @ gt[i]
        cur_gt = gt[i]
        gt_xyz = cur_gt[:3, 3] 

        # cur_pred = np.linalg.inv(pred_0) @ pred[i]
        cur_pred = pred[i]
        pred_xyz = cur_pred[:3, 3]

        align_err = gt_xyz - pred_xyz

        errors.append(np.sqrt(np.sum(align_err ** 2)))
    ate = np.sqrt(np.mean(np.asarray(errors) ** 2)) 
    return ate

##### APE #####
def rotation_distance(R1: torch.Tensor, R2: torch.Tensor,eps=1e-7) -> torch.Tensor:
    """
    same as rotation error, but input two batch R1, R2
    """
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle

def translation_distance(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    same as rotation error, but input two batch t1, t2
    """
    return (t1 - t2).norm(dim=-1)

@torch.no_grad()
def evaluate_camera_alignment(aligned_pred_poses, poses_gt):
    """
    measure errors in rotation and translation
    aligned_pred_poses: c2w
    poses_gt: c2w
    """
    R_aligned, t_aligned = aligned_pred_poses.split([3, 1], dim=-1)
    R_gt, t_gt = poses_gt.split([3, 1], dim=-1)

    R_error = rotation_distance(R_aligned[..., :3, :3], R_gt[..., :3, :3])
    t_error = (t_aligned - t_gt)[..., 0].norm(dim=-1)

    mean_rotation_error = np.rad2deg(R_error.mean().cpu()).item()
    mean_position_error = t_error.mean().item()
    # med_rotation_error = np.rad2deg(R_error.median().cpu()).item()
    # med_position_error = t_error.median().item()

    return mean_rotation_error, mean_position_error

##### AUC #####
def compute_rot_err(R1: np.ndarray, R2: np.ndarray):
    rot_err = R1[0:3,0:3].T.dot(R2[0:3,0:3])
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1,3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis = 1), -1) / np.pi * 180.
    return rot_err[0]

def compute_pose_err(pose: np.ndarray, pose_gt:np.ndarray):
    """
    pose: w2c
    pose_gt: w2c
    """
    trans_err = np.linalg.norm(np.linalg.inv(pose)[:3, 3] - np.linalg.inv(pose_gt)[:3, 3])
    rot_err = compute_rot_err(pose[:3, :3], pose_gt[:3, :3])
    return trans_err, rot_err

def eval_relpose(poses_ref: List[np.ndarray], poses_est: List[np.ndarray]):
    """
    poses_ref: w2c
    poses_est: w2c
    """
    num_poses = len(poses_est)
    rot_err_list, t_angle_list = [], []
    for i in range(num_poses - 1):
        pose1 = np.array(poses_est[i])
        pose1_gt = np.array(poses_ref[i])
        for j in range(i + 1, num_poses):
            pose2 = np.array(poses_est[j])
            pose2_gt = np.array(poses_ref[j])

            relR = pose1[:3, :3] @ np.transpose(pose2[:3, :3])
            relT = pose1[:3, 3] - relR @ pose2[:3, 3]
            relT_vec = relT / np.linalg.norm(relT)
            relR_gt = pose1_gt[:3, :3] @ np.transpose(pose2_gt[:3, :3])
            relT_gt = pose1_gt[:3, 3] - relR_gt @ pose2_gt[:3, 3]
            relT_gt_vec = relT_gt / np.linalg.norm(relT_gt)

            rot_err = compute_rot_err(relR, relR_gt)
            t_angle = np.arccos(np.clip(np.abs(relT_vec.dot(relT_gt_vec)), 0, 1)) * 180.0 / np.pi
            rot_err_list.append(rot_err)
            t_angle_list.append(t_angle)
    return rot_err_list, t_angle_list

def compute_recall(errors):
    num_elements = len(errors)
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(num_elements) + 1) / num_elements
    return errors, recall

def compute_auc(errors, thresholds, min_error: Optional[float] = None):
    errors, recall = compute_recall(errors)

    if min_error is not None:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / len(errors)
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]

    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        auc = np.trapz(r, x=e)/t
        aucs.append(auc*100)
    return aucs