import torch
import numpy as np
from eval_utils.utils_poses.align_traj import align_ate_c2b_use_a2b
from eval_utils.utils_poses.comp_ate import eval_relpose, compute_auc,evaluate_camera_alignment, compute_ATE, compute_rpe
from eval_utils.graphics_utils import getWorld2View

def save_pose(path, cameras, gt=False):
    poses = {}
    for cam in cameras:
        pose = getWorld2View(cam.R.detach().cpu(), cam.T.detach().cpu()) if not gt else getWorld2View(cam.R_gt.detach().cpu(), cam.T_gt.detach().cpu())
        poses[cam.image_name] = pose              
    torch.save(poses, path)

def load_pose(path, cameras):
    poses = torch.load(path)
    for cam in cameras:
        cam.c2w = poses[cam.image_name].inverse().to(cam.data_device)

def assign_pose(cameras, RTs):
    """
    Args: 
        RT: w2c 4x4
    """
    for cam, RT in zip(cameras, RTs):
        cam.c2w = RT.inverse().to(cam.data_device)
    return cameras


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def report_pose_error(w2c: torch.Tensor, w2c_gt: torch.Tensor, auc_idx=[1, 3, 5, 10, 20]):
    # report auc
    w2c_np, w2c_gt_np = w2c.numpy(), w2c_gt.numpy()
    rot_err_list, t_angle_list = eval_relpose([w2c_gt_np[i] for i in range(w2c_gt_np.shape[0])], [w2c_np[i] for i in range(w2c_np.shape[0])])
    err_list = [max(rot_err, t_angle) for rot_err, t_angle in zip(rot_err_list, t_angle_list)]
    auc = compute_auc(err_list, auc_idx) # compute auc accuracy
    # report ape
    w2c_aligned = align_ate_c2b_use_a2b(w2c.inverse(), w2c_gt.inverse(), w2c.inverse()).inverse()
    ape_rot, ape_trans = evaluate_camera_alignment(w2c_aligned[:, :3, :4], w2c_gt[:, :3, :4]) # sparf
    rpe_trans, rpe_rot = compute_rpe(np.linalg.inv(w2c_gt_np), np.linalg.inv(w2c_aligned)) # cf-3dgs
    ate = compute_ATE(np.linalg.inv(w2c_gt_np), np.linalg.inv(w2c_aligned)) # cf-3dgs
    auc_dict = {}
    for i in range(len(auc_idx)):
        auc_dict[auc_idx[i]] = auc[i]
    return auc_dict, ape_rot.mean().item(), ape_trans.mean().item(), rpe_rot, rpe_trans, ate    

def print_pose_error(auc_dict, ape_rot, ape_trans, rpe_rot, rpe_trans, ate):
    auc_print = 'AUC ['
    for k in auc_dict:
        auc_print += '{:d}/'.format(k)
    auc_print += ']:'
    for k in auc_dict:
        auc_print += '{:.4f}/'.format(auc_dict[k])
    ape_print = ' APE [R, t]: {:.4f}/{:.4f}'.format(ape_rot, ape_trans)
    rpe_print = ' RPE [R, t]: {:.4f}/{:.4f}'.format(rpe_rot, rpe_trans)
    ate_print = ' ATE: {:.4f}'.format(ate)
    return auc_print + ape_print + rpe_print + ate_print
