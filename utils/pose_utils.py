import torch
import numpy as np
from .utils_poses_ours.align_traj import align_ate_c2b_use_a2b
from .utils_poses_ours.comp_ate import eval_relpose, compute_auc,evaluate_camera_alignment, compute_ATE, compute_rpe

# def report_pose_error(w2c: torch.Tensor, w2c_gt: torch.Tensor, auc_idx=[1, 3, 5, 10, 20]):
#     # report auc
#     w2c_np, w2c_gt_np = w2c.numpy(), w2c_gt.numpy()
#     rot_err_list, t_angle_list = eval_relpose([w2c_gt_np[i] for i in range(w2c_gt_np.shape[0])], [w2c_np[i] for i in range(w2c_np.shape[0])])
#     err_list = [max(rot_err, t_angle) for rot_err, t_angle in zip(rot_err_list, t_angle_list)]
#     auc = compute_auc(err_list, auc_idx) # compute auc accuracy
#     # report ape
#     w2c_aligned = align_ate_c2b_use_a2b(w2c.inverse(), w2c_gt.inverse(), w2c.inverse()).inverse()
#     ape_rot, ape_trans = evaluate_camera_alignment(w2c_aligned[:, :3, :4], w2c_gt[:, :3, :4]) # sparf
#     rpe_trans, rpe_rot = compute_rpe(np.linalg.inv(w2c_gt_np), np.linalg.inv(w2c_aligned)) # cf-3dgs
#     ate = compute_ATE(np.linalg.inv(w2c_gt_np), np.linalg.inv(w2c_aligned)) # cf-3dgs
#     auc_dict = {}
#     for i in range(len(auc_idx)):
#         auc_dict[auc_idx[i]] = auc[i]
#     return auc_dict, ape_rot.mean().item(), ape_trans.mean().item(), rpe_rot, rpe_trans, ate     

def report_pose_error(c2w: torch.Tensor, c2w_gt: torch.Tensor, auc_idx=[1, 3, 5, 10, 20]):
    c2w, c2w_gt = c2w.float(), c2w_gt.float()
    # report auc
    w2c_np, w2c_gt_np = c2w.inverse().numpy(), c2w_gt.inverse().numpy()
    rot_err_list, t_angle_list = eval_relpose([w2c_gt_np[i] for i in range(w2c_gt_np.shape[0])], [w2c_np[i] for i in range(w2c_np.shape[0])])
    err_list = [max(rot_err, t_angle) for rot_err, t_angle in zip(rot_err_list, t_angle_list)]
    auc = compute_auc(err_list, auc_idx) # compute auc accuracy
    # report ape, rpe
    c2w_aligned = align_ate_c2b_use_a2b(c2w, c2w_gt)
    ape_r, ape_t = evaluate_camera_alignment(c2w_aligned, c2w_gt) # sparf
    auc_dict = {}
    for i in range(len(auc_idx)):
        auc_dict[auc_idx[i]] = auc[i]
    output = {}
    output['auc_dict'] = auc_dict
    output['ape'] = {'r': ape_r, 't': ape_t}
    return output

def print_pose_error(errors):
    output_print = ''
    if 'auc_dict' in errors:
        output_print += 'AUC ['
        for k in errors['auc_dict']:
            output_print += '{:d}/'.format(k)
        output_print += ']:'
        for k in errors['auc_dict']:
            output_print += '{:.4f}/'.format(errors['auc_dict'][k])
    if 'ape' in errors:
        output_print += ' APE [R, t]: {:.4f}/{:.4f}'.format(errors['ape']['r'], errors['ape']['t'])
    if 'rpe' in errors:
        output_print += ' RPE [R, t]: {:.4f}/{:.4f}'.format(errors['rpe']['r'], errors['rpe']['t'])
    if 'ate' in errors:
        output_print += ' ATE: {:.4f}'.format(errors['ate'])

    return output_print
