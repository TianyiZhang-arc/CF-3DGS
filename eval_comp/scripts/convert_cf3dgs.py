import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.append(os.path.abspath(BASE_DIR))
os.sys.path.append(os.path.join(os.path.abspath(BASE_DIR), '../'))
import sys
import torch
from argparse import ArgumentParser
from trainer.cf3dgs_trainer import CFGaussianTrainer
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene_eval.gaussian_model import GaussianModel
parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument("--iteration", type=int)
args = parser.parse_args(sys.argv[1:])
model_cfg = lp.extract(args)
pipe_cfg = pp.extract(args)
optim_cfg = op.extract(args)
data_path = model_cfg.source_path
trainer = CFGaussianTrainer(data_path, model_cfg, pipe_cfg, optim_cfg)

trainer.gs_render.gaussians.restore(
    torch.load(os.path.join(trainer.model_cfg.model_path, 'chkpnt/ep00_init.pth')), trainer.optim_cfg)
pose_dict_train = torch.load(os.path.join(trainer.model_cfg.model_path, 'pose/ep00_init.pth'))

ply_path = os.path.join(model_cfg.model_path, f'point_cloud/iteration_{args.iteration}/point_cloud.ply')
pose_path = os.path.join(model_cfg.model_path, f'train_pose/pose_{args.iteration}.pt')
pose_gt_path = os.path.join(model_cfg.model_path, f'train_pose/pose_gt.pt')
os.makedirs(os.path.dirname(ply_path), exist_ok=True)
os.makedirs(os.path.dirname(pose_path), exist_ok=True)
trainer.gs_render.gaussians.save_ply(ply_path)
torch.save(pose_dict_train['poses_pred_dict'], pose_path)
torch.save(pose_dict_train['poses_gt_dict'], pose_gt_path)