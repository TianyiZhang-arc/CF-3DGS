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
import numpy as np
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.append(os.path.abspath(BASE_DIR))
from tqdm import tqdm
import torchvision
from argparse import ArgumentParser
# import rerun as rr

from scene_eval.arguments import ModelParams, PipelineParams
from scene_eval.gaussian_model import GaussianModel # NOTE: change for different models if needed
from scene_eval import Scene
from scene_eval.renderer import render


from eval_utils.pose_utils import load_pose, assign_pose, save_pose
from eval_utils.utils_poses.align_traj import align_ate_c2b_use_a2b
from eval_utils.vis_utils import apply_depth_colormap
# from utils.utils_2dgs import depth_to_normal
from eval_utils.general_utils import get_expon_lr_func
from eval_utils.graphics_utils import getWorld2View

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(normal_path, exist_ok=True)
    
    gaussians.eval()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        image = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(image, os.path.join(render_path, f"{view.image_name}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{view.image_name}.png"))
        if 'depth' in render_pkg:
            depth_image = apply_depth_colormap(render_pkg["depth"].squeeze().unsqueeze(-1), render_pkg["opacity"].squeeze().unsqueeze(-1), near_plane=None, far_plane=None).permute(2, 0, 1)
            torchvision.utils.save_image(depth_image, os.path.join(depth_path, f"{view.image_name}.png"))
            # depth_normal_image = (0.5 * (depth_to_normal(view, render_pkg["depth"].squeeze()[None, ...])[0] + 1)).permute(2, 0, 1)
            # torchvision.utils.save_image(depth_normal_image, os.path.join(normal_path, f"{view.image_name}_depth2normal.png"))
        if 'normal' in render_pkg:
            normal_image = 0.5 * (torch.nn.functional.normalize(render_pkg["normal"].squeeze(), p=2, dim=0) + 1)
            torchvision.utils.save_image(normal_image, os.path.join(normal_path, f"{view.image_name}.png"))
            



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    # with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
        load_pose(os.path.join(dataset.model_path, 'train_pose', f'pose_{iteration}.pt'), scene.getTrainCameras())
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

    if not skip_test:
        # align test poses to train poses
        test_cameras, train_cameras = scene.getTestCameras(), scene.getTrainCameras()
        load_pose(os.path.join(dataset.model_path, 'train_pose', f'pose_{iteration}.pt'), scene.getTrainCameras())
        train_poses = [getWorld2View(train_cameras[i].R, train_cameras[i].T) for i in range(len(train_cameras))]
        train_poses = [train_cameras[i].c2w.inverse() for i in range(len(train_cameras))]
        train_poses_gt = [getWorld2View(train_cameras[i].R_gt, train_cameras[i].T_gt) for i in range(len(train_cameras))]
        test_poses_gt = [getWorld2View(test_cameras[i].R_gt, test_cameras[i].T_gt) for i in range(len(test_cameras))]
        test_poses_aligned = align_ate_c2b_use_a2b(torch.stack(train_poses_gt).inverse(), torch.stack(train_poses).inverse().cpu(), torch.stack(test_poses_gt).inverse()).inverse()
        train_poses_aligned = align_ate_c2b_use_a2b(torch.stack(train_poses_gt).inverse(), torch.stack(train_poses).inverse().cpu(), torch.stack(train_poses_gt).inverse()).inverse()
        test_cameras = assign_pose(test_cameras, test_poses_aligned)
        # plot_alignment(torch.stack(train_poses).inverse(), torch.stack(train_poses_gt).inverse(), train_poses_aligned.inverse(), \
        #                 torch.stack(test_poses_gt).inverse(), test_poses_aligned.inverse(), \
        #                     test_cameras[0].intrinsic_matrix, test_cameras[0].image_height, test_cameras[0].image_width, scene, \
        #                     save_path=os.path.join(dataset.model_path, 'test_pose', f'pose_{iteration}.rrd')) 
        if args.optim_test_pose:          
            os.makedirs(os.path.join(dataset.model_path, 'test_pose'), exist_ok=True)
            test_pose_path = os.path.join(dataset.model_path, 'test_pose', f'pose_{iteration}.pt')
            optim_test_pose(test_pose_path, test_cameras, (render, gaussians, pipeline, background), args.optim_test_pose_iter)
            load_pose(test_pose_path, scene.getTestCameras())
        
        render_set(dataset.model_path, "test", scene.loaded_iter, test_cameras, gaussians, pipeline, background)

def optim_test_pose(path, cameras, render_pkgs, optim_iter=500, lr=0.001, enable_scheduler=False):
    """
    Args:
        render_pkgs: (renderFunc, gaussians, pipe, bg)
    """
    def update_learning_rate(optimizer, scheduler_args, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in optimizer.param_groups:
                lr = scheduler_args(iteration)
                param_group['lr'] = lr

    render, gaussians, pipe, bg = render_pkgs
    gaussians.eval()

    cam_params = []
    for viewpoint in cameras:
        viewpoint.pose_delta.requires_grad_(True)
        cam_params.append(
            {
                "params": [viewpoint.pose_delta],
                "lr": lr,
                "initial_lr": lr,
                "name": "cam_{}".format(viewpoint.uid),
            }
        )
        optimizer_cam = torch.optim.Adam(cam_params)
        cam_scheduler_args = get_expon_lr_func(lr_init=lr, lr_final=0.1*lr, max_steps=optim_iter) # keep the lr of camera pose and pts3d to be the same

    gt = torch.stack([viewpoint.original_image for viewpoint in cameras], dim=0).cuda()
    progress_bar = tqdm(range(optim_iter), desc=f"Optimize test poses")
    for iteration in range(optim_iter):
        if enable_scheduler:
            update_learning_rate(optimizer_cam, cam_scheduler_args, iteration)
        for viewpoint in cameras:
            viewpoint.c2w = viewpoint.update_pose(update_rot=True)

        image_stack = []
        for viewpoint in cameras:
            render_pkg = render(viewpoint, gaussians, pipe, bg)
            image_stack.append(render_pkg['render'])
        rendering = torch.stack(image_stack)
        loss = torch.abs(gt - rendering).mean()
        loss.backward()

        with torch.no_grad():
            optimizer_cam.step()
            optimizer_cam.zero_grad()
            if iteration%10==0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(10)

    progress_bar.close()
    save_pose(path, cameras)

# def plot_alignment(train_poses, train_poses_gt, train_poses_aligned, test_poses_gt, test_poses_aligned, intrinsics, H, W, scene, save_path, scale=0.5):
#     # Create rerun plotter
#     train_cameras, test_cameras = scene.getTrainCameras(), scene.getTestCameras()
#     rr.init("rerun_test_pose_optim")
#     rr.save(save_path)
#     for i, (train_pose, train_pose_gt, train_pose_aligned) in enumerate(zip(train_poses, train_poses_gt, train_poses_aligned)):
#         gt_image = (torch.clamp(train_cameras[i].original_image.to("cuda"), 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
#         name = train_cameras[i].image_name
#         rr.log(f"train_pose/{name}", rr.Transform3D(translation=train_pose.detach().cpu()[:3, 3], mat3x3=train_pose.detach().cpu()[:3, :3], scale=scale))
#         rr.log(f"train_pose/{name}", rr.Pinhole(image_from_camera=intrinsics.detach().cpu(), resolution=[H, W]))
#         rr.log(f"train_pose/{name}/image", rr.Image(gt_image))
#         rr.log(f"train_pose_gt/{name}", rr.Transform3D(translation=train_pose_gt.detach().cpu()[:3, 3], mat3x3=train_pose_gt.detach().cpu()[:3, :3], scale=scale))
#         rr.log(f"train_pose_gt/{name}", rr.Pinhole(image_from_camera=intrinsics.detach().cpu(), resolution=[H, W]))
#         rr.log(f"train_pose_gt/{name}/image", rr.Image(gt_image))
#         rr.log(f"train_pose_aligned/{name}", rr.Transform3D(translation=train_pose_aligned.detach().cpu()[:3, 3], mat3x3=train_pose_aligned.detach().cpu()[:3, :3], scale=scale))
#         rr.log(f"train_pose_aligned/{name}", rr.Pinhole(image_from_camera=intrinsics.detach().cpu(), resolution=[H, W]))
#         rr.log(f"train_pose_aligned/{name}/image", rr.Image(gt_image))
#     for i, (test_pose_gt, test_pose_aligned) in enumerate(zip(test_poses_gt, test_poses_aligned)):
#         gt_image = (torch.clamp(test_cameras[i].original_image.to("cuda"), 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
#         name = test_cameras[i].image_name
#         rr.log(f"test_pose_gt/{name}", rr.Transform3D(translation=test_pose_gt.detach().cpu()[:3, 3], mat3x3=test_pose_gt.detach().cpu()[:3, :3], scale=scale))
#         rr.log(f"test_pose_gt/{name}", rr.Pinhole(image_from_camera=intrinsics.detach().cpu(), resolution=[H, W]))
#         rr.log(f"test_pose_gt/{name}/image", rr.Image(gt_image))
#         rr.log(f"test_pose_aligned/{name}", rr.Transform3D(translation=test_pose_aligned.detach().cpu()[:3, 3], mat3x3=test_pose_aligned.detach().cpu()[:3, :3], scale=scale))
#         rr.log(f"test_pose_aligned/{name}", rr.Pinhole(image_from_camera=intrinsics.detach().cpu(), resolution=[H, W]))
#         rr.log(f"test_pose_aligned/{name}/image", rr.Image(gt_image))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--optim_test_pose", action="store_true")
    parser.add_argument("--optim_test_pose_iter", default=500, type=int)
    parser.add_argument("--model", type=str, default='3dgs')
    args = parser.parse_args(sys.argv[1:])
    # args = get_combined_args(parser)

    print("Rendering " + args.model_path)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)