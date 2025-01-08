import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List
import open3d as o3d

def apply_depth_colormap(
    depth,
    accumulation,
    near_plane = 2.0,
    far_plane = 6.0,
    cmap="turbo",
):

    def apply_colormap(image, cmap="turbo"):
        colormap = plt.get_cmap(cmap)
        colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
        image_long = (image * 255).long()
        image_long_min = torch.min(image_long)
        image_long_max = torch.max(image_long)
        assert image_long_min >= 0, f"the min value is {image_long_min}"
        assert image_long_max <= 255, f"the max value is {image_long_max}"
        return colormap[image_long[..., 0]]

    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    # depth = torch.nan_to_num(depth, nan=0.0) # TODO(ethan): remove this

    colored_image = apply_colormap(depth, cmap=cmap)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image

def open3d_get_cameras(intrinsics: List[np.ndarray], poses: List[np.ndarray], color: List=[0., 0., 0.], scale: int=0.1):
    """
    Args:
        poses: w2c
    """
    cameras = o3d.geometry.LineSet()
    num_poses = len(poses)
    for i in range(num_poses):
        camera = o3d.geometry.LineSet.create_camera_visualization(int(intrinsics[i][0, 2] * 2), int(intrinsics[i][1, 2] * 2), intrinsics[i], poses[i], scale=scale)
        color_list = [np.array(color) for j in range(np.asarray(camera.lines).shape[0])]
        camera.colors = o3d.utility.Vector3dVector(np.stack(color_list, axis=0))
        cameras += camera
    return cameras