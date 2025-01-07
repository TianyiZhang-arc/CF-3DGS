import torch
from gsplat.rendering import rasterization
from scene.gaussian_model import GaussianModel



def render(
    viewpoint_camera,
    pc: GaussianModel,
    args,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0:
        return None

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    colors = pc.get_features
    if args.detach_xyz:
        means3D = means3D.detach()
    if args.detach_opacity:
        opacity = opacity.detach()
    if args.detach_features:
        colors = colors.detach()

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if args.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # check if the covariance is isotropic
        if pc.get_scaling.shape[-1] == 1:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = pc.get_scaling
        rotations = pc.get_rotation
    
    if args.detach_rotation:
        rotations = rotations.detach()
    if args.detach_scaling:
        scales = scales.detach()
    
    # # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    # colors_precomp = None
    # if colors_precomp is None:
    #     if args.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(
    #             -1, 3, (pc.max_sh_degree + 1) ** 2
    #         )
    #         dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
    #             pc.get_features.shape[0], 1
    #         )
    #         dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #     else:
    #         shs = pc.get_features
    # else:
    #     colors_precomp = override_color
    
    w2c = torch.eye(4).cuda().float()
    w2c[:3, :3] = viewpoint_camera.R
    w2c[:3, 3] = viewpoint_camera.T
    render, rendered_alpha, info = rasterization(
        means=means3D,
        quats=rotations,
        scales=scales,
        opacities=opacity.squeeze(),
        colors=colors,
        viewmats=w2c.unsqueeze(0),  # [C, 4, 4]
        Ks=viewpoint_camera.intrinsic_matrix.unsqueeze(0),  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        sh_degree = pc.active_sh_degree,
        # backgrounds = None,
        backgrounds = bg_color.unsqueeze(0),
        render_mode = 'RGB+ED',
        packed=False,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode="classic"
    )   

    rendered_image = render[..., 0:3]
    rendered_depth = torch.where(
        rendered_alpha > 0, render[..., 3:4], -1
    )


    return {
        "render": rendered_image.squeeze().permute(2, 0, 1),
        "viewspace_points": info["means2d"],
        "visibility_filter": info["radii"].squeeze() > 0,
        "radii": info["radii"].squeeze(),
        "depth": rendered_depth.squeeze(),
        "opacity": rendered_alpha.squeeze(),
        "info": info,
    }