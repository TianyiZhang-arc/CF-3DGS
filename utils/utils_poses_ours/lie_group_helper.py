import numpy as np
import torch
import torch.nn.functional as F
from typing import List

# TODO: delete the unused functions
#############################################################################################
# functions for torch and numpy

def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output

#############################################################################################
# functions for torch

def inv_3x4(pose: torch.Tensor, use_inverse=False) -> torch.Tensor:
    """ invert a 3x4 camera pose
    :param pose: (3, 4) 
    """
    R,t = pose[...,:3],pose[...,3:]
    R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
    t_inv = (-R_inv@t)[...,0]
    pose_inv = torch.cat([R_inv,t_inv[...,None]],dim=-1)
    return pose_inv

class Lie():
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self,w: torch.Tensor) -> torch.Tensor: # [...,3]
        # from lie algebra to rotation matrix
        # rodrigues formula 
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        # I = torch.eye(3,device=w.device,dtype=torch.float32)
        I = torch.eye(3,device=w.device,dtype=w.dtype)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I+A*wx+B*wx@wx
        return R

    def SO3_to_so3(self,R: torch.Tensor,eps=1e-7) -> torch.Tensor: # [...,3,3]
        # rotation matrix to lie algebra formulation, as rotation around axis of amount equal to norm of axis
        trace = R[...,0,0]+R[...,1,1]+R[...,2,2]
        theta = ((trace-1)/2).clamp(-1+eps,1-eps).acos_()[...,None,None]%np.pi # ln(R) will explode if theta==pi
        lnR = 1/(2*self.taylor_A(theta)+1e-8)*(R-R.transpose(-2,-1)) # FIXME: wei-chiu finds it weird
        w0,w1,w2 = lnR[...,2,1],lnR[...,0,2],lnR[...,1,0]
        w = torch.stack([w0,w1,w2],dim=-1)
        return w

    def se3_to_SE3(self,wu: torch.Tensor) -> torch.Tensor: # [...,3]
        # se3 is lie algebra and translation vector 
        w,u = wu.split([3,3],dim=-1)

        # so3 to SO3
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        # I = torch.eye(3,device=w.device,dtype=torch.float32)
        I = torch.eye(3,device=w.device,dtype=w.dtype)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I+A*wx+B*wx@wx

        V = I+B*wx+C*wx@wx
        Rt = torch.cat([R,(V@u[...,None])],dim=-1)
        return Rt

    def SE3_to_se3(self,Rt: torch.Tensor,eps=1e-8) -> torch.Tensor: # [...,3,4]
        R,t = Rt.split([3,1],dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        # I = torch.eye(3,device=w.device,dtype=torch.float32)
        I = torch.eye(3,device=w.device,dtype=w.dtype)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
        u = (invV@t)[...,0]
        wu = torch.cat([w,u],dim=-1)
        return wu    

    def skew_symmetric(self,w: torch.Tensor) -> torch.Tensor:
        w0,w1,w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                          torch.stack([w2,O,-w0],dim=-1),
                          torch.stack([-w1,w0,O],dim=-1)],dim=-2)
        return wx

    def taylor_A(self,x: torch.Tensor,nth=10) -> torch.Tensor:
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

    def taylor_B(self,x: torch.Tensor,nth=10) -> torch.Tensor:
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

    def taylor_C(self,x: torch.Tensor,nth=10) -> torch.Tensor:
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

def compose(pose_list: List[torch.Tensor]) -> torch.Tensor:
     # compose a sequence of poses together
     # pose_new(x) = poseN o ... o pose2 o pose1(x)
     pose_new = pose_list[0]
     for pose in pose_list[1:]:
          pose_new = _compose_pair_b_at_a(pose_a=pose_new,pose_b=pose)
     return pose_new

def _compose_pair_b_at_a(pose_a: torch.Tensor,pose_b: torch.Tensor) -> torch.Tensor:
     # pose_new(x) = pose_b o pose_a(x)
     R_a,t_a = pose_a[...,:3],pose_a[...,3:]
     R_b,t_b = pose_b[...,:3],pose_b[...,3:]
     R_new = R_b@R_a
     t_new = (R_b@t_a+t_b)[...,0]
     pose_new = torch.cat([R_new,t_new[...,None]],dim=-1)
     return pose_new


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def quat2rotation(q):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quat (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q).cuda()

    norm = torch.sqrt(
        q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    )
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3)).to(q)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot

def rotation2quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix).cuda()

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def quatmultiply(q1, q2):
    """
    Multiply two quaternions together using quaternion arithmetic
    """
    # Extract scalar and vector parts of the quaternions
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    # Calculate the quaternion product
    result_quaternion = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )

    return result_quaternion


def quatTrans2matrix(inputs):
    """ Convert quaternion and translation to transformation matrix.
    :inputs:    [quat, trans] (7,) torch.Tensor
    :returns:   matrix (4, 4) torch.Tensor
    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quat, T = inputs[:, :4], inputs[:, 4:]
    w2c = torch.eye(4).to(inputs).float()
    w2c[:3, :3] = quat2rotation(quat)
    w2c[:3, 3] = T
    return w2c

def matrix2quatTrans(matrix):
    """ Convert transformation matrix to quaternion and translation.
    :input:     matrix: (4, 4) torch.Tensor
    :returns:   [quat, trans] (7,) torch.Tensor
    """
    rot = matrix[:3, :3].unsqueeze(0).detach() # TODO: check detach
    quat = rotation2quat(rot).squeeze()
    tran = matrix[:3, 3].detach()
    return torch.cat([quat, tran])