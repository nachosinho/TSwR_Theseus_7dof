"""Theseus AutoDiff cost functions for the cleaned Panda baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .kinematics import fk_urdf_tcp, link_control_points_world
from .tensor_utils import as_batched, scalar_like


def start_joint_error(optim_vars, aux_vars) -> torch.Tensor:
    (q0,) = optim_vars
    q_start, joint_scale = aux_vars
    batch_size = q0.tensor.shape[0]
    return (q0.tensor - as_batched(q_start.tensor, batch_size)) / scalar_like(joint_scale.tensor, batch_size)


def _safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1.0e-12) -> torch.Tensor:
    """Numerically safe Euclidean norm.

    ``torch.linalg.norm`` has an undefined derivative exactly at zero. That is
    usually harmless for a forward-only planner, but it can create NaNs when we
    differentiate through the Theseus solver in the dGPMP2 demo. Adding a tiny
    epsilon keeps the derivative finite while changing distances only at the
    sub-micrometer numerical level.
    """
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + eps)


def _so3_axis_near_pi(R: torch.Tensor) -> torch.Tensor:
    """Recover a stable rotation axis for rotations close to 180 degrees.

    The usual skew-symmetric formula loses all information at exactly pi
    because R - R^T becomes zero. This branch uses diagonal-dominant formulas
    and normalizes the result defensively. The sign of the axis is ambiguous
    at exactly pi, which is expected for SO(3) logarithms.
    """
    eps = torch.as_tensor(1.0e-12, dtype=R.dtype, device=R.device)

    r00, r11, r22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    r01, r02 = R[:, 0, 1], R[:, 0, 2]
    r10, r12 = R[:, 1, 0], R[:, 1, 2]
    r20, r21 = R[:, 2, 0], R[:, 2, 1]

    # Compute the axis from the largest diagonal component to avoid division
    # by a tiny number. Formulas follow the standard quaternion-from-matrix
    # conversion specialized to a pi-angle rotation.
    x = 0.5 * torch.sqrt(torch.clamp(1.0 + r00 - r11 - r22, min=eps))
    axis_x = torch.stack((x, (r01 + r10) / (4.0 * x), (r02 + r20) / (4.0 * x)), dim=1)

    y = 0.5 * torch.sqrt(torch.clamp(1.0 - r00 + r11 - r22, min=eps))
    axis_y = torch.stack(((r01 + r10) / (4.0 * y), y, (r12 + r21) / (4.0 * y)), dim=1)

    z = 0.5 * torch.sqrt(torch.clamp(1.0 - r00 - r11 + r22, min=eps))
    axis_z = torch.stack(((r02 + r20) / (4.0 * z), (r12 + r21) / (4.0 * z), z), dim=1)

    use_x = (r00 >= r11) & (r00 >= r22)
    use_y = (~use_x) & (r11 >= r22)

    axis = torch.where(use_x[:, None], axis_x, torch.where(use_y[:, None], axis_y, axis_z))
    return axis / _safe_norm(axis, dim=1, keepdim=True)


def _so3_log_vector(R: torch.Tensor) -> torch.Tensor:
    """Return log(R) as an axis-angle vector [B,3].

    The ordinary formula ``theta / (2 sin(theta)) * vee(R - R.T)`` is stable
    for most rotations, but it degenerates near theta=pi because the
    skew-symmetric part approaches zero. This implementation keeps the safe
    small-angle path and switches to a diagonal-based axis recovery near pi.
    """
    if R.ndim != 3 or R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape [B,3,3], got {tuple(R.shape)}")

    v = torch.stack(
        (
            R[:, 2, 1] - R[:, 1, 2],
            R[:, 0, 2] - R[:, 2, 0],
            R[:, 1, 0] - R[:, 0, 1],
        ),
        dim=1,
    )
    sin_theta = 0.5 * _safe_norm(v, dim=1, keepdim=True)
    cos_theta = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1.0) * 0.5).clamp(-1.0, 1.0).unsqueeze(1)
    theta = torch.atan2(sin_theta, cos_theta)

    small = sin_theta < 1.0e-7
    scale = torch.where(small, 0.5 + theta * theta / 12.0, theta / (2.0 * sin_theta.clamp_min(1.0e-12)))
    regular_log = scale * v

    near_pi = cos_theta.squeeze(1) < -0.9999
    pi_axis = _so3_axis_near_pi(R)
    pi_log = theta * pi_axis
    return torch.where(near_pi[:, None], pi_log, regular_log)

def tcp_position_error(optim_vars, aux_vars) -> torch.Tensor:
    (qN,) = optim_vars
    (
        tcp_goal_pos,
        joint_origin_xyz,
        joint_origin_rpy,
        joint_axis,
        tcp_fixed_transform,
        tool_transform,
        tcp_pos_scale,
    ) = aux_vars

    q = qN.tensor
    batch_size = q.shape[0]
    T_tcp = fk_urdf_tcp(
        q,
        joint_origin_xyz.tensor,
        joint_origin_rpy.tensor,
        joint_axis.tensor,
        tcp_fixed_transform.tensor,
        tool_transform.tensor,
    )
    p_tcp = T_tcp[:, :3, 3]
    p_goal = as_batched(tcp_goal_pos.tensor, batch_size)
    return (p_tcp - p_goal) / scalar_like(tcp_pos_scale.tensor, batch_size)


def tcp_orientation_error(optim_vars, aux_vars) -> torch.Tensor:
    (qN,) = optim_vars
    (
        tcp_goal_rot,
        joint_origin_xyz,
        joint_origin_rpy,
        joint_axis,
        tcp_fixed_transform,
        tool_transform,
        tcp_rot_scale,
    ) = aux_vars

    q = qN.tensor
    batch_size = q.shape[0]
    T_tcp = fk_urdf_tcp(
        q,
        joint_origin_xyz.tensor,
        joint_origin_rpy.tensor,
        joint_axis.tensor,
        tcp_fixed_transform.tensor,
        tool_transform.tensor,
    )
    R_tcp = T_tcp[:, :3, :3]
    R_goal = as_batched(tcp_goal_rot.tensor, batch_size)
    R_err = torch.matmul(R_goal.transpose(1, 2), R_tcp)
    rot_vec = _so3_log_vector(R_err)
    return rot_vec / scalar_like(tcp_rot_scale.tensor, batch_size)


def joint_velocity_boundary_error(optim_vars, aux_vars) -> torch.Tensor:
    """Boundary velocity error for qdot_0 or qdot_N."""
    (qdot_var,) = optim_vars
    qdot_target, qdot_scale = aux_vars
    qdot = qdot_var.tensor
    batch_size = qdot.shape[0]
    target = as_batched(qdot_target.tensor, batch_size)
    return (qdot - target) / scalar_like(qdot_scale.tensor, batch_size)


def gpmp2_constant_velocity_prior_error(optim_vars, aux_vars) -> torch.Tensor:
    """Whitened GPMP2 constant-velocity prior between consecutive states.

    State is x_i = [q_i, qdot_i]. The prior assumes a white-noise-on-acceleration
    model with transition:

        q_{i+1}    = q_i + dt * qdot_i
        qdot_{i+1} = qdot_i

    and covariance Q = qc * [[dt^3/3, dt^2/2], [dt^2/2, dt]] per joint.
    The returned 14D residual is whitened by Q^{-1/2}.
    """
    q_i, qdot_i, q_j, qdot_j = optim_vars
    gp_dt, gp_qc = aux_vars

    q_i_t = q_i.tensor
    qdot_i_t = qdot_i.tensor
    q_j_t = q_j.tensor
    qdot_j_t = qdot_j.tensor
    batch_size = q_i_t.shape[0]

    dt = scalar_like(gp_dt.tensor, batch_size)
    qc = scalar_like(gp_qc.tensor, batch_size)

    pos_err = q_j_t - q_i_t - dt * qdot_i_t
    vel_err = qdot_j_t - qdot_i_t

    # Q^{-1} = (1/qc) * [[12/dt^3, -6/dt^2], [-6/dt^2, 4/dt]]
    a = 12.0 / (qc * dt.pow(3))
    b = -6.0 / (qc * dt.pow(2))
    c = 4.0 / (qc * dt)

    # Lower Cholesky of Q^{-1} = L L^T, applied as row-vector whitening e @ L.
    l11 = torch.sqrt(a)
    l21 = b / l11
    l22 = torch.sqrt(torch.clamp(c - l21.pow(2), min=1.0e-12))

    white_pos = l11 * pos_err + l21 * vel_err
    white_vel = l22 * vel_err
    return torch.stack((white_pos, white_vel), dim=-1).reshape(batch_size, -1)


def joint_limit_error(optim_vars, aux_vars) -> torch.Tensor:
    """Hinge penalty for joint-limit violations; exactly zero inside limits."""
    (q_var,) = optim_vars
    q_min, q_max, limit_margin, limit_scale = aux_vars
    q = q_var.tensor
    batch_size = q.shape[0]
    q_min_t = as_batched(q_min.tensor, batch_size)
    q_max_t = as_batched(q_max.tensor, batch_size)
    margin = scalar_like(limit_margin.tensor, batch_size)
    scale = scalar_like(limit_scale.tensor, batch_size)
    lower = torch.clamp(q_min_t + margin - q, min=0.0) / scale
    upper = torch.clamp(q - (q_max_t - margin), min=0.0) / scale
    return torch.cat((lower, upper), dim=1)


def smooth_hinge(x: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    return F.softplus(beta * x) / beta


def _obstacle_centers_batched(centers: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return obstacle centers as [B, O, 3]."""
    if centers.ndim == 2:
        centers = centers.unsqueeze(0)  # [O,3] -> [1,O,3]
    if centers.ndim != 3 or centers.shape[-1] != 3:
        raise ValueError(f"obstacle_centers must have shape [O,3] or [B,O,3], got {tuple(centers.shape)}")
    return as_batched(centers, batch_size)


def _obstacle_radii_batched(radii: torch.Tensor, batch_size: int, num_obstacles: int) -> torch.Tensor:
    """Return obstacle radii as [B, O]."""
    if radii.ndim == 0:
        radii = radii.reshape(1, 1)
    elif radii.ndim == 1:
        radii = radii.reshape(1, -1)
    elif radii.ndim == 3 and radii.shape[-1] == 1:
        radii = radii.squeeze(-1)
    elif radii.ndim != 2:
        raise ValueError(f"obstacle_radii must have shape [O], [B,O] or [B,O,1], got {tuple(radii.shape)}")
    radii = as_batched(radii, batch_size)
    if radii.shape[1] == 1 and num_obstacles > 1:
        radii = radii.expand(batch_size, num_obstacles)
    if radii.shape[1] != num_obstacles:
        raise ValueError(f"Expected {num_obstacles} obstacle radii, got {radii.shape[1]}")
    return radii


def link_sphere_collision_error(optim_vars, aux_vars) -> torch.Tensor:
    """Collision residual for dense link control points vs multiple spherical obstacles.

    Returns [B, O * L * P]. Each control point is treated as a sphere center with
    one global radius supplied by ``link_collision_radius``. For O=1 this is
    equivalent to the previous single-obstacle cost.
    """
    (q_var,) = optim_vars
    (
        obstacle_centers,
        obstacle_radii,
        collision_safety_margin,
        link_collision_radius,
        joint_origin_xyz,
        joint_origin_rpy,
        joint_axis,
        tcp_fixed_transform,
        link_control_points,
        collision_scale,
    ) = aux_vars

    q = q_var.tensor
    batch_size = q.shape[0]
    points_world = link_control_points_world(
        q,
        joint_origin_xyz.tensor,
        joint_origin_rpy.tensor,
        joint_axis.tensor,
        link_control_points.tensor,
        tcp_fixed_transform.tensor,
    )  # [B,L,P,3]

    centers = _obstacle_centers_batched(obstacle_centers.tensor, batch_size)  # [B,O,3]
    num_obstacles = centers.shape[1]
    radii = _obstacle_radii_batched(obstacle_radii.tensor, batch_size, num_obstacles)  # [B,O]
    margin = scalar_like(collision_safety_margin.tensor, batch_size).squeeze(1)  # [B]
    link_radius = scalar_like(link_collision_radius.tensor, batch_size).squeeze(1)  # [B]
    scale = scalar_like(collision_scale.tensor, batch_size).squeeze(1)  # [B]

    # distance: [B,O,L,P]
    distance = _safe_norm(points_world[:, None, :, :, :] - centers[:, :, None, None, :], dim=-1)
    required = radii[:, :, None, None] + margin[:, None, None, None] + link_radius[:, None, None, None]
    violation = (required - distance) / scale[:, None, None, None]
    return smooth_hinge(violation, beta=10.0).reshape(batch_size, -1)
