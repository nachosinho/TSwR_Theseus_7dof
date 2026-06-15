"""Differentiable URDF-style kinematics for the 7-DOF Franka Panda arm."""

from __future__ import annotations

import torch

from .tensor_utils import as_batched, make_eye4


def _make_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(t[:, 0])
    one = torch.ones_like(t[:, 0])
    row0 = torch.stack((R[:, 0, 0], R[:, 0, 1], R[:, 0, 2], t[:, 0]), dim=-1)
    row1 = torch.stack((R[:, 1, 0], R[:, 1, 1], R[:, 1, 2], t[:, 1]), dim=-1)
    row2 = torch.stack((R[:, 2, 0], R[:, 2, 1], R[:, 2, 2], t[:, 2]), dim=-1)
    row3 = torch.stack((zero, zero, zero, one), dim=-1)
    return torch.stack((row0, row1, row2, row3), dim=-2)


def _rot_x(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(a), torch.ones_like(a)
    return torch.stack(
        (
            torch.stack((o, z, z), dim=-1),
            torch.stack((z, c, -s), dim=-1),
            torch.stack((z, s, c), dim=-1),
        ),
        dim=-2,
    )


def _rot_y(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(a), torch.ones_like(a)
    return torch.stack(
        (
            torch.stack((c, z, s), dim=-1),
            torch.stack((z, o, z), dim=-1),
            torch.stack((-s, z, c), dim=-1),
        ),
        dim=-2,
    )


def _rot_z(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(a), torch.ones_like(a)
    return torch.stack(
        (
            torch.stack((c, -s, z), dim=-1),
            torch.stack((s, c, z), dim=-1),
            torch.stack((z, z, o), dim=-1),
        ),
        dim=-2,
    )


def rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """URDF RPY convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return torch.matmul(torch.matmul(_rot_z(rpy[:, 2]), _rot_y(rpy[:, 1])), _rot_x(rpy[:, 0]))


def fixed_transform_from_xyz_rpy(xyz: torch.Tensor, rpy: torch.Tensor) -> torch.Tensor:
    return _make_transform(rpy_to_matrix(rpy), xyz)


def axis_angle_transform(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Homogeneous rotation transform around an arbitrary axis."""
    axis = axis / torch.clamp(torch.linalg.norm(axis, dim=1, keepdim=True), min=1e-12)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1.0 - c

    R = torch.stack(
        (
            torch.stack((c + x * x * C, x * y * C - z * s, x * z * C + y * s), dim=-1),
            torch.stack((y * x * C + z * s, c + y * y * C, y * z * C - x * s), dim=-1),
            torch.stack((z * x * C - y * s, z * y * C + x * s, c + z * z * C), dim=-1),
        ),
        dim=-2,
    )
    t = torch.zeros(axis.shape[0], 3, dtype=axis.dtype, device=axis.device)
    return _make_transform(R, t)


def fk_urdf_all_frames(
    q: torch.Tensor,
    joint_origin_xyz: torch.Tensor,
    joint_origin_rpy: torch.Tensor,
    joint_axis: torch.Tensor,
) -> torch.Tensor:
    """Return base-to-link transforms for the seven active Panda arm links.

    Args:
        q: [B, 7]
        joint_origin_xyz, joint_origin_rpy, joint_axis: [B, 7, 3] or [1, 7, 3]

    Returns:
        Tensor [B, 7, 4, 4].
    """
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError(f"q must have shape [B, 7], got {tuple(q.shape)}")

    batch_size = q.shape[0]
    xyz = as_batched(joint_origin_xyz, batch_size)
    rpy = as_batched(joint_origin_rpy, batch_size)
    axis = as_batched(joint_axis, batch_size)

    T = make_eye4(batch_size, dtype=q.dtype, device=q.device)
    frames = []
    for j in range(7):
        T_origin = fixed_transform_from_xyz_rpy(xyz[:, j, :], rpy[:, j, :])
        T_joint = axis_angle_transform(axis[:, j, :], q[:, j])
        T = torch.matmul(T, torch.matmul(T_origin, T_joint))
        frames.append(T)
    return torch.stack(frames, dim=1)


def fk_urdf_tcp(
    q: torch.Tensor,
    joint_origin_xyz: torch.Tensor,
    joint_origin_rpy: torch.Tensor,
    joint_axis: torch.Tensor,
    tcp_fixed_transform: torch.Tensor,
    tool_transform: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return TCP transform [B, 4, 4].

    Return the selected TCP frame. The base frame computed by the arm chain is
    panda_link8/flange. ``tool_transform`` can shift the optimized TCP to a
    physical tool/grasp point, for example panda_grasptarget.
    """
    batch_size = q.shape[0]
    frames = fk_urdf_all_frames(q, joint_origin_xyz, joint_origin_rpy, joint_axis)
    T = torch.matmul(frames[:, -1], as_batched(tcp_fixed_transform, batch_size))
    if tool_transform is not None:
        T = torch.matmul(T, as_batched(tool_transform, batch_size))
    return T


def fk_urdf_collision_frames(
    q: torch.Tensor,
    joint_origin_xyz: torch.Tensor,
    joint_origin_rpy: torch.Tensor,
    joint_axis: torch.Tensor,
    tcp_fixed_transform: torch.Tensor,
) -> torch.Tensor:
    """Return transforms for collision control-point groups [B, 8, 4, 4].

    The first seven transforms are the active arm link frames. The last transform
    is attached to the panda_link8/TCP region and is used for the additional
    panda_hand collision-control-point cloud.
    """
    batch_size = q.shape[0]
    frames = fk_urdf_all_frames(q, joint_origin_xyz, joint_origin_rpy, joint_axis)
    hand_frame = torch.matmul(frames[:, -1], as_batched(tcp_fixed_transform, batch_size))
    return torch.cat((frames, hand_frame[:, None, :, :]), dim=1)


def transform_link_control_points(frames: torch.Tensor, link_control_points: torch.Tensor) -> torch.Tensor:
    """Transform local collision control points to the world frame.

    Args:
        frames: [B, L, 4, 4]
        link_control_points: [B, L, P, 3] or [1, L, P, 3]
    Returns:
        Tensor [B, L, P, 3].
    """
    batch_size = frames.shape[0]
    points = as_batched(link_control_points, batch_size)
    if points.shape[1] != frames.shape[1]:
        raise ValueError(f"Expected {frames.shape[1]} collision-link point groups, got {points.shape[1]}")
    R = frames[:, :, :3, :3]
    t = frames[:, :, :3, 3]
    world = torch.matmul(R.unsqueeze(2), points.unsqueeze(-1)).squeeze(-1)
    return world + t.unsqueeze(2)


def link_control_points_world(
    q: torch.Tensor,
    joint_origin_xyz: torch.Tensor,
    joint_origin_rpy: torch.Tensor,
    joint_axis: torch.Tensor,
    link_control_points: torch.Tensor,
    tcp_fixed_transform: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return collision-control-point world positions [B, L, P, 3].

    If ``link_control_points`` contains seven groups, only arm frames are used.
    If it contains eight groups, the last group is attached to panda_link8/TCP
    and represents the panda_hand collision proxy.
    """
    frames = fk_urdf_all_frames(q, joint_origin_xyz, joint_origin_rpy, joint_axis)
    points_count = link_control_points.shape[1]
    if points_count == frames.shape[1]:
        collision_frames = frames
    elif points_count == frames.shape[1] + 1:
        if tcp_fixed_transform is None:
            raise ValueError("tcp_fixed_transform is required when hand collision points are present")
        hand_frame = torch.matmul(frames[:, -1], as_batched(tcp_fixed_transform, q.shape[0]))
        collision_frames = torch.cat((frames, hand_frame[:, None, :, :]), dim=1)
    else:
        raise ValueError(f"Unsupported number of collision point groups: {points_count}")
    return transform_link_control_points(collision_frames, link_control_points)


def default_tool_transform(
    batch_size: int = 1,
    *,
    dtype: torch.dtype = torch.double,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return make_eye4(batch_size, dtype=dtype, device=torch.device(device))


def tool_transform_for_tcp_link(
    tcp_link_name: str = "panda_link8",
    batch_size: int = 1,
    *,
    dtype: torch.dtype = torch.double,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the fixed transform from panda_link8 to the selected TCP frame.

    Supported frames mirror the PyBullet Panda URDF fixed links commonly used
    for the end-effector region:

    - ``panda_link8``: identity transform.
    - ``panda_hand``: fixed hand joint, rpy=(0, 0, -pi/4), xyz=(0, 0, 0).
    - ``panda_grasptarget``: panda_hand plus z=0.105 m, useful later for a
      visible grasp target / gripper-tip style TCP.

    The project default now uses ``panda_grasptarget`` because ``panda_hand``
    has the same origin as panda_link8 in the PyBullet URDF.
    """
    name = str(tcp_link_name)
    T = default_tool_transform(batch_size, dtype=dtype, device=device)
    dev = torch.device(device)
    if name in ("panda_link8", "link8", "flange"):
        return T
    if name in ("panda_hand", "hand"):
        R = _rot_z(torch.full((batch_size,), -torch.pi / 4.0, dtype=dtype, device=dev))
        T[:, :3, :3] = R
        return T
    if name in ("panda_grasptarget", "grasptarget", "tool_tip"):
        R = _rot_z(torch.full((batch_size,), -torch.pi / 4.0, dtype=dtype, device=dev))
        T[:, :3, :3] = R
        T[:, 2, 3] = 0.105
        return T
    raise ValueError(
        f"Unsupported TCP link {tcp_link_name!r}. Supported: "
        "panda_link8, panda_hand, panda_grasptarget"
    )
