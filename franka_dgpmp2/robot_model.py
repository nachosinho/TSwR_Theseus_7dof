"""Hard-coded Franka Panda URDF kinematics and collision control points."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


# Seven arm frames plus one additional sphere cloud attached to the Panda
# hand/TCP-region. The last group is not an actuated joint; it is transformed by
# the panda_link8/TCP frame and is used only for obstacle clearance.
COLLISION_LINK_NAMES = (
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link8_axis",
    "panda_hand",
)
NUM_COLLISION_LINKS = len(COLLISION_LINK_NAMES)


def collision_link_name(link_idx: int) -> str:
    if 0 <= int(link_idx) < len(COLLISION_LINK_NAMES):
        return COLLISION_LINK_NAMES[int(link_idx)]
    return f"collision_link_{int(link_idx)}"


@dataclass(frozen=True)
class RobotURDFKinematics:
    joint_origin_xyz: torch.Tensor       # [7, 3]
    joint_origin_rpy: torch.Tensor       # [7, 3]
    joint_axis: torch.Tensor             # [7, 3]
    joint_min: torch.Tensor              # [7]
    joint_max: torch.Tensor              # [7]
    joint_velocity: torch.Tensor         # [7]
    joint_effort: torch.Tensor           # [7]
    tcp_fixed_transform: torch.Tensor    # [4, 4], panda_link7 -> panda_link8/TCP reference frame

    @staticmethod
    def franka_panda(dtype: torch.dtype = torch.double, device: torch.device | str = "cpu") -> "RobotURDFKinematics":
        """Return Panda arm constants matching pybullet_data/franka_panda/panda.urdf."""
        device = torch.device(device)
        pi = math.pi

        joint_origin_xyz = torch.tensor(
            [
                [0.0, 0.0, 0.333],
                [0.0, 0.0, 0.0],
                [0.0, -0.316, 0.0],
                [0.0825, 0.0, 0.0],
                [-0.0825, 0.384, 0.0],
                [0.0, 0.0, 0.0],
                [0.088, 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        joint_origin_rpy = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [-pi / 2.0, 0.0, 0.0],
                [pi / 2.0, 0.0, 0.0],
                [pi / 2.0, 0.0, 0.0],
                [-pi / 2.0, 0.0, 0.0],
                [pi / 2.0, 0.0, 0.0],
                [pi / 2.0, 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        joint_axis = torch.tensor([[0.0, 0.0, 1.0]] * 7, dtype=dtype, device=device)

        joint_min = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            dtype=dtype,
            device=device,
        )
        joint_max = torch.tensor(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            dtype=dtype,
            device=device,
        )
        joint_velocity = torch.tensor([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610], dtype=dtype, device=device)
        joint_effort = torch.tensor([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=dtype, device=device)

        tcp_fixed_transform = torch.eye(4, dtype=dtype, device=device)
        tcp_fixed_transform[2, 3] = 0.107

        return RobotURDFKinematics(
            joint_origin_xyz=joint_origin_xyz,
            joint_origin_rpy=joint_origin_rpy,
            joint_axis=joint_axis,
            joint_min=joint_min,
            joint_max=joint_max,
            joint_velocity=joint_velocity,
            joint_effort=joint_effort,
            tcp_fixed_transform=tcp_fixed_transform,
        )


def _perpendicular_basis_from_segment(segment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    direction = segment / torch.clamp(torch.linalg.norm(segment), min=1e-12)
    ref_z = torch.tensor([0.0, 0.0, 1.0], dtype=segment.dtype, device=segment.device)
    ref_x = torch.tensor([1.0, 0.0, 0.0], dtype=segment.dtype, device=segment.device)
    ref = torch.where(torch.abs(torch.dot(direction, ref_z)) > 0.85, ref_x, ref_z)
    u = torch.linalg.cross(direction, ref, dim=0)
    u = u / torch.clamp(torch.linalg.norm(u), min=1e-12)
    v = torch.linalg.cross(direction, u, dim=0)
    v = v / torch.clamp(torch.linalg.norm(v), min=1e-12)
    return u, v


def _segment_sphere_cloud(segment: torch.Tensor, half_width: torch.Tensor, points_per_link: int) -> torch.Tensor:
    """Create centerline + cross-section offsets for one link local frame."""
    offsets_per_station = 5
    stations = max(1, math.ceil(points_per_link / offsets_per_station))
    alphas = (
        torch.tensor([0.5], dtype=segment.dtype, device=segment.device)
        if stations == 1
        else torch.linspace(0.08, 0.92, stations, dtype=segment.dtype, device=segment.device)
    )
    u, v = _perpendicular_basis_from_segment(segment)
    offsets = torch.stack(
        (
            torch.zeros(3, dtype=segment.dtype, device=segment.device),
            half_width * u,
            -half_width * u,
            half_width * v,
            -half_width * v,
        ),
        dim=0,
    )
    candidates = (alphas[:, None, None] * segment[None, None, :] + offsets[None, :, :]).reshape(-1, 3)
    if candidates.shape[0] == points_per_link:
        return candidates
    idx = torch.linspace(0, candidates.shape[0] - 1, points_per_link, dtype=torch.float64, device=segment.device).round().long()
    return candidates[idx]


def _hand_sphere_cloud(points_per_link: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Approximate the Panda hand around the TCP/link8 frame.

    These local points are intentionally conservative. They do not replace the
    PyBullet URDF geometry; they give Theseus a differentiable proxy for the
    hand body, which PyBullet previously reported as the closest colliding link.
    """
    # Five stations along the local z direction and five cross-section offsets
    # per station -> 25 points by default. The values are approximate but chosen
    # to cover the hand/flange volume around panda_link8 while keeping one global
    # collision radius.
    z_values = torch.linspace(-0.035, 0.085, max(1, math.ceil(points_per_link / 5)), dtype=dtype, device=device)
    xy_width = torch.tensor(0.055, dtype=dtype, device=device)
    offsets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    offsets[:, :2] *= xy_width
    candidates = []
    for z in z_values:
        base = torch.stack((torch.zeros((), dtype=dtype, device=device), torch.zeros((), dtype=dtype, device=device), z))
        candidates.append(base[None, :] + offsets)
    cloud = torch.cat(candidates, dim=0)
    if cloud.shape[0] == points_per_link:
        return cloud
    idx = torch.linspace(0, cloud.shape[0] - 1, points_per_link, dtype=torch.float64, device=device).round().long()
    return cloud[idx]


def default_link_control_points(
    *,
    points_per_link: int = 25,
    dtype: torch.dtype = torch.double,
    device: torch.device | str = "cpu",
    include_hand_points: bool = True,
) -> torch.Tensor:
    """Return a dense sphere-cloud approximation [1, L, P, 3].

    L is 8 by default: seven arm-link point clouds plus one additional
    panda_hand/TCP-region point cloud. Each point is later inflated by the single
    global ``link_collision_radius`` used in the collision cost.
    """
    if points_per_link < 1:
        raise ValueError("points_per_link must be >= 1")

    device = torch.device(device)
    link_segment_ends = torch.tensor(
        [
            [0.00, 0.00, 0.12],
            [0.00, -0.316, 0.00],
            [0.0825, 0.00, 0.00],
            [-0.0825, 0.384, 0.00],
            [0.00, 0.00, 0.16],
            [0.088, 0.00, 0.00],
            [0.00, 0.00, 0.107],
        ],
        dtype=dtype,
        device=device,
    )
    cross_section_half_width = torch.tensor(
        [0.030, 0.035, 0.030, 0.040, 0.045, 0.030, 0.025],
        dtype=dtype,
        device=device,
    )

    link_points = []
    for link_idx in range(7):
        link_points.append(_segment_sphere_cloud(link_segment_ends[link_idx], cross_section_half_width[link_idx], points_per_link))

    if include_hand_points:
        link_points.append(_hand_sphere_cloud(points_per_link, dtype=dtype, device=device))

    return torch.stack(link_points, dim=0).unsqueeze(0)
