"""Neural warm-start helpers for the Panda GPMP2/Theseus demo.

The neural model does not produce the final robot trajectory. It only predicts
an offset for the initial trajectory. The full Theseus/GPMP2 optimizer is still
responsible for the final trajectory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


MAX_OBSTACLES_DEFAULT = 4
MISSING_OBSTACLE_CENTER = (10.0, 10.0, 10.0)



class WarmStartMLP(torch.nn.Module):
    """Small MLP that predicts corrections for internal trajectory knots."""

    def __init__(self, input_dim: int, num_internal_knots: int, dof: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.num_internal_knots = num_internal_knots
        self.dof = dof
        self.hidden_dim = hidden_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_internal_knots * dof),
        )
        # Start from the ordinary linear warm-start. Training then learns an offset.
        last = self.net[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

    def forward(self, features: torch.Tensor, *, max_delta: float) -> torch.Tensor:
        raw = self.net(features)
        delta = max_delta * torch.tanh(raw)
        return delta.view(-1, self.num_internal_knots, self.dof)



def _as_2d_centers(centers, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(centers):
        t = centers.to(dtype=dtype, device=device)
    else:
        t = torch.tensor(centers, dtype=dtype, device=device)
    if t.numel() == 0:
        return torch.empty(0, 3, dtype=dtype, device=device)
    return t.reshape(-1, 3)


def _as_1d_radii(radii, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(radii):
        t = radii.to(dtype=dtype, device=device)
    else:
        t = torch.tensor(radii, dtype=dtype, device=device)
    if t.numel() == 0:
        return torch.empty(0, dtype=dtype, device=device)
    return t.reshape(-1)


def pad_obstacles_for_features(
    obstacle_centers,
    obstacle_radii,
    *,
    max_obstacles: int = MAX_OBSTACLES_DEFAULT,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad obstacle features to a fixed size.

    Missing obstacles are encoded as zeros in the MLP feature vector.
    This gives a stable input dimension independent of the number of obstacles.
    """
    centers = _as_2d_centers(obstacle_centers, dtype=dtype, device=device)
    radii = _as_1d_radii(obstacle_radii, dtype=dtype, device=device)
    if centers.shape[0] != radii.shape[0]:
        raise ValueError(f"centers/radii count mismatch: {centers.shape[0]} vs {radii.shape[0]}")
    if centers.shape[0] > max_obstacles:
        raise ValueError(f"got {centers.shape[0]} obstacles, but max_obstacles={max_obstacles}")
    out_centers = torch.zeros(max_obstacles, 3, dtype=dtype, device=device)
    out_radii = torch.zeros(max_obstacles, dtype=dtype, device=device)
    n = centers.shape[0]
    if n:
        out_centers[:n] = centers
        out_radii[:n] = radii
    return out_centers, out_radii


def pad_obstacles_for_planner(
    obstacle_centers,
    obstacle_radii,
    *,
    max_obstacles: int = MAX_OBSTACLES_DEFAULT,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad obstacles for Theseus collision costs.

    Missing obstacles are moved far away, because radius=0 at (0,0,0) would still
    create an artificial keep-away region due to the robot link collision radius.
    """
    centers = _as_2d_centers(obstacle_centers, dtype=dtype, device=device)
    radii = _as_1d_radii(obstacle_radii, dtype=dtype, device=device)
    if centers.shape[0] != radii.shape[0]:
        raise ValueError(f"centers/radii count mismatch: {centers.shape[0]} vs {radii.shape[0]}")
    if centers.shape[0] > max_obstacles:
        raise ValueError(f"got {centers.shape[0]} obstacles, but max_obstacles={max_obstacles}")
    far = torch.tensor(MISSING_OBSTACLE_CENTER, dtype=dtype, device=device)
    out_centers = far.repeat(max_obstacles, 1)
    out_radii = torch.zeros(max_obstacles, dtype=dtype, device=device)
    n = centers.shape[0]
    if n:
        out_centers[:n] = centers
        out_radii[:n] = radii
    return out_centers, out_radii

def build_scene_features(
    q_start: torch.Tensor,
    q_target: torch.Tensor,
    obstacle_centers: torch.Tensor | Sequence[Sequence[float]],
    obstacle_radii: torch.Tensor | Sequence[float],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    max_obstacles: int = MAX_OBSTACLES_DEFAULT,
) -> torch.Tensor:
    """Feature vector used by the warm-start MLP.

    Features are intentionally simple and deterministic:
        [q_start, q_target, q_target - q_start, obstacle_centers, obstacle_radii]
    """
    dtype = dtype or q_start.dtype
    device = device or q_start.device
    q_start = q_start.to(dtype=dtype, device=device).reshape(1, -1)
    q_target = q_target.to(dtype=dtype, device=device).reshape(1, -1)
    q_delta = q_target - q_start

    obstacle_centers, obstacle_radii = pad_obstacles_for_features(
        obstacle_centers,
        obstacle_radii,
        max_obstacles=max_obstacles,
        dtype=dtype,
        device=device,
    )

    obstacle_centers = obstacle_centers.reshape(1, -1)
    obstacle_radii = obstacle_radii.reshape(1, -1)
    return torch.cat([q_start, q_target, q_delta, obstacle_centers, obstacle_radii], dim=1)


def interpolate_delta_to_num_steps(
    delta_internal: torch.Tensor,
    *,
    source_num_steps: int,
    target_num_steps: int,
) -> torch.Tensor:
    """Interpolate an internal-knot correction to another trajectory resolution."""
    if delta_internal.ndim != 3:
        raise ValueError(f"delta_internal must have shape [B, K, dof], got {tuple(delta_internal.shape)}")
    batch, _, dof = delta_internal.shape
    delta_source = torch.zeros(
        batch,
        source_num_steps + 1,
        dof,
        dtype=delta_internal.dtype,
        device=delta_internal.device,
    )
    delta_source[:, 1:-1, :] = delta_internal
    delta_interp = F.interpolate(
        delta_source.permute(0, 2, 1),
        size=target_num_steps + 1,
        mode="linear",
        align_corners=True,
    ).permute(0, 2, 1)
    delta_interp[:, 0, :] = 0.0
    delta_interp[:, -1, :] = 0.0
    return delta_interp


def load_warmstart_checkpoint(path: str | Path, *, dtype: torch.dtype, device: torch.device):
    """Load a trained warm-start MLP checkpoint."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    input_dim = int(ckpt["input_dim"])
    num_internal_knots = int(ckpt["num_internal_knots"])
    hidden_dim = int(ckpt.get("hidden_dim", 128))
    dof = int(ckpt.get("dof", 7))
    model = WarmStartMLP(input_dim, num_internal_knots, dof=dof, hidden_dim=hidden_dim).to(dtype=dtype, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def predict_warmstart_q_init(
    *,
    q_init_linear: torch.Tensor,
    q_start: torch.Tensor,
    q_target: torch.Tensor,
    obstacle_centers,
    obstacle_radii,
    checkpoint_path: str | Path,
    target_num_steps: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Return q_init produced by linear interpolation plus neural correction."""
    model, ckpt = load_warmstart_checkpoint(checkpoint_path, dtype=dtype, device=device)
    max_delta = float(ckpt.get("max_delta", 0.20))
    source_num_steps = int(ckpt.get("train_steps", ckpt["num_internal_knots"] + 1))
    max_obstacles = int(ckpt.get("max_obstacles", MAX_OBSTACLES_DEFAULT))

    features = build_scene_features(
        q_start,
        q_target,
        obstacle_centers,
        obstacle_radii,
        dtype=dtype,
        device=device,
        max_obstacles=max_obstacles,
    )
    if features.shape[1] != int(ckpt["input_dim"]):
        raise ValueError(
            f"Checkpoint expects input_dim={ckpt['input_dim']}, but scene features have {features.shape[1]}."
        )

    with torch.no_grad():
        delta_internal = model(features, max_delta=max_delta)
        delta_full = interpolate_delta_to_num_steps(
            delta_internal,
            source_num_steps=source_num_steps,
            target_num_steps=target_num_steps,
        )
        q_init_learned = q_init_linear.to(dtype=dtype, device=device) + delta_full
        # Keep endpoints exact.
        q_init_learned[:, 0, :] = q_init_linear[:, 0, :]
        q_init_learned[:, -1, :] = q_init_linear[:, -1, :]
    return q_init_learned, delta_full, ckpt