"""Small tensor helpers shared by kinematics, costs and kinematics."""

from __future__ import annotations

from typing import Dict

import torch

TensorDict = Dict[str, torch.Tensor]


def as_batched(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Expand a tensor with leading dimension 1 to batch_size."""
    if x.shape[0] == batch_size:
        return x
    if x.shape[0] == 1:
        return x.expand(batch_size, *x.shape[1:])
    raise ValueError(f"Tensor batch {x.shape[0]} is incompatible with batch_size={batch_size}.")


def scalar_like(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return a scalar tensor as [B, 1]."""
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.reshape(-1, 1)
    return as_batched(x, batch_size)


def make_eye4(batch_size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a batch of homogeneous identity matrices [B, 4, 4]."""
    return torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(batch_size, 1, 1)


def tensor_from_value(value: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Create a Theseus-friendly scalar tensor with shape [1, 1]."""
    return torch.tensor([[value]], dtype=dtype, device=device)
