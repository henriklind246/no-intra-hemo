"""Pointwise MLP baseline.

Every point is predicted on its own from its 8 normalized point inputs
(coordinates, wall distance, the four geodesic distances), the case's flow
split and the prediction time:

    [x, y, z, d_wall, d_inlet, d_outlet3, d_outlet4, d_aneurysm,
     flow_split (2), sin/cos(2 pi k phase) for k = 1..H]  ->  u, v, w, p_gauge

It sees no neighbours, so everything it knows about the geometry comes from
the distance features; how far that gets is what the baseline measures.

The first layer is split into a per-point and a per-(case, time) part that
are added, so the global inputs are never copied to every point. Optional
random Fourier features of the coordinates (``fourier_features > 0``) help an
MLP represent sharper spatial detail.

Interface shared by all models here (see ``src.models.build_model``):
    forward(inputs (B, K, P), flow_split (B, S), time (B, s)) -> (B, s, K, 4)
with normalized inputs and outputs, as produced by ``src.data.dataset``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

ACTIVATIONS = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}


def time_features(phase: torch.Tensor, harmonics: int) -> torch.Tensor:
    """(..., ) phase in [0, 1) -> (..., 2 * harmonics) sin/cos features (periodic in the phase)."""
    k = torch.arange(1, harmonics + 1, device=phase.device, dtype=phase.dtype)
    a = 2 * math.pi * phase[..., None] * k
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class PointwiseMLP(nn.Module):
    def __init__(self, num_point_inputs: int = 8, num_flow_split: int = 2, num_outputs: int = 4, *,
                 width: int = 256, depth: int = 6, activation: str = "gelu", time_harmonics: int = 4,
                 fourier_features: int = 0, fourier_scale: float = 1.0, fourier_seed: int = 0):
        super().__init__()
        if depth < 1 or width < 1:
            raise ValueError("depth and width must be at least 1")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
        self.config = dict(num_point_inputs=num_point_inputs, num_flow_split=num_flow_split,
                           num_outputs=num_outputs, width=width, depth=depth, activation=activation,
                           time_harmonics=time_harmonics, fourier_features=fourier_features,
                           fourier_scale=fourier_scale, fourier_seed=fourier_seed)
        self.time_harmonics = time_harmonics
        if fourier_features > 0:
            g = torch.Generator().manual_seed(fourier_seed)
            self.register_buffer("fourier_matrix", torch.randn(3, fourier_features, generator=g) * fourier_scale)
        else:
            self.fourier_matrix: Optional[torch.Tensor] = None
        point_dim = num_point_inputs + 2 * fourier_features
        global_dim = num_flow_split + 2 * time_harmonics
        self.point_in = nn.Linear(point_dim, width)
        self.global_in = nn.Linear(global_dim, width, bias=False)
        act = ACTIVATIONS[activation]
        layers = []
        for _ in range(depth - 1):
            layers += [act(), nn.Linear(width, width)]
        layers += [act(), nn.Linear(width, num_outputs)]
        self.body = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, flow_split: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        x = inputs
        if self.fourier_matrix is not None:
            proj = 2 * math.pi * x[..., :3] @ self.fourier_matrix
            x = torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)
        s = time.shape[1]
        g = torch.cat([flow_split[:, None, :].expand(-1, s, -1), time_features(time, self.time_harmonics)], -1)
        h = self.point_in(x)[:, None, :, :] + self.global_in(g)[:, :, None, :]  # (B, s, K, W)
        return self.body(h)
