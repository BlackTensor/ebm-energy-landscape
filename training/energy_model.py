"""Energy network for the Energy Landscape Visualizer.

This module defines the model at the heart of the project: a function

    E(scene, trajectory) -> scalar

that scores how plausible a trajectory is for a given scene. Low energy means
"this looks like a valid route through this scene"; high energy means "this is a
bad path". The training loop (Phase 3) shapes that scalar so valleys form
around real routes, and the Langevin sampler (Phase 4) walks downhill in it to
*generate* new trajectories. Because generation runs by following the gradient
of the energy with respect to the trajectory, the whole network is built to be
smoothly differentiable in its trajectory input (see Phase 2.3).

Architecture (as specified for Task 2.1)
---------------------------------------

Three stages, combined by a head MLP:

1. :class:`MapEncoder` — a small CNN. The scene (start, goal, circular
   obstacles) is rasterised into a fixed-resolution, multi-channel image and
   convolved down to a single ``map_dim`` embedding vector. This is the
   conditioning context: it does not depend on the trajectory.
2. :class:`TrajectoryEncoder` — a sequence encoder (a bidirectional LSTM). The
   ordered ``(x, y)`` points, augmented with per-step velocity, are read in
   sequence and pooled to a single ``traj_dim`` embedding vector.
3. :class:`EnergyHead` — an MLP that takes the concatenated map and trajectory
   embeddings and emits one scalar energy per sample.

:class:`EnergyModel` wires the three together. Its ``forward`` takes a batch of
rasterised maps ``(B, C, H, W)`` and a batch of trajectories ``(B, N, 2)`` and
returns a batch of scalar energies ``(B,)``.

Design choices worth knowing
----------------------------

- **GroupNorm, not BatchNorm.** Energy-based models are evaluated one sample at
  a time during Langevin sampling and trained on small contrastive batches.
  Normalisation must not couple samples through batch statistics, so every
  normalisation layer is batch-size independent.
- **Smooth activations (SiLU).** Langevin dynamics follows ``grad(E)`` through
  the trajectory; piecewise-constant gradients from hard ReLUs make that walk
  jittery. Smooth activations give a smooth landscape to descend.
- **The map path carries no trajectory gradient.** Rasterisation happens once
  per scene in numpy and is a fixed conditioning tensor; only the trajectory
  branch needs to be differentiable for sampling.

The rasteriser :func:`rasterize_world` lives here too so the map representation
the CNN expects is defined in one place. Only numpy and torch are required, so
this runs unchanged on the Colab free tier and locally.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

try:  # works when imported as part of the ``training`` package
    from training.environment import GridWorld
    from training.trajectory import Trajectory
except ImportError:  # falls back when run directly as ``python energy_model.py``
    from environment import GridWorld
    from trajectory import Trajectory

__all__ = [
    "MAP_CHANNELS",
    "rasterize_world",
    "worlds_to_tensor",
    "trajectories_to_tensor",
    "MapEncoder",
    "TrajectoryEncoder",
    "EnergyHead",
    "EnergyModel",
]

# The rasterised scene has three channels, defined by :func:`rasterize_world`:
#   0 - obstacle field   (smooth signed distance; negative inside an obstacle)
#   1 - distance to start (normalised radial field)
#   2 - distance to goal  (normalised radial field)
MAP_CHANNELS = 3


# --- scene rasterisation ----------------------------------------------------


def rasterize_world(world: GridWorld, resolution: int = 64) -> np.ndarray:
    """Render a :class:`GridWorld` into the ``(C, H, W)`` image the CNN reads.

    Channels (see :data:`MAP_CHANNELS`):

    0. **Obstacle field.** The signed distance to the nearest obstacle boundary,
       squashed through ``tanh`` so it is smooth and bounded: strongly negative
       deep inside an obstacle, near zero on the boundary, near +1 in open
       space. Scenes with no obstacles read as open space everywhere.
    1. **Distance to start.** Euclidean distance from each pixel to the start,
       normalised by the world's diagonal so it lives in ``[0, 1]``.
    2. **Distance to goal.** Same, measured to the goal.

    The grid spans ``[0, size] x [0, size]``. Row index increases with ``y`` and
    column index with ``x`` (standard image convention); the model only needs
    this to be internally consistent, which it is by construction.

    Returns a ``float32`` array of shape ``(MAP_CHANNELS, resolution,
    resolution)``.
    """
    if resolution < 1:
        raise ValueError(f"resolution must be positive; got {resolution}.")

    xs = np.linspace(0.0, world.size, resolution)
    ys = np.linspace(0.0, world.size, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)  # (H, W), row -> y, col -> x
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (H*W, 2)

    # Obstacle field: smooth signed distance. With no obstacles the distance is
    # +inf, which we treat as "far / open space".
    sdf = world.obstacle_signed_distance(grid)
    sdf = np.where(np.isfinite(sdf), sdf, world.size)
    obstacle = np.tanh(sdf / (0.1 * world.size)).reshape(resolution, resolution)

    diag = np.sqrt(2.0) * world.size
    dist_start = (
        np.linalg.norm(grid - np.asarray(world.start, float), axis=1) / diag
    ).reshape(resolution, resolution)
    dist_goal = (
        np.linalg.norm(grid - np.asarray(world.goal, float), axis=1) / diag
    ).reshape(resolution, resolution)

    channels = np.stack([obstacle, dist_start, dist_goal], axis=0)
    return channels.astype(np.float32)


def worlds_to_tensor(
    worlds: Sequence[GridWorld],
    resolution: int = 64,
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Rasterise a list of scenes into one ``(B, C, H, W)`` float tensor."""
    if len(worlds) == 0:
        raise ValueError("Cannot rasterise an empty list of worlds.")
    stacked = np.stack([rasterize_world(w, resolution) for w in worlds], axis=0)
    return torch.from_numpy(stacked).to(device=device, dtype=torch.float32)


def trajectories_to_tensor(
    trajectories: Sequence[Trajectory],
    *,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """Stack equal-length trajectories into one ``(B, N, 2)`` float tensor.

    Set ``requires_grad=True`` to obtain a leaf tensor suitable for Langevin
    sampling, where the energy is differentiated with respect to the path.
    """
    if len(trajectories) == 0:
        raise ValueError("Cannot stack an empty list of trajectories.")
    arrays = np.stack([t.points for t in trajectories], axis=0)
    tensor = torch.from_numpy(arrays).to(device=device, dtype=torch.float32)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


# --- sub-networks -----------------------------------------------------------


def _group_norm(channels: int, *, max_groups: int = 8) -> nn.GroupNorm:
    """A GroupNorm whose group count divides ``channels`` and stays small.

    GroupNorm keeps normalisation independent of batch size, which matters for an
    EBM evaluated one sample at a time during sampling.
    """
    groups = max_groups
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class MapEncoder(nn.Module):
    """CNN that turns a rasterised scene ``(B, C, H, W)`` into ``(B, map_dim)``.

    A short stack of stride-2 convolutions halves the spatial resolution at each
    step, then global average pooling and a linear projection produce a single
    embedding per scene.
    """

    def __init__(
        self,
        in_channels: int = MAP_CHANNELS,
        map_dim: int = 128,
        widths: Sequence[int] = (32, 64, 128),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for width in widths:
            layers += [
                nn.Conv2d(prev, width, kernel_size=3, stride=2, padding=1),
                _group_norm(width),
                nn.SiLU(),
            ]
            prev = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(prev, map_dim)
        self.map_dim = map_dim

    def forward(self, maps: Tensor) -> Tensor:
        if maps.dim() != 4:
            raise ValueError(
                f"MapEncoder expects (B, C, H, W); got shape {tuple(maps.shape)}."
            )
        feats = self.features(maps)
        pooled = self.pool(feats).flatten(1)  # (B, width)
        return self.project(pooled)  # (B, map_dim)


class TrajectoryEncoder(nn.Module):
    """Bidirectional LSTM that turns a path ``(B, N, 2)`` into ``(B, traj_dim)``.

    Each point is augmented with its per-step velocity ``(dx, dy)`` so the
    encoder sees motion directly, then a linear layer lifts the 4 features to the
    recurrent width. The LSTM reads the sequence both ways and the outputs are
    mean-pooled over time into a single embedding. Coordinates are divided by
    ``coord_scale`` (the world size) to keep inputs near unit range. All
    operations are smooth in the input coordinates so the energy stays
    differentiable with respect to the trajectory.
    """

    def __init__(
        self,
        traj_dim: int = 128,
        hidden: int = 128,
        num_layers: int = 1,
        coord_scale: float = 1.0,
    ):
        super().__init__()
        self.coord_scale = float(coord_scale)
        self.input_proj = nn.Linear(4, hidden)  # (x, y, dx, dy)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.project = nn.Linear(2 * hidden, traj_dim)
        self.act = nn.SiLU()
        self.traj_dim = traj_dim

    def forward(self, trajectory: Tensor) -> Tensor:
        if trajectory.dim() != 3 or trajectory.shape[-1] != 2:
            raise ValueError(
                f"TrajectoryEncoder expects (B, N, 2); got shape "
                f"{tuple(trajectory.shape)}."
            )
        coords = trajectory / self.coord_scale

        # Per-step velocity, padded at the front so it lines up point-for-point
        # with the coordinates and keeps the sequence length N.
        velocity = coords[:, 1:, :] - coords[:, :-1, :]
        velocity = torch.cat([velocity[:, :1, :], velocity], dim=1)

        features = torch.cat([coords, velocity], dim=-1)  # (B, N, 4)
        embedded = self.act(self.input_proj(features))
        outputs, _ = self.lstm(embedded)  # (B, N, 2 * hidden)
        pooled = outputs.mean(dim=1)  # (B, 2 * hidden)
        return self.project(pooled)  # (B, traj_dim)


class EnergyHead(nn.Module):
    """MLP mapping concatenated map+trajectory embeddings to a scalar energy."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, joint: Tensor) -> Tensor:
        return self.net(joint).squeeze(-1)  # (B,)


# --- full model -------------------------------------------------------------


class EnergyModel(nn.Module):
    """The scalar energy ``E(scene, trajectory)`` for the whole project.

    Parameters
    ----------
    map_channels:
        Channels in the rasterised scene; must match :func:`rasterize_world`.
    map_dim, traj_dim:
        Embedding widths produced by the two encoders.
    hidden:
        Width of the recurrent encoder and the energy head MLP.
    lstm_layers:
        Depth of the trajectory LSTM.
    coord_scale:
        Trajectory coordinates are divided by this (the world size) before
        encoding, keeping inputs near unit range.

    The ``forward`` pass takes ``maps`` of shape ``(B, map_channels, H, W)`` and
    ``trajectory`` of shape ``(B, N, 2)`` and returns energies of shape
    ``(B,)`` — one scalar per sample.
    """

    def __init__(
        self,
        map_channels: int = MAP_CHANNELS,
        map_dim: int = 128,
        traj_dim: int = 128,
        hidden: int = 128,
        lstm_layers: int = 1,
        coord_scale: float = 1.0,
    ):
        super().__init__()
        # Record the constructor arguments so a saved checkpoint can rebuild the
        # exact architecture before loading weights (used by the training
        # checkpoints and by the Phase 4/5 samplers that reload this model).
        self.init_kwargs = {
            "map_channels": map_channels,
            "map_dim": map_dim,
            "traj_dim": traj_dim,
            "hidden": hidden,
            "lstm_layers": lstm_layers,
            "coord_scale": coord_scale,
        }
        self.map_encoder = MapEncoder(map_channels, map_dim)
        self.trajectory_encoder = TrajectoryEncoder(
            traj_dim, hidden=hidden, num_layers=lstm_layers, coord_scale=coord_scale
        )
        self.head = EnergyHead(map_dim + traj_dim, hidden=hidden)

    def forward(self, maps: Tensor, trajectory: Tensor) -> Tensor:
        if maps.shape[0] != trajectory.shape[0]:
            raise ValueError(
                f"Batch size mismatch: maps has {maps.shape[0]} but trajectory "
                f"has {trajectory.shape[0]}."
            )
        map_emb = self.map_encoder(maps)
        traj_emb = self.trajectory_encoder(trajectory)
        joint = torch.cat([map_emb, traj_emb], dim=-1)
        return self.head(joint)  # (B,)


def _demo() -> None:
    """Textual self-check. Run ``python energy_model.py`` for a forward pass."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # Build a small batch of real scenes and matching trajectories so the demo
    # exercises the actual rasteriser and tensor helpers, not just random noise.
    from generators import generate_valid_trajectory  # local import for demo

    worlds = [GridWorld.random(rng, n_obstacles=(3, 5)) for _ in range(4)]
    trajectories = [generate_valid_trajectory(w, rng, n_points=48) for w in worlds]

    maps = worlds_to_tensor(worlds, resolution=64)
    paths = trajectories_to_tensor(trajectories)

    model = EnergyModel()
    n_params = sum(p.numel() for p in model.parameters())

    model.eval()
    with torch.no_grad():
        energy = model(maps, paths)

    print("EnergyModel")
    print(f"  parameters     : {n_params:,}")
    print(f"  map tensor     : {tuple(maps.shape)}")
    print(f"  path tensor    : {tuple(paths.shape)}")
    print(f"  energy shape   : {tuple(energy.shape)}")
    print(f"  energy values  : {[round(float(e), 4) for e in energy]}")

    # The defining contract of an energy function: one scalar per sample.
    assert energy.shape == (len(worlds),), "energy must be one scalar per sample"
    assert torch.all(torch.isfinite(energy)), "energy must be finite"
    print("  self-check passed.")


if __name__ == "__main__":
    _demo()
