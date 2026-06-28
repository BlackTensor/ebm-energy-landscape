"""End-to-end shape verification for the energy network (Task 2.2).

This script drives dummy forward passes through :mod:`energy_model` and asserts
the shape contract at every stage, so a refactor that quietly breaks a tensor
shape fails loudly here instead of deep inside training or sampling.

It uses random tensors on purpose: the point is the *plumbing*, not whether the
energies mean anything yet (training shapes the values later). The one contract
that matters above all is checked explicitly and repeatedly:

    the model emits exactly one scalar energy per sample -> output shape (B,).

Stages verified, for several batch sizes / point counts / map resolutions:

- ``MapEncoder``         : (B, C, H, W)  -> (B, map_dim)
- ``TrajectoryEncoder``  : (B, N, 2)     -> (B, traj_dim)
- ``EnergyHead``         : (B, D)        -> (B,)
- ``EnergyModel``        : maps + path   -> (B,)

Run ``python verify_shapes.py`` (or ``python -m training.verify_shapes``).
"""

from __future__ import annotations

import itertools

import torch

try:  # works when imported as part of the ``training`` package
    from training.energy_model import (
        MAP_CHANNELS,
        EnergyHead,
        EnergyModel,
        MapEncoder,
        TrajectoryEncoder,
    )
except ImportError:  # falls back when run directly as ``python verify_shapes.py``
    from energy_model import (
        MAP_CHANNELS,
        EnergyHead,
        EnergyModel,
        MapEncoder,
        TrajectoryEncoder,
    )

# Embedding widths used across the checks; kept in one place so the head's
# input dimension always matches the two encoders.
MAP_DIM = 128
TRAJ_DIM = 128

# Configurations to sweep: batch size, trajectory point count, map resolution.
# Includes a single-sample batch (B=1) since sampling evaluates one path at a
# time, and varies N and resolution to confirm nothing is hard-coded.
BATCH_SIZES = (1, 4, 16)
POINT_COUNTS = (2, 32, 48, 96)
RESOLUTIONS = (32, 64)


def _check(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    """Assert a shape and print a one-line confirmation."""
    assert actual == expected, f"{name}: expected {expected}, got {actual}"
    print(f"  ok  {name:<34} {str(actual)}")


def _verify_submodules() -> None:
    """Each sub-network maps its input to the documented output shape."""
    print("Sub-network shapes")
    torch.manual_seed(0)

    map_encoder = MapEncoder(MAP_CHANNELS, MAP_DIM).eval()
    traj_encoder = TrajectoryEncoder(TRAJ_DIM).eval()
    head = EnergyHead(MAP_DIM + TRAJ_DIM).eval()

    b, n, res = 4, 48, 64
    with torch.no_grad():
        maps = torch.randn(b, MAP_CHANNELS, res, res)
        paths = torch.randn(b, n, 2)

        map_emb = map_encoder(maps)
        traj_emb = traj_encoder(paths)
        joint = torch.cat([map_emb, traj_emb], dim=-1)
        energy = head(joint)

    _check("MapEncoder (B,C,H,W)->(B,map_dim)", tuple(map_emb.shape), (b, MAP_DIM))
    _check("TrajectoryEncoder (B,N,2)->(B,td)", tuple(traj_emb.shape), (b, TRAJ_DIM))
    _check("EnergyHead (B,D)->(B,)", tuple(energy.shape), (b,))


def _verify_full_model() -> None:
    """The full model returns one scalar per sample for every configuration."""
    print("Full model: one scalar energy per sample")
    torch.manual_seed(0)
    model = EnergyModel(map_dim=MAP_DIM, traj_dim=TRAJ_DIM).eval()

    for b, n, res in itertools.product(BATCH_SIZES, POINT_COUNTS, RESOLUTIONS):
        with torch.no_grad():
            maps = torch.randn(b, MAP_CHANNELS, res, res)
            paths = torch.randn(b, n, 2)
            energy = model(maps, paths)

        # The core contract: a single scalar per sample, all finite.
        assert energy.shape == (b,), (
            f"B={b} N={n} res={res}: energy must be (B,), got {tuple(energy.shape)}"
        )
        assert energy.dim() == 1, "energy must be 1-D (one scalar per sample)"
        assert torch.all(torch.isfinite(energy)), "energy must be finite"
        print(f"  ok  B={b:<2} N={n:<3} res={res:<3} -> energy {tuple(energy.shape)}")


def _verify_rejects_bad_shapes() -> None:
    """Malformed inputs are rejected rather than silently mis-broadcast."""
    print("Input guards")
    model = EnergyModel(map_dim=MAP_DIM, traj_dim=TRAJ_DIM).eval()

    cases = {
        "map missing channel dim": (torch.randn(4, 64, 64), torch.randn(4, 48, 2)),
        "trajectory not (N,2)": (
            torch.randn(4, MAP_CHANNELS, 64, 64),
            torch.randn(4, 48, 3),
        ),
        "batch size mismatch": (
            torch.randn(4, MAP_CHANNELS, 64, 64),
            torch.randn(8, 48, 2),
        ),
    }
    for label, (maps, paths) in cases.items():
        try:
            model(maps, paths)
        except (ValueError, RuntimeError):
            print(f"  ok  rejected: {label}")
        else:
            raise AssertionError(f"expected an error for bad input: {label}")


def main() -> None:
    _verify_submodules()
    _verify_full_model()
    _verify_rejects_bad_shapes()
    print("Shape verification passed: output is one scalar per sample end to end.")


if __name__ == "__main__":
    main()
