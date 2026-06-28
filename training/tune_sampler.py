"""Langevin sampler tuning and verification (Task 4.2).

Task 4.1 built the sampling *mechanism*; this script does Task 4.2's job: find
the ``step_size`` / ``noise_scale`` / ``n_steps`` that make chaotic random paths
*reliably* evolve into valid routes on the trained energy, and prove it.

What "reliably evolve into a valid route" means here is concrete and measured,
not eyeballed:

- **Energy descends.** Each seed's energy falls from its chaotic start toward
  the low-energy basin. We report the mean initial/final energy and how the
  final energy compares to the reference valid paths (the training low-energy
  examples) and bad paths (the high-energy ones) for the same scene.
- **The path becomes a legal route.** The final trajectory is collision-free at
  every vertex and stays in the world, with endpoints pinned to start/goal — i.e.
  ``Trajectory.is_valid`` against the scene. The headline number is the fraction
  of seeds, across several scenes, that land valid.

The script sweeps a small grid of configs, ranks them by validity rate (ties
broken by how far below the reference-valid energy the samples land), prints a
table, and re-checks the winner on held-out scenes. Run it after training has
written ``exports/energy_model.pt``::

    python tune_sampler.py            # full sweep + verification
    python tune_sampler.py --quick    # smaller sweep, fewer scenes/seeds

Only numpy and torch are required.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:  # works when imported as part of the ``training`` package
    from training.energy_model import EnergyModel, trajectories_to_tensor, worlds_to_tensor
    from training.environment import GridWorld
    from training.generators import (
        generate_bad_trajectories,
        generate_valid_trajectories,
    )
    from training.sampler import LangevinConfig, LangevinSampler
    from training.train import load_checkpoint
    from training.trajectory import Trajectory
except ImportError:  # falls back when run directly as ``python tune_sampler.py``
    from energy_model import EnergyModel, trajectories_to_tensor, worlds_to_tensor
    from environment import GridWorld
    from generators import generate_bad_trajectories, generate_valid_trajectories
    from sampler import LangevinConfig, LangevinSampler
    from train import load_checkpoint
    from trajectory import Trajectory

__all__ = ["SceneEval", "evaluate_config", "sweep", "main"]


@dataclass
class SceneEval:
    """Per-config aggregate over a set of scenes."""

    config: LangevinConfig
    validity_rate: float  # fraction of seeds whose final path is_valid
    mean_init_energy: float
    mean_final_energy: float
    ref_valid_energy: float  # mean energy of generator valid paths (the target)
    ref_bad_energy: float  # mean energy of generator bad paths
    energy_below_ref: float  # ref_valid_energy - mean_final_energy (>0 = at/under data)
    descended: float  # fraction of seeds whose energy fell

    def score(self) -> tuple[float, float]:
        """Sort key: validity first, then how far under the reference we land."""
        return (self.validity_rate, self.energy_below_ref)


def _reference_energies(
    model: EnergyModel,
    worlds: list[GridWorld],
    rng: np.random.Generator,
    *,
    n_points: int,
    resolution: int,
    device: torch.device,
    paths_per_scene: int = 4,
) -> tuple[float, float]:
    """Mean energy the trained model assigns to generator valid vs bad paths.

    These are the yardsticks the sampled paths are measured against: a good
    sampler should land at or below the valid reference and well under the bad.
    """
    valids: list[Trajectory] = []
    bads: list[Trajectory] = []
    valid_world_ids: list[int] = []
    bad_world_ids: list[int] = []
    for i, world in enumerate(worlds):
        try:
            vs = generate_valid_trajectories(world, paths_per_scene, rng, n_points=n_points)
        except RuntimeError:
            continue
        bs = generate_bad_trajectories(world, paths_per_scene, rng, n_points=n_points)
        valids.extend(vs)
        bads.extend(bs)
        valid_world_ids.extend([i] * len(vs))
        bad_world_ids.extend([i] * len(bs))

    maps = worlds_to_tensor(worlds, resolution=resolution, device=device)
    with torch.no_grad():
        v_maps = maps[torch.tensor(valid_world_ids, device=device)]
        b_maps = maps[torch.tensor(bad_world_ids, device=device)]
        e_valid = model(v_maps, trajectories_to_tensor(valids, device=device))
        e_bad = model(b_maps, trajectories_to_tensor(bads, device=device))
    return float(e_valid.mean()), float(e_bad.mean())


def evaluate_config(
    model: EnergyModel,
    worlds: list[GridWorld],
    config: LangevinConfig,
    *,
    n_seeds: int,
    n_points: int,
    resolution: int,
    device: torch.device,
    rng: np.random.Generator,
    collision_margin: float = 0.0,
) -> SceneEval:
    """Run ``n_seeds`` descents per scene under ``config`` and aggregate quality."""
    valid_count = 0
    total = 0
    init_e: list[float] = []
    final_e: list[float] = []
    descended = 0

    for world in worlds:
        sampler = LangevinSampler(model, world, resolution=resolution, device=device)
        result = sampler.sample(
            n_samples=n_seeds, n_points=n_points, config=config, rng=rng
        )
        init_e.extend(result.initial_energy.tolist())
        final_e.extend(result.final_energy.tolist())
        descended += int((result.final_energy < result.initial_energy).sum())
        for traj in result.trajectories:
            total += 1
            if traj.is_valid(world, collision_margin=collision_margin):
                valid_count += 1

    ref_valid, ref_bad = _reference_energies(
        model, worlds, rng, n_points=n_points, resolution=resolution, device=device
    )
    mean_final = float(np.mean(final_e))
    return SceneEval(
        config=config,
        validity_rate=valid_count / max(total, 1),
        mean_init_energy=float(np.mean(init_e)),
        mean_final_energy=mean_final,
        ref_valid_energy=ref_valid,
        ref_bad_energy=ref_bad,
        energy_below_ref=ref_valid - mean_final,
        descended=descended / max(total, 1),
    )


def sweep(
    model: EnergyModel,
    worlds: list[GridWorld],
    configs: list[LangevinConfig],
    *,
    n_seeds: int,
    n_points: int,
    resolution: int,
    device: torch.device,
    seed: int = 0,
) -> list[SceneEval]:
    """Evaluate every candidate config and return results sorted best-first."""
    results: list[SceneEval] = []
    for i, config in enumerate(configs):
        rng = np.random.default_rng(seed + i)  # same seeds reused per config below
        evaluation = evaluate_config(
            model, worlds, config,
            n_seeds=n_seeds, n_points=n_points, resolution=resolution,
            device=device, rng=rng,
        )
        results.append(evaluation)
        c = config
        ss = f"{c.step_size:.3g}" + (f"->{c.final_step_size:.3g}" if c.final_step_size is not None else "")
        ns = f"{c.noise_scale:.3g}" + (f"->{c.final_noise_scale:.3g}" if c.final_noise_scale is not None else "")
        print(
            f"  step={ss:<13} noise={ns:<13} steps={c.n_steps:<4d} -> "
            f"valid={evaluation.validity_rate:5.1%}  "
            f"E {evaluation.mean_init_energy:+.2f}->{evaluation.mean_final_energy:+.2f}  "
            f"(ref_valid={evaluation.ref_valid_energy:+.2f} "
            f"ref_bad={evaluation.ref_bad_energy:+.2f})  "
            f"desc={evaluation.descended:4.0%}"
        )
    results.sort(key=lambda e: e.score(), reverse=True)
    return results


def _solvable_worlds(
    n: int, rng: np.random.Generator, *, n_points: int, n_obstacles=(3, 6)
) -> list[GridWorld]:
    """Draw ``n`` scenes that the valid generator can actually solve."""
    worlds: list[GridWorld] = []
    tries = 0
    while len(worlds) < n and tries < n * 10:
        tries += 1
        world = GridWorld.random(rng, n_obstacles=n_obstacles)
        try:
            generate_valid_trajectories(world, 1, rng, n_points=n_points)
        except RuntimeError:
            continue
        worlds.append(world)
    if len(worlds) < n:
        raise RuntimeError(f"Only assembled {len(worlds)}/{n} solvable scenes.")
    return worlds


def _candidate_configs(quick: bool) -> list[LangevinConfig]:
    """The sampler configurations to compare.

    Anchored on the shipped :class:`LangevinConfig` default (a moderate step with
    light noise that anneals to zero so paths settle onto the valley floor), with
    a handful of neighbours: a constant-noise variant (no settling), a shorter
    and a longer run, and a higher/lower step. The default should win or tie;
    this is the evidence behind the Task 4.2 tuning, not a blind grid.
    """
    base = dict(grad_clip=1.0, record_history=False)
    configs = [
        # The shipped default: 300 steps, step 0.02->0.005, noise 0.006->0.
        LangevinConfig(**base),
        # No settling (constant step/noise) — shows the anneal earns its keep.
        LangevinConfig(n_steps=500, step_size=0.02, noise_scale=0.006,
                       final_step_size=None, final_noise_scale=None, **base),
        # Longer run.
        LangevinConfig(n_steps=500, **base),
        # Larger initial step.
        LangevinConfig(n_steps=500, step_size=0.03, final_step_size=0.005, **base),
        # Lower noise floor of exploration.
        LangevinConfig(n_steps=500, noise_scale=0.004, **base),
    ]
    if not quick:
        # Longer run + a gentler step for the full sweep.
        configs.append(LangevinConfig(n_steps=800, **base))
        configs.append(LangevinConfig(n_steps=500, step_size=0.015, **base))
    return configs


def main() -> None:
    import sys

    quick = "--quick" in sys.argv
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint = repo_root / "exports" / "energy_model.pt"
    if not checkpoint.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint}. Run `python train.py` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint, map_location=device)
    n_points = (payload.get("train_config") or {}).get("n_points", 48)
    resolution = (payload.get("train_config") or {}).get("resolution", 64)

    n_scenes = 3 if quick else 6
    n_seeds = 6 if quick else 8
    rng = np.random.default_rng(0)
    worlds = _solvable_worlds(n_scenes, rng, n_points=n_points)

    print("Langevin sampler tuning (Task 4.2)")
    print(
        f"  device={device}  scenes={n_scenes}  seeds/scene={n_seeds}  "
        f"n_points={n_points}  resolution={resolution}"
    )
    configs = _candidate_configs(quick)
    print(f"  sweeping {len(configs)} configs:")
    results = sweep(
        model, worlds, configs,
        n_seeds=n_seeds, n_points=n_points, resolution=resolution, device=device,
    )

    best = results[0]
    c = best.config
    print("\nBest config")
    print(f"  step_size   : {c.step_size}")
    print(f"  noise_scale : {c.noise_scale}")
    print(f"  n_steps     : {c.n_steps}")
    print(f"  grad_clip   : {c.grad_clip}")
    print(f"  validity    : {best.validity_rate:.1%}")
    print(f"  energy      : {best.mean_init_energy:+.3f} -> {best.mean_final_energy:+.3f}")
    print(f"  ref valid   : {best.ref_valid_energy:+.3f}  (sample is "
          f"{best.energy_below_ref:+.3f} below it)")
    print(f"  ref bad     : {best.ref_bad_energy:+.3f}")

    # Held-out verification: fresh scenes the sweep never saw.
    verify_rng = np.random.default_rng(999)
    verify_worlds = _solvable_worlds(n_scenes, verify_rng, n_points=n_points)
    held = evaluate_config(
        model, verify_worlds, best.config,
        n_seeds=n_seeds, n_points=n_points, resolution=resolution,
        device=device, rng=verify_rng,
    )
    print("\nHeld-out verification (unseen scenes)")
    print(f"  validity    : {held.validity_rate:.1%}")
    print(f"  energy      : {held.mean_init_energy:+.3f} -> {held.mean_final_energy:+.3f}")
    print(f"  vs ref valid: {held.energy_below_ref:+.3f} below")

    ok = held.validity_rate >= 0.6 and held.mean_final_energy < held.ref_bad_energy
    if ok:
        print("\n  PASS: chaotic paths reliably descend into valid routes.")
    else:
        print("\n  WEAK: tune the grid further or retrain; validity below target.")


if __name__ == "__main__":
    main()
