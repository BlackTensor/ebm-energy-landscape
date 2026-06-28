"""Export layer for the Energy Landscape Visualizer (Tasks 5.1 and 5.2).

Phases 1-4 built the model and proved it generates multimodal routes; this is
where that work crosses the boundary into the web app. The visualiser is a
static site (GitHub Pages, per the project rules), so it cannot run PyTorch — it
reads plain JSON. This module renders everything the front end needs into
``/exports`` as JSON, with no Python or model dependency at view time.

The export writes four artifacts:

1. **The scene layout** (``scene.json``) — the start, goal, and circular
   obstacles, exactly as :class:`GridWorld` already serialises them. This is the
   board the heatmap and trajectories are drawn on.
2. **The energy heatmap** (``energy_field.json``) — a 2D grid of the learned
   energy over the world, so the front end can paint valleys in ``--valley`` and
   ridges in ``--ridge``. How a *trajectory* energy becomes a *2D field* is the
   one real design choice here; it is documented under "The energy field" below.
3. **A set of sampled trajectories** (``trajectories.json``, Task 5.1) — the
   valid routes the Langevin sampler reached, the payload of the multi-sample
   view (final states only).
4. **The descent history** (``descent.json``, Task 5.2) — the *path at each step*
   for a curated handful of those descents, so the web app can animate the slide
   from chaos into the valleys rather than only showing the settled result. It is
   documented under "The descent history" below.

A single sampler run feeds both (3) and (4): the trajectories are the run's valid
final states, and the descent history is the per-step record of a distinct subset
of the same run, so the animation ends exactly on the routes the multi-sample
view shows.

The energy field
----------------

The model scores a *whole trajectory*, ``E(scene, path) -> scalar``; it has no
native notion of "the energy at point (x, y)". To draw a heatmap we need a
scalar field over the world, so we define one directly from the model in a way
that means what the picture implies — *how willing is the model to route through
this location?*

For a location ``g`` on the grid, take valid reference routes for the scene and
bend each one smoothly so it passes through ``g``, then read off the energy. The
bend translates the reference's nearest point onto ``g`` and tapers that shift to
zero along the path with a Gaussian window (endpoints stay pinned to start and
goal), so the route stays a smooth, well-formed path that merely detours through
``g``. Over reference routes ``P``::

    field(g) = min over P of  E(scene, bend(P, through=g))

Reading it: a ``g`` on a valid corridor needs only a small bend, so the route
stays valid and the energy stays low — a **valley**. A ``g`` inside an obstacle
forces the bent route through a collision, and a ``g`` far from every corridor
forces a large detour; both read as high energy — a **ridge**. Bending a whole
*neighbourhood* of the path (not a single waypoint) is what gives the field a
strong, legible signal: the energy depends on a real change in the route's shape,
not on one point the sequence encoder would average away.

Taking the minimum over *several* reference routes means valleys form along
*every* distinct route, so the heatmap shows the landscape of all valid futures
rather than a single answer — the project's core thesis, made visible. The field
is genuinely the trained model's opinion (it is just ``E`` evaluated on smooth
probe routes), deterministic given the scene, and costs only forward passes. We
export the raw energies plus a ``[0, 1]`` normalisation (0 = deepest valley,
1 = ridge) with the min/max, so the renderer can map colour without rescanning.

The descent history
-------------------

The Langevin sampler already records, optionally, the full path at every step
(``record_history``); Task 5.2 serialises it so the web app can play the descent.
Two practical shaping choices keep the artifact honest *and* web-friendly:

- **Which descents.** Animating all sixteen seeds would be noise; we export a
  curated handful — the valid routes, picked most-distinct-first (greedy on mean
  pointwise distance, the same notion as the Task 4.3 multimodality check) so the
  animation lands on visibly different valleys. The final frame of each exported
  descent is exactly one of the routes in ``trajectories.json``.
- **Which frames.** A 300-step descent is far more temporal resolution than an
  animation needs, and writing every step for several paths bloats the file. We
  keep a uniform subsample of the steps — always including step 0 (chaos) and the
  last step (settled) — so the *whole* evolution is represented at a size the
  static site loads instantly. ``frame_steps`` records which original step each
  frame came from. Each frame also carries the path's energy at that step, so the
  front end can drive an energy readout falling alongside the motion.

Every frame keeps the endpoints pinned to start and goal (the sampler holds them
there), so the animation shows the interior of the path resolving from chaos into
a route while the endpoints stay put.

Running it
----------

``python export.py`` loads ``exports/energy_model.pt``, picks a deterministic
scene, and writes the four JSON files into ``/exports``. Only numpy and torch
are required, so it runs unchanged on the Colab free tier and locally. Schema
validation of the written files is Task 5.3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

try:  # works when imported as part of the ``training`` package
    from training.energy_model import EnergyModel, rasterize_world
    from training.environment import GridWorld
    from training.generators import generate_valid_trajectories
    from training.sampler import LangevinConfig, LangevinSampler, SampleResult
    from training.trajectory import Trajectory
except ImportError:  # falls back when run directly as ``python export.py``
    from energy_model import EnergyModel, rasterize_world
    from environment import GridWorld
    from generators import generate_valid_trajectories
    from sampler import LangevinConfig, LangevinSampler, SampleResult
    from trajectory import Trajectory

__all__ = [
    "EXPORT_FORMAT_VERSION",
    "EnergyField",
    "cached_map_energy",
    "compute_energy_field",
    "scene_payload",
    "energy_field_payload",
    "trajectories_payload",
    "descent_payload",
    "export_scene",
]

# Bumped if the JSON layout written here changes incompatibly. The web app and
# the Task 5.3 validator both read this to refuse mismatched bundles.
EXPORT_FORMAT_VERSION = 1


@dataclass
class EnergyField:
    """A 2D energy grid over the world, ready to serialise.

    Attributes
    ----------
    energy:
        ``(resolution, resolution)`` raw model energies. Row index increases with
        ``y`` and column index with ``x`` (the convention of
        :func:`energy_model.rasterize_world`), each spanning ``[0, size]``.
    resolution:
        Side length of the square grid.
    size:
        The world extent the grid covers (``[0, size] x [0, size]``).
    vmin, vmax:
        The minimum and maximum raw energy on the grid — the endpoints of the
        ``[0, 1]`` normalisation, handed to the renderer so it need not rescan.
    """

    energy: np.ndarray
    resolution: int
    size: float
    vmin: float
    vmax: float

    @property
    def normalized(self) -> np.ndarray:
        """Energies rescaled to ``[0, 1]`` (0 = deepest valley, 1 = ridge)."""
        span = self.vmax - self.vmin
        if span <= 0.0:  # a flat field; avoid divide-by-zero
            return np.zeros_like(self.energy)
        return (self.energy - self.vmin) / span


# --- energy field construction ----------------------------------------------


def cached_map_energy(
    model: EnergyModel, map_emb: Tensor, trajectories: Tensor
) -> Tensor:
    """Energy of a batch of paths against a *pre-encoded* scene.

    :meth:`EnergyModel.forward` re-runs the map CNN on every call. The energy
    field evaluates thousands of paths against one fixed scene, so we encode the
    map once (:meth:`EnergyModel.map_encoder`) and reuse the embedding here,
    running only the trajectory encoder and head per batch. This is the same
    scalar the model would return — just without recomputing the identical map
    convolution each time.

    Parameters
    ----------
    map_emb:
        The map embedding, shape ``(1, map_dim)``, from ``model.map_encoder``.
    trajectories:
        A batch of paths, shape ``(B, N, 2)``.
    """
    traj_emb = model.trajectory_encoder(trajectories)
    joint = torch.cat([map_emb.expand(traj_emb.shape[0], -1), traj_emb], dim=-1)
    return model.head(joint)


def _bend_paths(
    refs: Tensor, grid: Tensor, window: float, size: float
) -> Tensor:
    """Bend each reference route smoothly through each grid point.

    For grid point ``g`` and reference ``P``, the anchor is ``P``'s point nearest
    ``g``; the whole path is shifted by ``g - P[anchor]`` weighted by a Gaussian
    in index distance from the anchor, so the anchor lands exactly on ``g`` and
    the shift tapers to zero away from it. The endpoint weights are forced to zero
    so start and goal stay pinned, and points are clamped to the world.

    Returns a ``(G, R, N, 2)`` tensor of bent routes.
    """
    g, r, n = grid.shape[0], refs.shape[0], refs.shape[1]
    # Nearest reference point to each grid point: anchor index a, shape (G, R).
    diff = grid[:, None, None, :] - refs[None, :, :, :]  # (G, R, N, 2)
    dist2 = diff.pow(2).sum(-1)  # (G, R, N)
    anchor = dist2.argmin(dim=2)  # (G, R)

    idx = torch.arange(n, device=refs.device).float()  # (N,)
    offset = idx[None, None, :] - anchor[:, :, None].float()  # (G, R, N)
    weight = torch.exp(-offset.pow(2) / (2.0 * window * window))  # (G, R, N)
    weight[:, :, 0] = 0.0  # pin start
    weight[:, :, -1] = 0.0  # pin goal

    anchor_pt = torch.gather(
        refs[None].expand(g, r, n, 2), 2,
        anchor[:, :, None, None].expand(g, r, 1, 2),
    ).squeeze(2)  # (G, R, 2)
    delta = grid[:, None, :] - anchor_pt  # (G, R, 2)

    bent = refs[None] + weight[:, :, :, None] * delta[:, :, None, :]  # (G, R, N, 2)
    return bent.clamp(0.0, size)


def _gaussian_blur2d(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Light separable Gaussian blur in pure numpy (no scipy dependency).

    Used only to smooth the discretization seams the nearest-point anchor leaves
    in the energy field; the kernel is small so structure (valleys, ridges) is
    preserved. Edges are handled by clamping (``np.pad`` edge mode).
    """
    if sigma <= 0.0:
        return grid
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    out = grid.astype(np.float64)
    padded = np.pad(out, ((0, 0), (radius, radius)), mode="edge")
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 1, padded)
    padded = np.pad(out, ((radius, radius), (0, 0)), mode="edge")
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 0, padded)
    return out.astype(np.float32)


def compute_energy_field(
    model: EnergyModel,
    world: GridWorld,
    references: list[np.ndarray],
    *,
    resolution: int = 64,
    raster_resolution: int = 64,
    window: Optional[float] = None,
    smooth_sigma: float = 0.8,
    chunk: int = 64,
    device: Optional[torch.device] = None,
) -> EnergyField:
    """Build the 2D energy heatmap by bending reference routes through the grid.

    Implements the field defined in the module docstring: for each grid location
    ``g`` and over every reference route ``P``, smoothly bend ``P`` so it passes
    through ``g`` (:func:`_bend_paths`) and keep the lowest resulting energy. The
    map is encoded once and reused (:func:`cached_map_energy`); grid points are
    processed in chunks so the batch stays bounded in memory.

    Parameters
    ----------
    references:
        Valid reference routes as ``(N, 2)`` arrays. Valleys form along all of
        them; supplying several distinct routes makes the multimodal landscape
        visible.
    resolution:
        Side length of the exported heatmap grid.
    raster_resolution:
        Resolution the scene is rasterised at for the map encoder; must match the
        model's training resolution (the project default is 64).
    window:
        Standard deviation (in path-index units) of the Gaussian bend window —
        how much of the route flexes toward each probe point. Defaults to
        ``N / 8``, a localised but smooth bump.
    smooth_sigma:
        Standard deviation (in grid cells) of a light Gaussian blur applied to
        the finished field, smoothing the discretization seams the nearest-point
        anchor leaves behind. ``0`` disables it.
    chunk:
        Number of grid points evaluated per batch (bounds peak memory).
    """
    if not references:
        raise ValueError("compute_energy_field needs at least one reference path.")
    device = device or next(model.parameters()).device
    model.eval()

    size = float(world.size)
    n_points = references[0].shape[0]
    if any(r.shape != (n_points, 2) for r in references):
        raise ValueError("All reference paths must share shape (N, 2).")
    window = window if window is not None else max(2.0, n_points / 8.0)

    # Encode the fixed scene once.
    raster = rasterize_world(world, resolution=raster_resolution)
    map_t = torch.from_numpy(raster).to(device=device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        map_emb = model.map_encoder(map_t)  # (1, map_dim)

    refs = torch.from_numpy(np.stack(references, axis=0).astype(np.float32)).to(device)
    n_ref = refs.shape[0]

    # Grid of probe locations, row -> y, col -> x, matching rasterize_world.
    xs = np.linspace(0.0, size, resolution)
    ys = np.linspace(0.0, size, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    grid_t = torch.from_numpy(grid).to(device)  # (G, 2)

    field = np.empty(grid.shape[0], dtype=np.float32)

    with torch.no_grad():
        for start_i in range(0, grid_t.shape[0], chunk):
            pts = grid_t[start_i : start_i + chunk]  # (g, 2)
            g = pts.shape[0]
            bent = _bend_paths(refs, pts, window, size)  # (g, R, N, 2)
            flat = bent.reshape(g * n_ref, n_points, 2)
            energy = cached_map_energy(model, map_emb, flat).reshape(g, n_ref)
            field[start_i : start_i + g] = energy.min(dim=1).values.cpu().numpy()

    field2d = _gaussian_blur2d(field.reshape(resolution, resolution), smooth_sigma)
    return EnergyField(
        energy=field2d,
        resolution=resolution,
        size=size,
        vmin=float(field2d.min()),
        vmax=float(field2d.max()),
    )


# --- trajectory sampling -----------------------------------------------------


def sample_valid_routes(
    model: EnergyModel,
    world: GridWorld,
    *,
    n_seeds: int = 12,
    n_points: int = 48,
    config: Optional[LangevinConfig] = None,
    record_history: bool = False,
    seed: int = 0,
) -> tuple[list[Trajectory], np.ndarray, SampleResult]:
    """Run Langevin descents and return the valid routes, energies, and result.

    Uses the shipped (Task 4.2-tuned) :class:`LangevinConfig` defaults, descends
    ``n_seeds`` independent random paths, and keeps the ones that landed on valid
    routes (collision-free, endpoints pinned). Set ``record_history=True`` to keep
    the full per-step descent on the returned :class:`SampleResult` (Task 5.2 uses
    it; the fast scene scan does not).

    Returns ``(routes, energies, result)`` where ``routes``/``energies`` are the
    valid descents sorted by ascending final energy (deepest-valley route first)
    and ``result`` is the raw sampler output for every seed.
    """
    cfg = config or LangevinConfig(seed=seed)
    cfg.record_history = record_history
    sampler = LangevinSampler(model, world)
    rng = np.random.default_rng(seed)
    result = sampler.sample(n_samples=n_seeds, n_points=n_points, config=cfg, rng=rng)

    valid: list[tuple[float, Trajectory]] = []
    for energy, traj in zip(result.final_energy, result.trajectories):
        if traj.is_valid(world):
            valid.append((float(energy), traj))

    valid.sort(key=lambda pair: pair[0])
    routes = [traj for _, traj in valid]
    energies = np.array([e for e, _ in valid], dtype=np.float32)
    return routes, energies, result


# --- JSON payloads -----------------------------------------------------------


def scene_payload(world: GridWorld) -> dict:
    """The scene layout: world size, start, goal, and circular obstacles."""
    return {
        "format_version": EXPORT_FORMAT_VERSION,
        "scene": world.to_dict(),
    }


def energy_field_payload(field: EnergyField) -> dict:
    """The energy heatmap: raw + normalised 2D grids and their value range."""
    return {
        "format_version": EXPORT_FORMAT_VERSION,
        "resolution": int(field.resolution),
        "size": float(field.size),
        # Row index increases with y, column with x; both span [0, size].
        "orientation": "row->y, col->x, both in [0, size]",
        "definition": (
            "field(x, y) = min over reference routes P of E(scene, P bent "
            "smoothly to pass through (x, y)). Low = valley (valid corridor), "
            "high = ridge (obstacle or forced detour)."
        ),
        "vmin": float(field.vmin),
        "vmax": float(field.vmax),
        "energy": np.round(field.energy, 5).tolist(),
        "energy_normalized": np.round(field.normalized, 5).tolist(),
    }


def trajectories_payload(
    routes: list[Trajectory], energies: np.ndarray
) -> dict:
    """A set of sampled valid routes as lists of ``[x, y]`` pairs, with energies."""
    return {
        "format_version": EXPORT_FORMAT_VERSION,
        "count": len(routes),
        "n_points": int(routes[0].num_points) if routes else 0,
        "trajectories": [
            {
                "points": np.round(traj.points, 5).tolist(),
                "energy": float(energy),
            }
            for traj, energy in zip(routes, energies)
        ],
    }


def _mean_pointwise_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Euclidean distance between corresponding points of two paths."""
    return float(np.linalg.norm(a - b, axis=1).mean())


def select_descents(
    result: SampleResult,
    world: GridWorld,
    *,
    max_descents: int = 6,
    distinct_threshold: float = 0.05,
) -> list[int]:
    """Choose which descents to animate: distinct valid routes, best energy first.

    Returns indices into ``result`` (its ``trajectories``/``history`` columns).
    Valid descents are considered in ascending final-energy order and accepted
    when they differ from every already-chosen route by more than
    ``distinct_threshold`` (mean pointwise distance) — the same distinctness
    notion as the Task 4.3 multimodality check — so the animation lands on visibly
    different valleys. If fewer than ``max_descents`` distinct valid routes exist,
    the remaining valid routes top the list up; if no descent reached a valid
    route at all, the lowest-energy descents are used so the animation is never
    empty.
    """
    order = sorted(range(len(result.trajectories)), key=lambda i: result.final_energy[i])
    valid = [i for i in order if result.trajectories[i].is_valid(world)]

    chosen: list[int] = []
    reps: list[np.ndarray] = []
    for i in valid:
        path = result.trajectories[i].points
        if all(_mean_pointwise_distance(path, r) > distinct_threshold for r in reps):
            chosen.append(i)
            reps.append(path)
        if len(chosen) >= max_descents:
            break

    if len(chosen) < max_descents:  # top up with remaining valid routes
        for i in valid:
            if i not in chosen:
                chosen.append(i)
                if len(chosen) >= max_descents:
                    break

    if not chosen:  # no valid descent — fall back to the lowest-energy ones
        chosen = order[:max_descents]
    return chosen


def _frame_steps(n_steps: int, max_frames: int) -> list[int]:
    """A uniform subsample of step indices ``0..n_steps``, always keeping both ends.

    The descent has ``n_steps + 1`` recorded states (row 0 is the chaotic start).
    We keep at most ``max_frames`` of them, evenly spaced, with step 0 and the
    final step always present so the animation spans the whole evolution.
    """
    total = n_steps + 1
    if max_frames >= total:
        return list(range(total))
    stride = int(np.ceil(total / max_frames))
    steps = list(range(0, total, stride))
    if steps[-1] != n_steps:
        steps.append(n_steps)
    return steps


def descent_payload(
    result: SampleResult,
    world: GridWorld,
    indices: list[int],
    *,
    max_frames: int = 60,
    coord_decimals: int = 4,
) -> dict:
    """The descent history: the path (and its energy) at each kept step.

    For each chosen descent (``indices`` into ``result``) we emit the path at a
    uniform subsample of the Langevin steps, plus the energy at each of those
    steps and whether the descent ended valid. ``frame_steps`` records which
    original step each frame came from. Requires ``result.history`` (the sampler
    must have run with ``record_history=True``).
    """
    if result.history is None:
        raise ValueError(
            "descent_payload needs result.history; run the sampler with "
            "record_history=True."
        )
    n_steps = result.energy_trace.shape[0] - 1
    n_points = result.history.shape[2]
    steps = _frame_steps(n_steps, max_frames)

    descents = []
    for i in indices:
        frames = [np.round(result.history[s, i], coord_decimals).tolist() for s in steps]
        energy = [round(float(result.energy_trace[s, i]), 4) for s in steps]
        descents.append(
            {
                "valid": bool(result.trajectories[i].is_valid(world)),
                "final_energy": round(float(result.final_energy[i]), 4),
                "frames": frames,
                "energy": energy,
            }
        )

    return {
        "format_version": EXPORT_FORMAT_VERSION,
        "n_descents": len(descents),
        "n_points": int(n_points),
        "n_steps": int(n_steps),
        "frame_steps": [int(s) for s in steps],
        "n_frames": len(steps),
        "descents": descents,
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def export_scene(
    model: EnergyModel,
    world: GridWorld,
    out_dir: Path,
    *,
    n_seeds: int = 12,
    n_points: int = 48,
    field_resolution: int = 64,
    raster_resolution: int = 64,
    max_descents: int = 6,
    max_frames: int = 60,
    seed: int = 0,
) -> dict[str, Path]:
    """Write the scene, energy heatmap, routes, and descent history to ``out_dir``.

    A single Langevin run (with history recorded) drives the generative outputs:
    its valid final states are the exported trajectory set (5.1), and the per-step
    history of a distinct subset of the same run is the descent animation (5.2),
    so the animation ends exactly on the routes the multi-sample view shows. The
    energy field is probed with a set of reference routes that spans the scene's
    distinct corridors: a backbone of generated valid routes, plus every valid
    sampled route, so the heatmap's valleys trace the routes drawn on top of it as
    well as the corridors the generator knows.

    Returns a dict mapping ``"scene"``, ``"energy_field"``, ``"trajectories"``,
    and ``"descent"`` to the paths written.
    """
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)

    # One descent run feeds both the final routes (5.1) and the history (5.2).
    routes, energies, result = sample_valid_routes(
        model, world, n_seeds=n_seeds, n_points=n_points,
        record_history=True, seed=seed,
    )

    # Reference routes for the field: a backbone of distinct generated valid
    # routes (always available) plus the sampler's own valid routes, so valleys
    # form along every corridor the model and generator agree is valid.
    references = [
        t.points.astype(np.float32)
        for t in generate_valid_trajectories(world, 6, rng, n_points=n_points)
    ]
    references += [r.points.astype(np.float32) for r in routes]

    field = compute_energy_field(
        model,
        world,
        references,
        resolution=field_resolution,
        raster_resolution=raster_resolution,
    )

    descent_indices = select_descents(result, world, max_descents=max_descents)

    written = {
        "scene": _write_json(out_dir / "scene.json", scene_payload(world)),
        "energy_field": _write_json(
            out_dir / "energy_field.json", energy_field_payload(field)
        ),
        "trajectories": _write_json(
            out_dir / "trajectories.json", trajectories_payload(routes, energies)
        ),
        "descent": _write_json(
            out_dir / "descent.json",
            descent_payload(result, world, descent_indices, max_frames=max_frames),
        ),
    }
    return written


def main() -> None:
    """Load the trained model, pick a scene, and write the Phase 5 exports."""
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint = repo_root / "exports" / "energy_model.pt"
    out_dir = repo_root / "exports"
    if not checkpoint.exists():
        raise SystemExit(f"No checkpoint at {checkpoint}. Run `python train.py` first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from train import load_checkpoint  # local import keeps module import light

    model, payload = load_checkpoint(checkpoint, map_location=device)
    n_points = (payload.get("train_config") or {}).get("n_points", 48)

    print("Energy landscape export (Tasks 5.1 + 5.2)")
    print(f"  checkpoint : {checkpoint.name}  device={device}")

    # Pick a deterministic scene the sampler actually solves well: scan a few
    # seeds and take the first whose Langevin descent yields several distinct
    # valid routes, so the exported trajectory set (and the multi-sample view it
    # feeds) shows real multimodality rather than a single or empty result.
    best: Optional[tuple] = None  # (n_valid, seed, world)
    for scene_seed in range(20):
        world = GridWorld.random(
            np.random.default_rng(scene_seed), n_obstacles=(3, 4)
        )
        routes, _, _ = sample_valid_routes(
            model, world, n_seeds=16, n_points=n_points, seed=0
        )
        print(f"  scan seed {scene_seed:>2}: obstacles={len(world.obstacles)} "
              f"valid_routes={len(routes)}")
        if best is None or len(routes) > best[0]:
            best = (len(routes), scene_seed, world)
        if len(routes) >= 5:
            break

    n_valid, scene_seed, world = best
    print(f"  chosen scene: seed={scene_seed}  obstacles={len(world.obstacles)}  "
          f"valid_routes={n_valid}")

    written = export_scene(
        model, world, out_dir, n_seeds=16, n_points=n_points, seed=0
    )

    # Report what landed, and confirm each file is valid JSON on disk.
    for name, path in written.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        size_kb = path.stat().st_size / 1024
        if name == "energy_field":
            detail = f"{data['resolution']}x{data['resolution']} grid"
        elif name == "trajectories":
            detail = f"{data['count']} valid routes"
        elif name == "descent":
            detail = f"{data['n_descents']} descents x {data['n_frames']} frames"
        else:
            detail = f"{len(data['scene']['obstacles'])} obstacles"
        print(f"  wrote {name:<13} -> {path.name:<20} ({detail}, {size_kb:.1f} KB)")

    # Self-check: the four artifacts exist, parse, and carry the right shapes.
    field_data = json.loads((out_dir / "energy_field.json").read_text("utf-8"))
    res = field_data["resolution"]
    assert len(field_data["energy"]) == res, "energy grid is not square"
    assert all(len(row) == res for row in field_data["energy"]), "ragged energy grid"
    assert field_data["vmax"] >= field_data["vmin"], "field range inverted"
    traj_data = json.loads((out_dir / "trajectories.json").read_text("utf-8"))
    assert traj_data["count"] == len(traj_data["trajectories"]), "route count mismatch"
    assert traj_data["count"] >= 1, "no sampled valid routes to export"
    scene_data = json.loads((out_dir / "scene.json").read_text("utf-8"))
    assert "start" in scene_data["scene"] and "goal" in scene_data["scene"]
    spread = field_data["vmax"] - field_data["vmin"]
    assert spread > 0.1, f"energy field is nearly flat (spread {spread:.3f})"

    # Descent history (Task 5.2): every descent has the same frame count, each
    # frame is a full N-point path, and the endpoints stay pinned across frames.
    desc = json.loads((out_dir / "descent.json").read_text("utf-8"))
    assert desc["n_descents"] >= 1, "no descents to animate"
    assert len(desc["frame_steps"]) == desc["n_frames"], "frame_steps/n_frames mismatch"
    assert desc["frame_steps"][0] == 0 and desc["frame_steps"][-1] == desc["n_steps"], (
        "descent must span step 0 (chaos) to the final step (settled)"
    )
    start_xy = scene_data["scene"]["start"]
    goal_xy = scene_data["scene"]["goal"]
    for d in desc["descents"]:
        assert len(d["frames"]) == desc["n_frames"], "ragged descent frame count"
        assert len(d["energy"]) == desc["n_frames"], "energy/frame count mismatch"
        for frame in d["frames"]:
            assert len(frame) == desc["n_points"], "frame is not a full N-point path"
        # Endpoints pinned at the first and last frame.
        for frame in (d["frames"][0], d["frames"][-1]):
            assert np.allclose(frame[0], start_xy, atol=1e-3), "frame start not pinned"
            assert np.allclose(frame[-1], goal_xy, atol=1e-3), "frame goal not pinned"
    n_settled = sum(d["energy"][-1] < d["energy"][0] for d in desc["descents"])
    print(
        f"  descent      : {desc['n_descents']} paths, {desc['n_frames']} frames "
        f"(steps {desc['frame_steps'][0]}..{desc['frame_steps'][-1]}), "
        f"{n_settled}/{desc['n_descents']} lowered energy"
    )
    print(
        f"  energy range : {field_data['vmin']:+.3f} (valley) .. "
        f"{field_data['vmax']:+.3f} (ridge)  spread={spread:.3f}"
    )
    print("  self-check passed: scene, heatmap, routes, and descent written to /exports.")


if __name__ == "__main__":
    main()
