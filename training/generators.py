"""Trajectory generators for the Energy Landscape Visualizer.

This module produces the training paths the energy model learns from. Task 1.3
covers the *valid* generator: smooth, obstacle-avoiding routes from start to
goal that serve as the **low-energy** examples. Task 1.4 adds the *bad*
generator: random walks, zig-zags, and obstacle-crossing routes that serve as
the **high-energy** examples. Together they are the two poles of the
contrastive training signal.

How a valid path is made
------------------------

We treat path generation as relaxing an **active contour** (a "snake"):

1. Seed a smooth initial curve from start to goal — a straight line bent by a
   randomly signed, randomly scaled bump. The randomness is what makes
   successive calls trace *distinct* valid routes (different ways around the
   obstacles), which the EBM later needs to look multimodal.
2. Relax the curve for a number of iterations under two forces, with the two
   endpoints pinned:
   - an **internal** smoothing force (discrete Laplacian) that keeps the curve
     short and smooth;
   - an **external** repulsion force that pushes any point sitting inside (or
     within a clearance band of) an obstacle radially out of it.
   The balance settles into a smooth curve that bows around obstacles.
3. Resample the relaxed curve to a fixed point count by arc length, so every
   trajectory has the same ``N`` and roughly even spacing.
4. Validate against the ``GridWorld``. If a draw fails to clear the obstacles,
   retry with a fresh seed and a stronger bump.

The result is a :class:`~training.trajectory.Trajectory` in the exact format
from Task 1.2. Only numpy is required, so this runs unchanged on Colab's free
tier and locally.

How a bad path is made
----------------------

Bad paths come in three flavours, chosen to cover distinct failure modes so the
energy model cannot pass by learning a single shortcut:

- ``"random_walk"`` — a Brownian wander from the start. It ignores both the
  goal and the obstacles, so it almost never ends where it should: bad by
  *not reaching the goal*.
- ``"zigzag"`` — start to goal, but routed through a handful of knots with
  large alternating sideways swings and sharp piecewise-linear corners: bad by
  *jaggedness* (and it usually clips obstacles too).
- ``"crossing"`` — start to goal routed straight *through* one or two obstacle
  centres: bad by *colliding*.

A path is only accepted as a high-energy example once it is "clearly bad" —
either invalid against the world (collides or misses the goal) or sharply
non-smooth. That keeps the high-energy set from accidentally containing a
perfectly fine route.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # works when imported as part of the ``training`` package
    from training.environment import GridWorld
    from training.trajectory import Trajectory
except ImportError:  # falls back when run directly as ``python generators.py``
    from environment import GridWorld
    from trajectory import Trajectory

__all__ = [
    "generate_valid_trajectory",
    "generate_valid_trajectories",
    "generate_bad_trajectory",
    "generate_bad_trajectories",
    "BAD_MODES",
]

# The distinct flavours of "bad" the high-energy generator can produce.
BAD_MODES = ("random_walk", "zigzag", "crossing")


# --- low-level curve helpers ------------------------------------------------


def _dedupe_consecutive(points: np.ndarray) -> np.ndarray:
    """Drop points that coincide with their predecessor (zero-length steps).

    Keeps arc-length resampling well defined. The first point is always kept.
    """
    if points.shape[0] < 2:
        return points
    keep = np.ones(points.shape[0], dtype=bool)
    keep[1:] = np.any(np.diff(points, axis=0) != 0.0, axis=1)
    return points[keep]


def _resample_by_arclength(points: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a polyline to ``n_points`` evenly spaced by arc length.

    Endpoints are preserved exactly. Input should have no zero-length steps
    (run :func:`_dedupe_consecutive` first).
    """
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumulative[-1]
    if total == 0.0:  # degenerate; should not happen for a real path
        return np.repeat(points[:1], n_points, axis=0)
    targets = np.linspace(0.0, total, n_points)
    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    return np.stack([x, y], axis=1)


def _max_turn_angle(points: np.ndarray) -> float:
    """Sharpest direction change between consecutive segments, in degrees.

    A smoothness score: 0 for a straight line, approaching 180 for a hairpin.
    """
    seg = np.diff(points, axis=0)
    unit = seg / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-9)
    cos = np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).max())


def _seed_curve(
    world: GridWorld,
    rng: np.random.Generator,
    n_points: int,
    bump: float,
) -> np.ndarray:
    """A smooth start-to-goal curve: straight line plus a random sine bump.

    The bump is applied perpendicular to the start->goal direction with a
    random sign and amplitude, so different calls lean to different sides of
    the scene. Amplitude is expressed as a fraction of the start-goal distance.
    """
    start = np.asarray(world.start, dtype=float)
    goal = np.asarray(world.goal, dtype=float)
    t = np.linspace(0.0, 1.0, n_points)
    line = start[None, :] + t[:, None] * (goal - start)[None, :]

    direction = goal - start
    span = float(np.linalg.norm(direction))
    if span == 0.0:
        return line
    perp = np.array([-direction[1], direction[0]]) / span

    # One or two sine lobes, random sign and amplitude -> varied curvature.
    lobes = float(rng.choice([1.0, 1.5, 2.0]))
    amplitude = rng.uniform(0.5, 1.0) * bump * span * rng.choice([-1.0, 1.0])
    offset = amplitude * np.sin(np.pi * lobes * t)
    return line + offset[:, None] * perp[None, :]


def _relax(
    points: np.ndarray,
    world: GridWorld,
    *,
    iterations: int,
    smoothing: float,
    repulsion: float,
    clearance: float,
) -> np.ndarray:
    """Relax a curve under Laplacian smoothing + obstacle repulsion.

    Endpoints are held fixed; interior points are clamped to the world. Works
    on a copy and returns it.
    """
    pts = points.copy()
    start, goal = pts[0].copy(), pts[-1].copy()
    eps = 1e-9

    for _ in range(iterations):
        # Internal force: pull each interior point toward its neighbours' mean.
        laplacian = pts[2:] - 2.0 * pts[1:-1] + pts[:-2]
        pts[1:-1] += smoothing * laplacian

        # External force: push points out of every obstacle's clearance band.
        for obstacle in world.obstacles:
            delta = pts - obstacle.center[None, :]
            dist = np.linalg.norm(delta, axis=1)
            signed = dist - obstacle.radius
            mask = signed < clearance
            if not np.any(mask):
                continue
            unit = delta[mask] / (dist[mask, None] + eps)
            push = (clearance - signed[mask])[:, None] * unit
            pts[mask] += repulsion * push

        pts[0], pts[-1] = start, goal
        pts = world.clamp(pts)
        pts[0], pts[-1] = start, goal  # clamp may nudge; re-pin endpoints

    return pts


# --- public API -------------------------------------------------------------


def generate_valid_trajectory(
    world: GridWorld,
    rng: Optional[np.random.Generator] = None,
    *,
    n_points: int = 48,
    clearance: float = 0.03,
    max_turn_deg: float = 75.0,
    max_tries: int = 24,
) -> Trajectory:
    """Generate one smooth, obstacle-avoiding path from start to goal.

    The returned :class:`Trajectory` is guaranteed valid against ``world``
    (in bounds, collision-free at every vertex with at least ``clearance``
    distance from obstacles during relaxation, endpoints matching). Successive
    calls with an advancing RNG yield visibly different routes.

    Smoothness is enforced, not merely hoped for: a draw is accepted as soon as
    it is valid *and* its sharpest turn is at most ``max_turn_deg``. If a draw
    is valid but kinkier than that (an awkward detour around tightly packed
    obstacles), the generator keeps trying other random seeds and returns the
    smoothest valid path it found, so hard scenes still yield a path.

    Parameters
    ----------
    world:
        The scene to plan through.
    rng:
        Seeded generator for reproducibility. A fresh default is used if None.
    n_points:
        Number of points ``N`` in the returned trajectory.
    clearance:
        Safety band (world units) kept from obstacle boundaries while relaxing.
        Validation itself uses a smaller margin so paths comfortably pass.
    max_turn_deg:
        Smoothness bar: the largest acceptable direction change at any vertex.
    max_tries:
        How many seeded attempts before giving up. Each retry uses a stronger
        bump and more relaxation, so dense scenes still resolve.

    Raises
    ------
    RuntimeError
        If no valid path was found within ``max_tries`` (only on pathological
        scenes, e.g. a goal sealed off by overlapping obstacles).
    """
    rng = np.random.default_rng() if rng is None else rng
    # Relax at a denser resolution than the output, then resample down.
    work_points = max(n_points, 64)
    # Validation margin sits just inside the relaxation clearance so a path
    # that cleared the band reliably passes the check.
    validate_margin = 0.5 * clearance

    best: Optional[Trajectory] = None
    best_turn = np.inf

    for attempt in range(max_tries):
        # Escalate aggressiveness on later attempts.
        bump = 0.15 + 0.1 * attempt
        iterations = 150 + 60 * attempt

        curve = _seed_curve(world, rng, work_points, bump=bump)
        curve = _relax(
            curve,
            world,
            iterations=iterations,
            smoothing=0.2,
            repulsion=0.8,
            clearance=clearance,
        )
        curve = _dedupe_consecutive(curve)
        if curve.shape[0] < 2:
            continue
        resampled = _resample_by_arclength(curve, n_points)
        # Pin endpoints exactly so the endpoint check is not lost to rounding.
        resampled[0] = np.asarray(world.start, dtype=float)
        resampled[-1] = np.asarray(world.goal, dtype=float)

        candidate = Trajectory(resampled)
        if not candidate.is_valid(world, collision_margin=validate_margin):
            continue

        turn = _max_turn_angle(candidate.points)
        if turn <= max_turn_deg:
            return candidate  # valid and smooth enough; take it
        if turn < best_turn:  # valid but kinky; remember the best so far
            best, best_turn = candidate, turn

    if best is not None:
        return best  # hard scene: smoothest valid path we could find

    raise RuntimeError(
        f"Could not generate a valid trajectory in {max_tries} tries. The scene "
        f"may be unsolvable (goal enclosed by obstacles). Try lowering clearance "
        f"or regenerating the world with fewer/smaller obstacles."
    )


def generate_valid_trajectories(
    world: GridWorld,
    n: int,
    rng: Optional[np.random.Generator] = None,
    *,
    n_points: int = 48,
    clearance: float = 0.03,
    max_turn_deg: float = 75.0,
    max_tries: int = 24,
) -> list[Trajectory]:
    """Generate ``n`` valid trajectories for one scene (the low-energy set).

    Shares a single RNG across draws so the batch is both reproducible and
    varied. Individual failures bubble up as ``RuntimeError`` from
    :func:`generate_valid_trajectory`.
    """
    rng = np.random.default_rng() if rng is None else rng
    return [
        generate_valid_trajectory(
            world,
            rng,
            n_points=n_points,
            clearance=clearance,
            max_turn_deg=max_turn_deg,
            max_tries=max_tries,
        )
        for _ in range(n)
    ]


# --- bad / high-energy generators (Task 1.4) --------------------------------


def _is_clearly_bad(
    trajectory: Trajectory,
    world: GridWorld,
    *,
    smooth_turn_deg: float = 80.0,
) -> bool:
    """Is this trajectory unambiguously a high-energy example?

    "Bad" is broader than "invalid": a path qualifies if it either fails the
    world's validity check (collides or misses an endpoint) *or* is sharply
    non-smooth. The smoothness gate (``smooth_turn_deg = 80`` by default) sits
    above the valid generator's ``75`` ceiling, so the two sets never overlap.
    """
    if not trajectory.is_valid(world):
        return True
    return _max_turn_angle(trajectory.points) > smooth_turn_deg


def _bad_random_walk(
    world: GridWorld,
    rng: np.random.Generator,
    n_points: int,
    *,
    step_scale: float = 0.08,
) -> np.ndarray:
    """A Brownian wander from the start, clamped to the world. Ignores the goal."""
    start = np.asarray(world.start, dtype=float)
    steps = rng.normal(0.0, step_scale * world.size, size=(n_points - 1, 2))
    pts = np.concatenate([start[None, :], start[None, :] + np.cumsum(steps, axis=0)])
    return world.clamp(pts)


def _bad_zigzag(
    world: GridWorld,
    rng: np.random.Generator,
    n_points: int,
) -> np.ndarray:
    """Start-to-goal with large alternating sideways swings and sharp corners."""
    start = np.asarray(world.start, dtype=float)
    goal = np.asarray(world.goal, dtype=float)
    direction = goal - start
    span = float(np.linalg.norm(direction))
    perp = (
        np.array([-direction[1], direction[0]]) / span
        if span > 0.0
        else np.array([0.0, 1.0])
    )

    n_knots = int(rng.integers(4, 8))
    t = np.linspace(0.0, 1.0, n_knots)
    base = start[None, :] + t[:, None] * direction[None, :]

    amplitude = rng.uniform(0.15, 0.35) * max(span, 0.3 * world.size)
    signs = rng.choice([-1.0, 1.0]) * ((-1.0) ** np.arange(n_knots))
    signs[0] = 0.0  # keep the endpoints anchored at start/goal
    signs[-1] = 0.0
    knots = base + (amplitude * signs)[:, None] * perp[None, :]

    knots = world.clamp(knots)
    knots = _dedupe_consecutive(knots)
    return _resample_by_arclength(knots, n_points)


def _bad_crossing(
    world: GridWorld,
    rng: np.random.Generator,
    n_points: int,
) -> Optional[np.ndarray]:
    """Start-to-goal routed straight through one or two obstacle centres.

    Returns ``None`` when the scene has no obstacles to cross.
    """
    if not world.obstacles:
        return None

    start = np.asarray(world.start, dtype=float)
    goal = np.asarray(world.goal, dtype=float)
    direction = goal - start
    span_sq = float(direction @ direction) + 1e-12

    centers = np.stack([o.center for o in world.obstacles], axis=0)
    # Project each obstacle onto the start->goal segment to find (a) how far off
    # the direct line it sits and (b) where along the line it falls.
    proj_t = np.clip(((centers - start) @ direction) / span_sq, 0.0, 1.0)
    nearest = start[None, :] + proj_t[:, None] * direction[None, :]
    off_line = np.linalg.norm(centers - nearest, axis=1)

    # Cross the obstacles closest to the direct route, ordered along the path.
    k = min(int(rng.integers(1, 3)), len(world.obstacles))
    chosen = np.argsort(off_line)[:k]
    chosen = chosen[np.argsort(proj_t[chosen])]

    waypoints = np.stack([start, *centers[chosen], goal], axis=0)
    waypoints = _dedupe_consecutive(waypoints)
    if waypoints.shape[0] < 2:
        return None
    return _resample_by_arclength(waypoints, n_points)


def generate_bad_trajectory(
    world: GridWorld,
    rng: Optional[np.random.Generator] = None,
    *,
    n_points: int = 48,
    mode: Optional[str] = None,
    ensure_bad: bool = True,
    max_tries: int = 12,
) -> Trajectory:
    """Generate one bad (high-energy) trajectory.

    Parameters
    ----------
    world:
        The scene the path is bad *with respect to*.
    rng:
        Seeded generator for reproducibility. A fresh default is used if None.
    n_points:
        Number of points ``N`` — matched to the valid generator so the model
        sees comparably shaped inputs for both classes.
    mode:
        One of :data:`BAD_MODES`, or None to pick at random. ``"crossing"``
        needs obstacles; on an empty world it falls back to ``"zigzag"``.
    ensure_bad:
        When True (default), keep drawing until the path is "clearly bad" (see
        :func:`_is_clearly_bad`), so an accidentally-fine route never slips into
        the high-energy set. A random walk is the guaranteed-bad fallback.
    max_tries:
        Draws attempted before falling back to a guaranteed-bad random walk.
    """
    rng = np.random.default_rng() if rng is None else rng
    available = BAD_MODES if world.obstacles else ("random_walk", "zigzag")

    def _draw(chosen_mode: str) -> np.ndarray:
        if chosen_mode == "random_walk":
            return _bad_random_walk(world, rng, n_points)
        if chosen_mode == "zigzag":
            return _bad_zigzag(world, rng, n_points)
        if chosen_mode == "crossing":
            pts = _bad_crossing(world, rng, n_points)
            return pts if pts is not None else _bad_zigzag(world, rng, n_points)
        raise ValueError(f"Unknown bad mode {chosen_mode!r}; expected one of {BAD_MODES}.")

    for _ in range(max_tries):
        chosen_mode = mode if mode is not None else str(rng.choice(available))
        candidate = Trajectory(_draw(chosen_mode))
        if not ensure_bad or _is_clearly_bad(candidate, world):
            return candidate

    # Fallback: a wider random walk almost surely misses the goal -> bad.
    return Trajectory(_bad_random_walk(world, rng, n_points, step_scale=0.12))


def generate_bad_trajectories(
    world: GridWorld,
    n: int,
    rng: Optional[np.random.Generator] = None,
    *,
    n_points: int = 48,
    ensure_bad: bool = True,
    max_tries: int = 12,
) -> list[Trajectory]:
    """Generate ``n`` bad trajectories for one scene (the high-energy set).

    Modes are cycled across the batch so it always spans the available failure
    flavours rather than landing on one. Shares a single RNG for reproducible,
    varied draws.
    """
    rng = np.random.default_rng() if rng is None else rng
    available = BAD_MODES if world.obstacles else ("random_walk", "zigzag")
    return [
        generate_bad_trajectory(
            world,
            rng,
            n_points=n_points,
            mode=available[i % len(available)],
            ensure_bad=ensure_bad,
            max_tries=max_tries,
        )
        for i in range(n)
    ]


def _demo() -> None:
    """Textual self-check. Run ``python generators.py`` to exercise the generator."""
    rng = np.random.default_rng(11)
    world = GridWorld.random(rng, n_obstacles=(4, 6))

    paths = generate_valid_trajectories(world, 6, rng)
    print("Valid trajectory generator")
    print(f"  world obstacles : {len(world.obstacles)}")
    print(f"  paths generated : {len(paths)}")

    for i, p in enumerate(paths):
        assert p.is_valid(world), f"path {i} failed validation"
        # Smoothness proxy: mean absolute turning angle between segments.
        seg = p.segments
        unit = seg / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-9)
        cos = np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)
        mean_turn = float(np.degrees(np.arccos(cos)).mean())
        print(
            f"    [{i}] points={p.num_points} arc_length={p.arc_length:.3f} "
            f"mean_turn={mean_turn:.2f}deg"
        )

    # Variety: distinct seeds should not collapse to the same route.
    spread = np.mean([
        np.linalg.norm(paths[0].points - paths[k].points, axis=1).mean()
        for k in range(1, len(paths))
    ])
    print(f"  mean route spread vs path 0 : {spread:.3f}")
    assert spread > 1e-3, "generated paths are not distinct"

    # Bad / high-energy generator: every flavour must come out clearly bad and
    # share the valid paths' point count.
    print("Bad trajectory generator")
    for m in BAD_MODES:
        bad = generate_bad_trajectory(world, rng, mode=m, n_points=paths[0].num_points)
        assert bad.num_points == paths[0].num_points
        assert _is_clearly_bad(bad, world), f"{m} produced a not-clearly-bad path"
        max_turn = _max_turn_angle(bad.points)
        print(
            f"    {m:<12} points={bad.num_points} valid={bad.is_valid(world)} "
            f"max_turn={max_turn:.1f}deg arc_length={bad.arc_length:.3f}"
        )

    batch = generate_bad_trajectories(world, 6, rng)
    assert all(_is_clearly_bad(b, world) for b in batch)
    print(f"  batch of {len(batch)} bad paths, all clearly bad.")
    print("  self-check passed.")


if __name__ == "__main__":
    _demo()
