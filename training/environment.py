"""Synthetic grid world for the Energy Landscape Visualizer.

This module defines the *scene* the model reasons about: a bounded continuous
2D world containing a start point, a goal point, and a set of circular
obstacles. Trajectories (defined elsewhere) are ordered lists of continuous
(x, y) points that travel from start to goal without entering an obstacle.

Design choices worth knowing:

- The world is continuous, not a discrete tile grid. Coordinates live in
  ``[0, size] x [0, size]``. We call it a "grid world" by convention, but
  collision and distance are computed in continuous space so the same scene
  can later feed gradient-based (Langevin) sampling.
- Obstacles are discs (centre + radius). Discs give a smooth, differentiable
  notion of "how far inside an obstacle am I", which the energy model can use.
- Everything is parameterised and randomisable. A scene can be built by hand
  or drawn from a seeded RNG so training data is reproducible.

The module is dependency-light (numpy only) so it runs unchanged on the Colab
free tier and on a local machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

# Type aliases kept loose on purpose: a point is anything shaped (2,) and an
# array of points is anything shaped (N, 2). We coerce to float arrays at the
# boundary so callers can pass tuples, lists, or numpy arrays freely.
Point = np.ndarray
Points = np.ndarray


@dataclass(frozen=True)
class Obstacle:
    """A single circular obstacle in continuous space."""

    x: float
    y: float
    radius: float

    @property
    def center(self) -> Point:
        return np.array([self.x, self.y], dtype=float)

    def signed_distance(self, points: Points) -> np.ndarray:
        """Signed distance from each point to this obstacle's boundary.

        Negative inside the disc, zero on the boundary, positive outside.
        Accepts a single point shaped (2,) or a batch shaped (N, 2) and
        always returns a 1D array.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        dist_to_center = np.linalg.norm(pts - self.center, axis=1)
        return dist_to_center - self.radius

    def contains(self, points: Points, margin: float = 0.0) -> np.ndarray:
        """Boolean mask: is each point inside the disc (optionally inflated)?

        ``margin`` inflates the obstacle, which is useful for keeping
        trajectories a safe clearance away from the true boundary.
        """
        return self.signed_distance(points) < margin

    def to_dict(self) -> dict:
        return {"x": float(self.x), "y": float(self.y), "radius": float(self.radius)}

    @classmethod
    def from_dict(cls, data: dict) -> "Obstacle":
        return cls(x=float(data["x"]), y=float(data["y"]), radius=float(data["radius"]))


@dataclass(frozen=True)
class GridWorld:
    """A bounded continuous scene: start, goal, and circular obstacles.

    Attributes
    ----------
    size:
        The world spans ``[0, size] x [0, size]``.
    start, goal:
        Endpoints as (2,) float arrays.
    obstacles:
        The circular obstacles present in the scene.
    """

    size: float = 1.0
    start: Point = field(default_factory=lambda: np.array([0.1, 0.1], dtype=float))
    goal: Point = field(default_factory=lambda: np.array([0.9, 0.9], dtype=float))
    obstacles: tuple[Obstacle, ...] = ()

    # --- geometry -------------------------------------------------------

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of the world."""
        return (0.0, 0.0, self.size, self.size)

    def in_bounds(self, points: Points) -> np.ndarray:
        """Boolean mask: is each point inside the world rectangle?"""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        inside = (
            (pts[:, 0] >= 0.0)
            & (pts[:, 0] <= self.size)
            & (pts[:, 1] >= 0.0)
            & (pts[:, 1] <= self.size)
        )
        return inside

    def clamp(self, points: Points) -> Points:
        """Clip points back into the world rectangle. Preserves input shape."""
        pts = np.asarray(points, dtype=float)
        return np.clip(pts, 0.0, self.size)

    def obstacle_signed_distance(self, points: Points) -> np.ndarray:
        """Distance to the *nearest* obstacle boundary for each point.

        Negative when the point sits inside any obstacle. With no obstacles
        present, returns +inf everywhere (nothing to collide with).
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if not self.obstacles:
            return np.full(pts.shape[0], np.inf)
        per_obstacle = np.stack([o.signed_distance(pts) for o in self.obstacles], axis=0)
        return per_obstacle.min(axis=0)

    def in_collision(self, points: Points, margin: float = 0.0) -> np.ndarray:
        """Boolean mask: does each point hit an obstacle (within ``margin``)?"""
        return self.obstacle_signed_distance(points) < margin

    def is_free(self, points: Points, margin: float = 0.0) -> np.ndarray:
        """Boolean mask: is each point both in bounds and collision-free?"""
        return self.in_bounds(points) & ~self.in_collision(points, margin=margin)

    # --- serialisation --------------------------------------------------

    def to_dict(self) -> dict:
        """Plain-dict form, ready to be written to JSON by the export layer."""
        return {
            "size": float(self.size),
            "start": [float(self.start[0]), float(self.start[1])],
            "goal": [float(self.goal[0]), float(self.goal[1])],
            "obstacles": [o.to_dict() for o in self.obstacles],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GridWorld":
        return cls(
            size=float(data.get("size", 1.0)),
            start=np.asarray(data["start"], dtype=float),
            goal=np.asarray(data["goal"], dtype=float),
            obstacles=tuple(Obstacle.from_dict(o) for o in data.get("obstacles", [])),
        )

    def with_endpoints(self, start: Point, goal: Point) -> "GridWorld":
        """Return a copy with new start/goal, leaving obstacles untouched."""
        return replace(
            self,
            start=np.asarray(start, dtype=float),
            goal=np.asarray(goal, dtype=float),
        )

    # --- factory --------------------------------------------------------

    @classmethod
    def random(
        cls,
        rng: Optional[np.random.Generator] = None,
        *,
        size: float = 1.0,
        n_obstacles: int | tuple[int, int] = (3, 6),
        radius_range: tuple[float, float] = (0.08, 0.16),
        endpoint_margin: float = 0.06,
        start_goal_min_separation: float = 0.5,
        obstacle_clearance: float = 0.04,
        max_tries: int = 2000,
    ) -> "GridWorld":
        """Draw a random, solvable-looking scene from a seeded RNG.

        Guarantees by construction:

        - start and goal sit inside the world, away from the edge by
          ``endpoint_margin``, and are at least ``start_goal_min_separation``
          apart (scaled by ``size``) so paths are non-trivial;
        - neither endpoint lands inside an obstacle (a clearance equal to
          ``obstacle_clearance`` is kept);
        - obstacles do not swallow an endpoint and do not fully overlap each
          other (light rejection sampling keeps scenes readable).

        Parameters mirror the attributes so callers can dial difficulty up or
        down. ``n_obstacles`` may be a fixed int or an inclusive (low, high)
        range to sample from.

        Raises
        ------
        RuntimeError
            If a valid scene could not be assembled within ``max_tries``.
            That only happens with self-contradictory parameters (e.g.
            demanding more large obstacles than fit), and the message says so.
        """
        rng = np.random.default_rng() if rng is None else rng

        lo = endpoint_margin * size
        hi = size - endpoint_margin * size
        min_sep = start_goal_min_separation * size

        # 1. Place start and goal far enough apart to be interesting.
        start = goal = None
        for _ in range(max_tries):
            cand_start = rng.uniform(lo, hi, size=2)
            cand_goal = rng.uniform(lo, hi, size=2)
            if np.linalg.norm(cand_goal - cand_start) >= min_sep:
                start, goal = cand_start, cand_goal
                break
        if start is None:
            raise RuntimeError(
                "Could not place start/goal: start_goal_min_separation is too "
                "large for the available area. Lower it or raise size."
            )

        # 2. Decide how many obstacles to place.
        if isinstance(n_obstacles, tuple):
            n_low, n_high = n_obstacles
            count = int(rng.integers(n_low, n_high + 1))
        else:
            count = int(n_obstacles)

        # 3. Reject-sample obstacles that keep the endpoints free and don't
        #    sit exactly on top of an already-placed obstacle.
        endpoints = np.stack([start, goal], axis=0)
        obstacles: list[Obstacle] = []
        tries = 0
        while len(obstacles) < count and tries < max_tries:
            tries += 1
            radius = float(rng.uniform(*radius_range)) * size
            center = rng.uniform(radius, size - radius, size=2)
            candidate = Obstacle(x=float(center[0]), y=float(center[1]), radius=radius)

            # Keep both endpoints clear of this obstacle: each endpoint must
            # sit outside the disc by at least the clearance band.
            if np.any(candidate.signed_distance(endpoints) < obstacle_clearance * size):
                continue

            # Avoid near-duplicate stacks: reject if centre is deep inside an
            # existing obstacle.
            if obstacles:
                centers = np.stack([o.center for o in obstacles], axis=0)
                radii = np.array([o.radius for o in obstacles])
                d = np.linalg.norm(centers - candidate.center, axis=1)
                if np.any(d < 0.5 * np.maximum(radii, radius)):
                    continue

            obstacles.append(candidate)

        return cls(
            size=float(size),
            start=np.asarray(start, dtype=float),
            goal=np.asarray(goal, dtype=float),
            obstacles=tuple(obstacles),
        )


def _demo() -> None:
    """Small textual self-check. Run ``python environment.py`` to see a scene.

    Intentionally text-only: the premium preview image is a later task (1.5).
    """
    rng = np.random.default_rng(7)
    world = GridWorld.random(rng)

    print("GridWorld")
    print(f"  size           : {world.size}")
    print(f"  start          : ({world.start[0]:.3f}, {world.start[1]:.3f})")
    print(f"  goal           : ({world.goal[0]:.3f}, {world.goal[1]:.3f})")
    print(f"  obstacles      : {len(world.obstacles)}")
    for i, o in enumerate(world.obstacles):
        print(f"    [{i}] center=({o.x:.3f}, {o.y:.3f}) radius={o.radius:.3f}")

    # Sanity: endpoints must be free, and the round-trip through dict must hold.
    assert bool(world.is_free(world.start)[0]), "start should be collision-free"
    assert bool(world.is_free(world.goal)[0]), "goal should be collision-free"
    restored = GridWorld.from_dict(world.to_dict())
    assert np.allclose(restored.start, world.start)
    assert len(restored.obstacles) == len(world.obstacles)

    # Sanity: a batch query returns one boolean per point.
    probe = np.array([world.start, world.goal, [world.size / 2, world.size / 2]])
    free_mask = world.is_free(probe)
    print(f"  free(start, goal, center): {free_mask.tolist()}")
    print("  self-check passed.")


if __name__ == "__main__":
    _demo()
