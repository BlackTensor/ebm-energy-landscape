"""Trajectory format for the Energy Landscape Visualizer.

This module pins down *the* representation of a path through a
:class:`~training.environment.GridWorld`, so every later stage — the valid and
bad path generators, the energy network, the Langevin sampler, and the JSON
export — agrees on exactly one shape and one set of conventions.

The trajectory format
---------------------

A trajectory is an **ordered list of continuous (x, y) points** describing a
route from a start location to a goal location. Concretely:

- **Storage.** A single float64 numpy array of shape ``(N, 2)``. Row ``i`` is
  the point ``(x_i, y_i)``; column 0 is x, column 1 is y.
- **Order matters.** Points are listed in the order they are visited. ``[0]``
  is where the path begins, ``[-1]`` is where it ends. Reversing the array
  reverses the path.
- **Continuous coordinates.** Values are real numbers in the world's frame,
  ``[0, size] x [0, size]`` (same frame as ``GridWorld``; default
  ``size = 1.0``). Points are *not* snapped to a grid — the "grid world" name
  is conventional only.
- **Length is the point count ``N``**, not physical distance. Physical
  distance along the path is the :pyattr:`Trajectory.arc_length`. Two
  trajectories may trace the same route with different ``N`` (coarser or finer
  sampling).
- **Minimum size.** A trajectory needs at least two points (a start and an
  end). One point is not a path.
- **Endpoints are just points.** This format does not *enforce* that ``[0]``
  equals the world's start or that the route avoids obstacles. Those are
  properties of *valid* trajectories, checked against a ``GridWorld`` via
  :meth:`Trajectory.is_valid` (and produced by the generators in later tasks).
  Keeping the container dumb lets the same type hold good paths, bad paths, and
  half-finished Langevin samples alike.

JSON form
---------

For the web app, a trajectory serialises as a plain list of ``[x, y]`` pairs::

    [[0.10, 0.10], [0.22, 0.31], ..., [0.90, 0.90]]

See :meth:`Trajectory.to_list` / :meth:`Trajectory.from_list`. The full
descent history exported in Phase 5 is simply a list of these.

The module is dependency-light (numpy only) so it runs unchanged on the Colab
free tier and on a local machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:  # avoid a hard import cycle; only needed for type hints
    from training.environment import GridWorld

# A trajectory's payload is anything coercible to a float array of shape
# (N, 2): a numpy array, a list of (x, y) tuples, etc.
PointsLike = Union[np.ndarray, Sequence[Sequence[float]]]


@dataclass(frozen=True)
class Trajectory:
    """An ordered sequence of continuous (x, y) points from start to goal.

    The canonical container for the project's path format. Construct it from
    anything shaped ``(N, 2)``; the constructor validates and stores an
    immutable, contiguous float64 copy in :pyattr:`points`.

    Examples
    --------
    >>> t = Trajectory([[0.1, 0.1], [0.5, 0.4], [0.9, 0.9]])
    >>> t.num_points
    3
    >>> t.start.tolist(), t.goal.tolist()
    ([0.1, 0.1], [0.9, 0.9])
    """

    points: np.ndarray

    def __init__(self, points: PointsLike):
        arr = np.array(points, dtype=float)  # copy so we own the buffer
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Trajectory points must have shape (N, 2); got {arr.shape}."
            )
        if arr.shape[0] < 2:
            raise ValueError(
                f"A trajectory needs at least 2 points; got {arr.shape[0]}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Trajectory points must all be finite (no NaN/inf).")
        arr.setflags(write=False)  # frozen data for a frozen dataclass
        object.__setattr__(self, "points", arr)

    # --- basic shape / endpoints ---------------------------------------

    def __len__(self) -> int:
        return self.points.shape[0]

    @property
    def num_points(self) -> int:
        """Number of points ``N`` in the trajectory."""
        return self.points.shape[0]

    @property
    def start(self) -> np.ndarray:
        """First point of the path, shape ``(2,)``."""
        return self.points[0]

    @property
    def goal(self) -> np.ndarray:
        """Last point of the path, shape ``(2,)``."""
        return self.points[-1]

    # --- geometry -------------------------------------------------------

    @property
    def segments(self) -> np.ndarray:
        """Per-step displacement vectors, shape ``(N - 1, 2)``."""
        return np.diff(self.points, axis=0)

    @property
    def segment_lengths(self) -> np.ndarray:
        """Euclidean length of each step, shape ``(N - 1,)``."""
        return np.linalg.norm(self.segments, axis=1)

    @property
    def arc_length(self) -> float:
        """Total physical distance travelled along the path."""
        return float(self.segment_lengths.sum())

    def as_array(self) -> np.ndarray:
        """Return a writable copy of the underlying ``(N, 2)`` array."""
        return self.points.copy()

    # --- validation against a scene ------------------------------------

    def is_valid(
        self,
        world: "GridWorld",
        *,
        endpoint_tol: float = 1e-6,
        collision_margin: float = 0.0,
        check_endpoints: bool = True,
    ) -> bool:
        """Does this trajectory describe a legal route through ``world``?

        A trajectory is considered valid when every point is in bounds and
        collision-free and (optionally) its endpoints match the world's start
        and goal. Collision is sampled *at the vertices only*; densely sampled
        trajectories (the kind the generators produce) make that a faithful
        proxy for the continuous path.

        Parameters
        ----------
        world:
            The scene to check against.
        endpoint_tol:
            How close ``[0]``/``[-1]`` must be to ``world.start``/``world.goal``.
        collision_margin:
            Passed through to ``world.is_free``; a positive value demands extra
            clearance from obstacles.
        check_endpoints:
            Set ``False`` to validate only bounds/collision (useful while a
            sample is still being refined and has not yet reached the goal).
        """
        if not bool(np.all(world.is_free(self.points, margin=collision_margin))):
            return False
        if check_endpoints:
            if np.linalg.norm(self.start - np.asarray(world.start, float)) > endpoint_tol:
                return False
            if np.linalg.norm(self.goal - np.asarray(world.goal, float)) > endpoint_tol:
                return False
        return True

    # --- serialisation --------------------------------------------------

    def to_list(self) -> list[list[float]]:
        """JSON-ready nested list of ``[x, y]`` pairs."""
        return self.points.tolist()

    @classmethod
    def from_list(cls, data: PointsLike) -> "Trajectory":
        """Inverse of :meth:`to_list`; accepts anything shaped ``(N, 2)``."""
        return cls(data)


def stack_trajectories(trajectories: Iterable[Trajectory]) -> np.ndarray:
    """Stack equal-length trajectories into one ``(B, N, 2)`` batch array.

    Convenience for the model and export layers. Raises ``ValueError`` if the
    trajectories do not all share the same point count ``N``.
    """
    arrays = [t.points for t in trajectories]
    if not arrays:
        raise ValueError("Cannot stack an empty collection of trajectories.")
    n = arrays[0].shape[0]
    if any(a.shape[0] != n for a in arrays):
        counts = sorted({a.shape[0] for a in arrays})
        raise ValueError(
            f"All trajectories must share the same point count to stack; "
            f"found counts {counts}."
        )
    return np.stack(arrays, axis=0)


def _demo() -> None:
    """Textual self-check. Run ``python trajectory.py`` to exercise the format."""
    t = Trajectory([[0.1, 0.1], [0.4, 0.35], [0.6, 0.7], [0.9, 0.9]])
    print("Trajectory")
    print(f"  num_points : {t.num_points}")
    print(f"  start      : ({t.start[0]:.3f}, {t.start[1]:.3f})")
    print(f"  goal       : ({t.goal[0]:.3f}, {t.goal[1]:.3f})")
    print(f"  arc_length : {t.arc_length:.4f}")

    # Round-trip through the JSON form must be lossless.
    restored = Trajectory.from_list(t.to_list())
    assert np.allclose(restored.points, t.points)

    # Stored points are immutable.
    try:
        t.points[0, 0] = 99.0
    except ValueError:
        pass
    else:
        raise AssertionError("trajectory points should be read-only")

    # Shape and minimum-length contracts are enforced.
    for bad in ([[0.0, 0.0, 0.0]], [[0.1, 0.2]], [[0.0, np.inf], [1.0, 1.0]]):
        try:
            Trajectory(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for malformed input: {bad}")

    # Batch stacking returns (B, N, 2); ragged input is rejected.
    batch = stack_trajectories([t, Trajectory(t.as_array() * 0.5)])
    assert batch.shape == (2, t.num_points, 2)
    print(f"  batch shape: {batch.shape}")
    print("  self-check passed.")


if __name__ == "__main__":
    _demo()
