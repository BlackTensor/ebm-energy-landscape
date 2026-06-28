"""Schema validation for the export bundle (Task 5.3).

Phase 5 writes four JSON files into ``/exports`` (:mod:`export`); Phase 6 reads
them in a browser with no Python in sight. This module is the **contract**
between the two: it pins down exactly the shape, types, value ranges, and
cross-file coherence the web app may rely on, and checks the written files
against it. If this passes, the front end can load the bundle without defensive
parsing; if it fails, the message says which file and which field is wrong.

The bundle
----------

- ``scene.json`` — the board: world ``size``, ``start``, ``goal``, and circular
  ``obstacles``.
- ``energy_field.json`` — the heatmap: a square ``resolution x resolution`` grid
  of ``energy`` plus its ``[0, 1]`` ``energy_normalized`` twin and the ``vmin`` /
  ``vmax`` that relate them.
- ``trajectories.json`` — the multi-sample view: ``count`` valid routes, each a
  list of ``n_points`` ``[x, y]`` pairs with a scalar ``energy``.
- ``descent.json`` — the animation: ``n_descents`` descents, each the path and
  energy at ``n_frames`` Langevin steps (``frame_steps``), spanning chaos (step
  0) to settled (the final step).

Every file carries ``format_version`` equal to
:data:`export.EXPORT_FORMAT_VERSION`; a mismatch is refused so the web app never
silently reads a layout it was not built for.

Cross-file coherence
--------------------

The four files describe *one* scene, so the validator also checks they agree:
all sizes match, the trajectory and descent point counts match, and every route
and every animation frame begins at the scene's ``start`` and ends at its
``goal`` (the endpoints the sampler pins). These are the invariants the renderer
would otherwise have to assume.

Design
------

Dependency-free on purpose (numpy only, already in the stack) — no ``jsonschema``
to install, so it runs unchanged on the Colab free tier and in CI. The checkers
*accumulate* errors rather than raising on the first, so a single run reports
everything wrong with a bundle at once. ``python validate_export.py`` validates
``/exports`` and exits non-zero if anything fails.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:  # works when imported as part of the ``training`` package
    from training.export import EXPORT_FORMAT_VERSION
except ImportError:  # falls back when run directly as ``python validate_export.py``
    from export import EXPORT_FORMAT_VERSION

__all__ = [
    "EXPORT_FILES",
    "ValidationError",
    "validate_scene",
    "validate_energy_field",
    "validate_trajectories",
    "validate_descent",
    "validate_cross_file",
    "validate_bundle",
]

# The four files that make up a complete export bundle.
EXPORT_FILES = ("scene.json", "energy_field.json", "trajectories.json", "descent.json")

# Tolerances. Coordinates are exported rounded (5 dp for the field/routes, 4 dp
# for descent frames); these bands keep the checks robust to that rounding.
_COORD_TOL = 2e-3
_NORM_TOL = 2e-3


class ValidationError(Exception):
    """Raised by :func:`validate_bundle` when one or more files fail the schema."""


# --- small typed checkers ---------------------------------------------------


def _is_number(x: Any) -> bool:
    """A real, finite JSON number (and not a bool, which JSON keeps distinct)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_point(x: Any) -> bool:
    """A ``[x, y]`` pair of finite numbers."""
    return isinstance(x, list) and len(x) == 2 and all(_is_number(v) for v in x)


def _check(errors: list[str], cond: bool, msg: str) -> bool:
    """Record ``msg`` when ``cond`` is false; return ``cond`` for short-circuiting."""
    if not cond:
        errors.append(msg)
    return cond


def _check_version(errors: list[str], data: dict, where: str) -> None:
    version = data.get("format_version")
    _check(
        errors,
        version == EXPORT_FORMAT_VERSION,
        f"{where}: format_version is {version!r}, expected {EXPORT_FORMAT_VERSION}",
    )


# --- per-file validators ----------------------------------------------------


def validate_scene(data: dict) -> list[str]:
    """Validate ``scene.json``: size, start, goal, and circular obstacles."""
    errors: list[str] = []
    _check_version(errors, data, "scene.json")

    scene = data.get("scene")
    if not _check(errors, isinstance(scene, dict), "scene.json: 'scene' must be an object"):
        return errors

    _check(errors, _is_number(scene.get("size")) and scene["size"] > 0,
           "scene.json: scene.size must be a positive number")
    _check(errors, _is_point(scene.get("start")), "scene.json: scene.start must be [x, y]")
    _check(errors, _is_point(scene.get("goal")), "scene.json: scene.goal must be [x, y]")

    obstacles = scene.get("obstacles")
    if _check(errors, isinstance(obstacles, list), "scene.json: scene.obstacles must be a list"):
        for i, o in enumerate(obstacles):
            if not _check(errors, isinstance(o, dict), f"scene.json: obstacle[{i}] must be an object"):
                continue
            _check(errors, _is_number(o.get("x")) and _is_number(o.get("y")),
                   f"scene.json: obstacle[{i}] needs numeric x and y")
            _check(errors, _is_number(o.get("radius")) and o.get("radius", 0) > 0,
                   f"scene.json: obstacle[{i}].radius must be a positive number")
    return errors


def validate_energy_field(data: dict) -> list[str]:
    """Validate ``energy_field.json``: a square grid, its [0,1] twin, and range."""
    errors: list[str] = []
    _check_version(errors, data, "energy_field.json")

    res = data.get("resolution")
    if not _check(errors, _is_int(res) and res > 0, "energy_field.json: resolution must be a positive int"):
        return errors
    _check(errors, _is_number(data.get("size")) and data["size"] > 0,
           "energy_field.json: size must be a positive number")
    _check(errors, isinstance(data.get("orientation"), str),
           "energy_field.json: orientation must be a string")
    _check(errors, isinstance(data.get("definition"), str),
           "energy_field.json: definition must be a string")

    vmin, vmax = data.get("vmin"), data.get("vmax")
    have_range = _check(errors, _is_number(vmin) and _is_number(vmax),
                        "energy_field.json: vmin and vmax must be numbers")
    if have_range:
        _check(errors, vmax >= vmin, f"energy_field.json: vmax ({vmax}) < vmin ({vmin})")

    def _square_grid(key: str) -> Optional[np.ndarray]:
        grid = data.get(key)
        if not _check(errors, isinstance(grid, list) and len(grid) == res,
                      f"energy_field.json: {key} must have {res} rows"):
            return None
        if not _check(errors, all(isinstance(row, list) and len(row) == res for row in grid),
                      f"energy_field.json: {key} rows must each have {res} columns"):
            return None
        arr = np.asarray(grid, dtype=float)
        if not _check(errors, np.all(np.isfinite(arr)), f"energy_field.json: {key} has non-finite values"):
            return None
        return arr

    energy = _square_grid("energy")
    norm = _square_grid("energy_normalized")

    if energy is not None and have_range:
        _check(errors, abs(float(energy.min()) - vmin) <= _COORD_TOL,
               f"energy_field.json: vmin ({vmin:.5f}) != min(energy) ({energy.min():.5f})")
        _check(errors, abs(float(energy.max()) - vmax) <= _COORD_TOL,
               f"energy_field.json: vmax ({vmax:.5f}) != max(energy) ({energy.max():.5f})")

    if norm is not None:
        _check(errors, norm.min() >= -_NORM_TOL and norm.max() <= 1.0 + _NORM_TOL,
               f"energy_field.json: energy_normalized out of [0,1] "
               f"([{norm.min():.5f}, {norm.max():.5f}])")
        # The normalised grid must be the raw grid mapped through [vmin, vmax].
        if energy is not None and have_range and vmax > vmin:
            expected = (energy - vmin) / (vmax - vmin)
            _check(errors, float(np.max(np.abs(expected - norm))) <= _NORM_TOL,
                   "energy_field.json: energy_normalized is inconsistent with energy/vmin/vmax")
    return errors


def validate_trajectories(data: dict) -> list[str]:
    """Validate ``trajectories.json``: count routes, each n_points [x,y] + energy."""
    errors: list[str] = []
    _check_version(errors, data, "trajectories.json")

    count = data.get("count")
    n_points = data.get("n_points")
    _check(errors, _is_int(count) and count >= 1, "trajectories.json: count must be an int >= 1")
    have_n = _check(errors, _is_int(n_points) and n_points >= 2,
                    "trajectories.json: n_points must be an int >= 2")

    routes = data.get("trajectories")
    if not _check(errors, isinstance(routes, list), "trajectories.json: 'trajectories' must be a list"):
        return errors
    if _is_int(count):
        _check(errors, len(routes) == count,
               f"trajectories.json: count ({count}) != number of routes ({len(routes)})")

    for i, route in enumerate(routes):
        if not _check(errors, isinstance(route, dict), f"trajectories.json: route[{i}] must be an object"):
            continue
        _check(errors, _is_number(route.get("energy")), f"trajectories.json: route[{i}].energy must be a number")
        pts = route.get("points")
        if not _check(errors, isinstance(pts, list), f"trajectories.json: route[{i}].points must be a list"):
            continue
        if have_n:
            _check(errors, len(pts) == n_points,
                   f"trajectories.json: route[{i}] has {len(pts)} points, expected {n_points}")
        _check(errors, all(_is_point(p) for p in pts),
               f"trajectories.json: route[{i}] has a malformed point")
    return errors


def validate_descent(data: dict) -> list[str]:
    """Validate ``descent.json``: descents of n_frames paths + energies over steps."""
    errors: list[str] = []
    _check_version(errors, data, "descent.json")

    n_descents = data.get("n_descents")
    n_points = data.get("n_points")
    n_steps = data.get("n_steps")
    n_frames = data.get("n_frames")
    _check(errors, _is_int(n_descents) and n_descents >= 1, "descent.json: n_descents must be an int >= 1")
    have_np = _check(errors, _is_int(n_points) and n_points >= 2, "descent.json: n_points must be an int >= 2")
    _check(errors, _is_int(n_steps) and n_steps >= 1, "descent.json: n_steps must be an int >= 1")
    have_nf = _check(errors, _is_int(n_frames) and n_frames >= 2, "descent.json: n_frames must be an int >= 2")

    steps = data.get("frame_steps")
    if _check(errors, isinstance(steps, list), "descent.json: frame_steps must be a list"):
        if have_nf:
            _check(errors, len(steps) == n_frames,
                   f"descent.json: frame_steps has {len(steps)} entries, expected n_frames={n_frames}")
        if steps and all(_is_int(s) for s in steps):
            _check(errors, steps == sorted(steps) and len(set(steps)) == len(steps),
                   "descent.json: frame_steps must be strictly increasing")
            _check(errors, steps[0] == 0, "descent.json: frame_steps must start at step 0 (chaos)")
            if _is_int(n_steps):
                _check(errors, steps[-1] == n_steps,
                       f"descent.json: frame_steps must end at the final step ({n_steps})")
        else:
            _check(errors, all(_is_int(s) for s in steps), "descent.json: frame_steps must be integers")

    descents = data.get("descents")
    if not _check(errors, isinstance(descents, list), "descent.json: 'descents' must be a list"):
        return errors
    if _is_int(n_descents):
        _check(errors, len(descents) == n_descents,
               f"descent.json: n_descents ({n_descents}) != number of descents ({len(descents)})")

    for i, d in enumerate(descents):
        if not _check(errors, isinstance(d, dict), f"descent.json: descent[{i}] must be an object"):
            continue
        _check(errors, isinstance(d.get("valid"), bool), f"descent.json: descent[{i}].valid must be a bool")
        _check(errors, _is_number(d.get("final_energy")), f"descent.json: descent[{i}].final_energy must be a number")

        energy = d.get("energy")
        if _check(errors, isinstance(energy, list), f"descent.json: descent[{i}].energy must be a list"):
            if have_nf:
                _check(errors, len(energy) == n_frames,
                       f"descent.json: descent[{i}].energy has {len(energy)} entries, expected {n_frames}")
            _check(errors, all(_is_number(e) for e in energy),
                   f"descent.json: descent[{i}].energy has a non-number")

        frames = d.get("frames")
        if not _check(errors, isinstance(frames, list), f"descent.json: descent[{i}].frames must be a list"):
            continue
        if have_nf:
            _check(errors, len(frames) == n_frames,
                   f"descent.json: descent[{i}] has {len(frames)} frames, expected {n_frames}")
        for f, frame in enumerate(frames):
            if not _check(errors, isinstance(frame, list), f"descent.json: descent[{i}].frames[{f}] must be a list"):
                continue
            if have_np:
                _check(errors, len(frame) == n_points,
                       f"descent.json: descent[{i}].frames[{f}] has {len(frame)} points, expected {n_points}")
            _check(errors, all(_is_point(p) for p in frame),
                   f"descent.json: descent[{i}].frames[{f}] has a malformed point")
    return errors


# --- cross-file coherence ---------------------------------------------------


def _endpoints_match(points: list, start: list, goal: list, tol: float) -> bool:
    return (
        _is_point(points[0]) and _is_point(points[-1])
        and abs(points[0][0] - start[0]) <= tol and abs(points[0][1] - start[1]) <= tol
        and abs(points[-1][0] - goal[0]) <= tol and abs(points[-1][1] - goal[1]) <= tol
    )


def validate_cross_file(
    scene: dict, field: dict, trajectories: dict, descent: dict
) -> list[str]:
    """Check the four files describe one coherent scene.

    Sizes agree, point counts agree, and every route and every animation frame
    starts at the scene's ``start`` and ends at its ``goal``. Skipped quietly
    where a per-file check already flagged the relevant field as missing, so this
    layer reports only genuine *inconsistencies* between otherwise-valid files.
    """
    errors: list[str] = []
    s = scene.get("scene", {})
    start, goal, size = s.get("start"), s.get("goal"), s.get("size")
    if not (_is_point(start) and _is_point(goal) and _is_number(size)):
        return errors  # scene itself is malformed; its own validator reported it

    if _is_number(field.get("size")):
        _check(errors, abs(field["size"] - size) <= _COORD_TOL,
               f"cross-file: energy_field.size ({field['size']}) != scene.size ({size})")

    tn = trajectories.get("n_points")
    dn = descent.get("n_points")
    if _is_int(tn) and _is_int(dn):
        _check(errors, tn == dn,
               f"cross-file: trajectories.n_points ({tn}) != descent.n_points ({dn})")

    for i, route in enumerate(trajectories.get("trajectories", []) or []):
        pts = route.get("points") if isinstance(route, dict) else None
        if isinstance(pts, list) and len(pts) >= 2:
            _check(errors, _endpoints_match(pts, start, goal, _COORD_TOL),
                   f"cross-file: trajectory[{i}] endpoints are not pinned to scene start/goal")

    for i, d in enumerate(descent.get("descents", []) or []):
        frames = d.get("frames") if isinstance(d, dict) else None
        if not isinstance(frames, list):
            continue
        for f in (0, -1):  # first (chaos) and last (settled) frames
            frame = frames[f] if frames else None
            if isinstance(frame, list) and len(frame) >= 2:
                _check(errors, _endpoints_match(frame, start, goal, _COORD_TOL),
                       f"cross-file: descent[{i}].frames[{f}] endpoints not pinned to scene start/goal")
    return errors


# --- bundle driver ----------------------------------------------------------

_PER_FILE = {
    "scene.json": validate_scene,
    "energy_field.json": validate_energy_field,
    "trajectories.json": validate_trajectories,
    "descent.json": validate_descent,
}


def validate_bundle(out_dir: Path, *, raise_on_error: bool = True) -> dict[str, list[str]]:
    """Validate the whole ``/exports`` bundle and return a per-file error report.

    Confirms each of the four files exists in ``out_dir``, parses as JSON, passes
    its own schema, and is mutually coherent with the others. Returns a dict
    mapping each file name (plus ``"cross-file"``) to its list of error strings;
    an all-empty report means the bundle is valid. Raises :class:`ValidationError`
    when ``raise_on_error`` and any errors were found.
    """
    out_dir = Path(out_dir)
    report: dict[str, list[str]] = {}
    loaded: dict[str, dict] = {}

    for name in EXPORT_FILES:
        path = out_dir / name
        if not path.exists():
            report[name] = [f"{name}: missing from {out_dir}"]
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report[name] = [f"{name}: invalid JSON ({exc})"]
            continue
        if not isinstance(data, dict):
            report[name] = [f"{name}: top level must be a JSON object"]
            continue
        loaded[name] = data
        report[name] = _PER_FILE[name](data)

    # Cross-file coherence only when all four parsed into objects.
    if len(loaded) == len(EXPORT_FILES):
        report["cross-file"] = validate_cross_file(
            loaded["scene.json"], loaded["energy_field.json"],
            loaded["trajectories.json"], loaded["descent.json"],
        )
    else:
        report["cross-file"] = ["cross-file: skipped (not all files loaded)"]

    total = sum(len(v) for v in report.values())
    if total and raise_on_error:
        lines = [f"  [{name}] {msg}" for name, errs in report.items() for msg in errs]
        raise ValidationError(
            f"Export bundle failed validation ({total} error(s)):\n" + "\n".join(lines)
        )
    return report


def main() -> None:
    """Validate the repository's ``/exports`` bundle and report the result."""
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "exports"

    print("Export schema validation (Task 5.3)")
    print(f"  format version : {EXPORT_FORMAT_VERSION}")
    print(f"  directory      : {out_dir}")

    report = validate_bundle(out_dir, raise_on_error=False)

    ok = True
    for name in (*EXPORT_FILES, "cross-file"):
        errs = report.get(name, [])
        status = "OK" if not errs else f"{len(errs)} ERROR(S)"
        print(f"  {name:<20} {status}")
        for msg in errs:
            ok = False
            print(f"      - {msg}")

    if not ok:
        raise SystemExit("Validation failed: the export bundle does not match the schema.")
    print("  self-check passed: all four files present in /exports and schema-valid.")


if __name__ == "__main__":
    main()
