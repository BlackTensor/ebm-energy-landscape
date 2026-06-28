"""Premium static preview of the synthetic environment (Task 1.5).

Renders one scene from :mod:`training.environment` together with a handful of
valid routes (from :mod:`training.generators`) in the project's committed
"field instrument" style: deep ``--void`` background, ``--valley`` for valid /
low-energy paths, ``--ridge`` for the obstacle (high-energy) regions, and
monospace data labels. No default matplotlib theme is ever shown to the user.

Fonts. The committed faces (Space Grotesk / Inter / JetBrains Mono) are not
installed by default, so to honour the zero-cost, no-download rule we fall back
to the closest faces Windows already ships: Segoe UI for display/body and
Consolas for the monospace data labels. Both degrade gracefully to matplotlib's
bundled DejaVu families if absent.

Run ``python training/preview.py`` to (re)generate ``assets/environment_preview.png``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle

try:  # works as part of the ``training`` package
    from training.environment import GridWorld
    from training.trajectory import Trajectory
    from training.generators import (
        generate_valid_trajectories,
        generate_bad_trajectory,
    )
except ImportError:  # falls back when run directly as ``python preview.py``
    from environment import GridWorld
    from trajectory import Trajectory
    from generators import generate_valid_trajectories, generate_bad_trajectory


# --- committed palette --------------------------------------------------------

VOID = "#0B0E14"     # deep background
PANEL = "#141925"    # raised surfaces
INK = "#E6EAF2"      # primary text
MUTED = "#7A8499"    # secondary text / labels
VALLEY = "#3DD2A4"   # low energy / valid
RIDGE = "#E8633A"    # high energy / invalid

# Derived tints kept inside the void/panel range so accents stay meaningful.
GRID = "#1B2233"
OBSTACLE_FILL = "#191F2E"
FRAME = "#2A3346"


def _pick_font(candidates: Sequence[str], fallback: str) -> str:
    """Return the first installed font from ``candidates``, else ``fallback``."""
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return name
    return fallback


DISPLAY_FONT = _pick_font(["Space Grotesk", "Inter", "Segoe UI"], "DejaVu Sans")
MONO_FONT = _pick_font(["JetBrains Mono", "Consolas"], "DejaVu Sans Mono")


def _glow_line(ax, xs, ys, color: str, *, core: float, alpha: float) -> None:
    """Draw a line with a soft glow: wide faint passes under a crisp core."""
    for width, a in ((core * 4.0, alpha * 0.10), (core * 2.2, alpha * 0.18)):
        ax.plot(xs, ys, color=color, linewidth=width, alpha=a,
                solid_capstyle="round", zorder=2)
    ax.plot(xs, ys, color=color, linewidth=core, alpha=alpha,
            solid_capstyle="round", zorder=3)


def render_environment_preview(
    world: GridWorld,
    valid_paths: Sequence[Trajectory],
    bad_path: Optional[Trajectory],
    out_path: Path,
) -> Path:
    """Render one premium preview of ``world`` and write it to ``out_path``."""
    size = world.size
    fig = plt.figure(figsize=(8.6, 9.4), dpi=200)
    fig.patch.set_facecolor(VOID)

    # Leave a calm margin; the scene sits below a quiet header band.
    ax = fig.add_axes([0.085, 0.075, 0.83, 0.78])
    ax.set_facecolor(VOID)
    # A small view pad keeps routes that skirt the world edge from colliding
    # with the frame, and reads like an instrument bezel around the field.
    pad = 0.035 * size
    ax.set_xlim(-pad, size + pad)
    ax.set_ylim(-pad, size + pad)
    ax.set_aspect("equal")

    # Faint instrument grid + monospace tick labels in muted ink.
    ticks = np.linspace(0, size, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.7, zorder=0)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(MONO_FONT)
        label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_color(FRAME)
        spine.set_linewidth(1.0)

    # Obstacles: dark discs with a ridge boundary and a soft ridge halo, so
    # they read as high-energy "stay out" regions without shouting.
    for o in world.obstacles:
        ax.add_patch(Circle((o.x, o.y), o.radius * 1.18, facecolor=RIDGE,
                             edgecolor="none", alpha=0.06, zorder=1))
        ax.add_patch(Circle((o.x, o.y), o.radius, facecolor=OBSTACLE_FILL,
                             edgecolor=RIDGE, linewidth=1.3, alpha=0.95, zorder=1))

    # One faint bad route in ridge to foreshadow the contrast, kept quiet.
    if bad_path is not None:
        ax.plot(bad_path.points[:, 0], bad_path.points[:, 1], color=RIDGE,
                linewidth=1.4, alpha=0.45, linestyle=(0, (4, 3)), zorder=2)

    # Valid routes in valley, with a soft glow; gentle alpha spread for depth.
    n = max(len(valid_paths), 1)
    for i, p in enumerate(valid_paths):
        alpha = 0.55 + 0.4 * (i / max(n - 1, 1))
        _glow_line(ax, p.points[:, 0], p.points[:, 1], VALLEY,
                   core=2.0, alpha=alpha)

    # Start marker: a hollow ink ring (neutral — it carries no energy meaning).
    sx, sy = world.start
    ax.scatter([sx], [sy], s=190, facecolors="none", edgecolors=INK,
               linewidths=1.8, zorder=5)
    ax.scatter([sx], [sy], s=18, c=INK, zorder=5)
    ax.annotate("START", (sx, sy), textcoords="offset points", xytext=(12, 10),
                color=MUTED, fontsize=8.5, fontfamily=MONO_FONT)
    ax.annotate(f"{sx:.2f}, {sy:.2f}", (sx, sy), textcoords="offset points",
                xytext=(12, -1), color=MUTED, fontsize=7, fontfamily=MONO_FONT,
                alpha=0.8)

    # Goal marker: valley fill with a ring — the desired low-energy endpoint.
    gx, gy = world.goal
    ax.scatter([gx], [gy], s=320, facecolors="none", edgecolors=VALLEY,
               linewidths=1.6, alpha=0.7, zorder=5)
    ax.scatter([gx], [gy], s=130, c=VALLEY, edgecolors=VOID, linewidths=1.5,
               zorder=6)
    ax.annotate("GOAL", (gx, gy), textcoords="offset points", xytext=(12, 10),
                color=VALLEY, fontsize=8.5, fontfamily=MONO_FONT)
    ax.annotate(f"{gx:.2f}, {gy:.2f}", (gx, gy), textcoords="offset points",
                xytext=(12, -1), color=MUTED, fontsize=7, fontfamily=MONO_FONT,
                alpha=0.8)

    # --- header band (figure-space text, instrument styling) ----------------
    fig.text(0.085, 0.945, "ENERGY  LANDSCAPE", color=INK, fontsize=21,
             fontfamily=DISPLAY_FONT, fontweight="bold")
    fig.text(0.087, 0.915, "SYNTHETIC ENVIRONMENT  ·  FIELD INSTRUMENT",
             color=MUTED, fontsize=9.5, fontfamily=MONO_FONT)

    # Right-aligned scene readout in monospace.
    readout = f"{len(world.obstacles)} OBSTACLES   {len(valid_paths)} VALID ROUTES"
    fig.text(0.915, 0.918, readout, color=MUTED, fontsize=9,
             fontfamily=MONO_FONT, ha="right")

    # --- legend strip (bottom) ---------------------------------------------
    legend_y = 0.028
    fig.patches.append(plt.Rectangle((0.085, legend_y), 0.018, 0.012,
                                     transform=fig.transFigure, facecolor=VALLEY,
                                     edgecolor="none"))
    fig.text(0.112, legend_y + 0.0005, "VALLEY  valid / low energy", color=MUTED,
             fontsize=8.5, fontfamily=MONO_FONT)
    fig.patches.append(plt.Rectangle((0.46, legend_y), 0.018, 0.012,
                                     transform=fig.transFigure, facecolor=RIDGE,
                                     edgecolor="none"))
    fig.text(0.487, legend_y + 0.0005, "RIDGE  obstacle / high energy",
             color=MUTED, fontsize=8.5, fontfamily=MONO_FONT)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=VOID, dpi=200)
    plt.close(fig)
    return out_path


def _build_scene(seed: int) -> tuple[GridWorld, list[Trajectory], Trajectory]:
    """Build a deterministic, visually clear scene for the preview."""
    rng = np.random.default_rng(seed)
    world = GridWorld.random(rng, n_obstacles=(4, 5))
    valid = generate_valid_trajectories(world, 6, rng)
    bad = generate_bad_trajectory(world, rng, mode="zigzag")
    return world, valid, bad


def main() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "assets" / "environment_preview.png"
    # Seed 57: valid routes split into two arcs around the obstacle cluster,
    # which best shows the environment affords many distinct valid futures.
    world, valid, bad = _build_scene(seed=57)
    render_environment_preview(world, valid, bad, out_path)
    print(f"Wrote preview -> {out_path}")
    print(f"  obstacles    : {len(world.obstacles)}")
    print(f"  valid routes : {len(valid)}")
    print(f"  display font : {DISPLAY_FONT}")
    print(f"  mono font    : {MONO_FONT}")
    return out_path


if __name__ == "__main__":
    main()
