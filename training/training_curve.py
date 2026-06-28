"""Premium training-curve chart for the Energy Landscape Visualizer (Task 3.4).

Renders the contrastive run saved by :mod:`train` (Task 3.3) in the project's
committed "field instrument" style: deep ``--void`` background, ``--valley`` for
the valid-path energy and ``--ridge`` for the bad-path energy, monospace data
labels, no default matplotlib theme. The chart's job is to make one thing
unmistakable — the energy of valid paths and bad paths *separates* over training
and stays separated, rather than collapsing together.

Two stacked panels share the epoch axis:

- **Energy separation (hero).** Mean energy per epoch for valid (valley) and bad
  (ridge) paths, with the widening gap shaded between them. A collapsed model
  would show the two lines pinned together near zero; a trained one shows the
  valley sinking and the ridge rising.
- **Loss and accuracy.** The contrastive loss descending (ink) and the
  pair-ordering accuracy climbing (muted), kept in the neutral void/ink range so
  the two accent colours stay reserved for energy meaning.

The history is read straight from ``exports/energy_model.pt`` (the checkpoint
bundles the full per-epoch history), so the chart always reflects the weights
that were actually saved. Run ``python training/training_curve.py`` after
training to (re)generate ``assets/training_curve.png``.

Fonts follow :mod:`preview`: the committed faces (Space Grotesk / Inter /
JetBrains Mono) are used when installed, otherwise the closest Windows faces
(Segoe UI, Consolas), degrading to matplotlib's bundled DejaVu families. No
download, zero cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import MaxNLocator

try:  # works as part of the ``training`` package
    from training.train import load_checkpoint
except ImportError:  # falls back when run directly as ``python training_curve.py``
    from train import load_checkpoint


# --- committed palette --------------------------------------------------------

VOID = "#0B0E14"     # deep background
PANEL = "#141925"    # raised surfaces
INK = "#E6EAF2"      # primary text / neutral data
MUTED = "#7A8499"    # secondary text / labels
VALLEY = "#3DD2A4"   # low energy / valid
RIDGE = "#E8633A"    # high energy / invalid

# Derived tints kept inside the void/panel range so accents stay meaningful.
GRID = "#1B2233"
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


def _glow_line(ax, xs, ys, color: str, *, core: float, alpha: float, zorder: int = 3) -> None:
    """Draw a line with a soft glow: wide faint passes under a crisp core."""
    for width, a in ((core * 4.0, alpha * 0.10), (core * 2.2, alpha * 0.18)):
        ax.plot(xs, ys, color=color, linewidth=width, alpha=a,
                solid_capstyle="round", zorder=zorder - 1)
    ax.plot(xs, ys, color=color, linewidth=core, alpha=alpha,
            solid_capstyle="round", zorder=zorder)


def _style_axis(ax, *, mono: bool = True) -> None:
    """Apply the instrument look to an axis: void face, framed, muted mono ticks."""
    ax.set_facecolor(VOID)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.7, zorder=0)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(FRAME)
        spine.set_linewidth(1.0)
    if mono:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(MONO_FONT)
            label.set_color(MUTED)


def render_training_curve(history: dict[str, list[float]], out_path: Path) -> Path:
    """Render the premium training-curve chart and write it to ``out_path``."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    e_valid = np.asarray(history["energy_valid"], dtype=float)
    e_bad = np.asarray(history["energy_bad"], dtype=float)
    loss = np.asarray(history["loss"], dtype=float)
    accuracy = np.asarray(history["accuracy"], dtype=float)
    gap_final = float(e_bad[-1] - e_valid[-1])

    fig = plt.figure(figsize=(9.2, 9.0), dpi=200)
    fig.patch.set_facecolor(VOID)

    # --- header band ---------------------------------------------------------
    fig.text(0.10, 0.945, "ENERGY  LANDSCAPE", color=INK, fontsize=21,
             fontfamily=DISPLAY_FONT, fontweight="bold")
    fig.text(0.102, 0.915, "CONTRASTIVE TRAINING  ·  FIELD INSTRUMENT",
             color=MUTED, fontsize=9.5, fontfamily=MONO_FONT)
    readout = f"{len(epochs)} EPOCHS   ACC {accuracy[-1]:.3f}   GAP {gap_final:+.2f}"
    fig.text(0.90, 0.918, readout, color=MUTED, fontsize=9,
             fontfamily=MONO_FONT, ha="right")

    # --- panel A: energy separation (hero) -----------------------------------
    ax_e = fig.add_axes([0.10, 0.45, 0.80, 0.40])
    _style_axis(ax_e)
    x_pad = max(len(epochs) * 0.015, 0.5)
    ax_e.set_xlim(epochs[0] - x_pad, epochs[-1] + x_pad)
    ax_e.axhline(0.0, color=MUTED, linewidth=0.8, alpha=0.45, zorder=1)

    # The widening gap, shaded in a neutral ink tint (accents stay energy-only).
    ax_e.fill_between(epochs, e_valid, e_bad, color=INK, alpha=0.05, zorder=1)
    _glow_line(ax_e, epochs, e_bad, RIDGE, core=2.0, alpha=0.95)
    _glow_line(ax_e, epochs, e_valid, VALLEY, core=2.0, alpha=0.95)

    # End-of-run readouts in monospace, coloured by energy meaning.
    ax_e.annotate(f"{e_bad[-1]:+.2f}", (epochs[-1], e_bad[-1]),
                  textcoords="offset points", xytext=(8, 2), color=RIDGE,
                  fontsize=9, fontfamily=MONO_FONT)
    ax_e.annotate(f"{e_valid[-1]:+.2f}", (epochs[-1], e_valid[-1]),
                  textcoords="offset points", xytext=(8, -8), color=VALLEY,
                  fontsize=9, fontfamily=MONO_FONT)

    ax_e.set_ylabel("ENERGY  E", color=MUTED, fontsize=9, fontfamily=MONO_FONT,
                    labelpad=8)
    ax_e.set_xticklabels([])  # epoch labels live on the lower panel
    ax_e.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax_e.text(0.012, 0.93, "ENERGY  ·  VALID vs BAD", transform=ax_e.transAxes,
              color=MUTED, fontsize=8.5, fontfamily=MONO_FONT)

    # --- panel B: loss and accuracy ------------------------------------------
    ax_l = fig.add_axes([0.10, 0.115, 0.80, 0.245])
    _style_axis(ax_l)
    ax_l.set_xlim(ax_e.get_xlim())
    _glow_line(ax_l, epochs, loss, INK, core=1.8, alpha=0.9)
    ax_l.set_ylabel("LOSS", color=MUTED, fontsize=9, fontfamily=MONO_FONT, labelpad=8)
    ax_l.set_xlabel("EPOCH", color=MUTED, fontsize=9, fontfamily=MONO_FONT, labelpad=6)
    ax_l.set_ylim(bottom=0.0)
    ax_l.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax_l.text(0.012, 0.90, "LOSS  ·  ACCURACY", transform=ax_l.transAxes,
              color=MUTED, fontsize=8.5, fontfamily=MONO_FONT)

    # Accuracy on a twin axis in muted ink (neutral, not an energy accent).
    ax_a = ax_l.twinx()
    ax_a.set_facecolor("none")
    ax_a.plot(epochs, accuracy, color=MUTED, linewidth=1.5, alpha=0.9,
              linestyle=(0, (5, 2)), zorder=3)
    ax_a.set_ylim(0.45, 1.02)
    ax_a.tick_params(colors=MUTED, length=0, labelsize=8)
    for label in ax_a.get_yticklabels():
        label.set_fontfamily(MONO_FONT)
        label.set_color(MUTED)
    for spine in ax_a.spines.values():
        spine.set_color(FRAME)
        spine.set_linewidth(1.0)
    ax_a.set_ylabel("ACCURACY", color=MUTED, fontsize=9, fontfamily=MONO_FONT,
                    labelpad=8, rotation=270, va="bottom")

    # --- legend strip (bottom) ----------------------------------------------
    legend_y = 0.030
    fig.patches.append(plt.Rectangle((0.10, legend_y), 0.016, 0.011,
                                     transform=fig.transFigure, facecolor=VALLEY,
                                     edgecolor="none"))
    fig.text(0.124, legend_y - 0.0008, "VALLEY  valid energy", color=MUTED,
             fontsize=8.5, fontfamily=MONO_FONT)
    fig.patches.append(plt.Rectangle((0.36, legend_y), 0.016, 0.011,
                                     transform=fig.transFigure, facecolor=RIDGE,
                                     edgecolor="none"))
    fig.text(0.384, legend_y - 0.0008, "RIDGE  bad energy", color=MUTED,
             fontsize=8.5, fontfamily=MONO_FONT)
    fig.text(0.60, legend_y - 0.0008, "LOSS ink solid   ACCURACY muted dashed",
             color=MUTED, fontsize=8.5, fontfamily=MONO_FONT)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=VOID, dpi=200)
    plt.close(fig)
    return out_path


def main(checkpoint: Optional[Path] = None) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint = checkpoint or repo_root / "exports" / "energy_model.pt"
    if not checkpoint.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint}. Run training first "
            f"(python training/train.py) to produce it, then re-run this script."
        )

    _, payload = load_checkpoint(checkpoint)
    history = payload["history"]
    if not history:
        raise SystemExit("Checkpoint has no training history to plot.")

    out_path = repo_root / "assets" / "training_curve.png"
    render_training_curve(history, out_path)

    # Confirm the separation is real, not collapsed: valid sits clearly below
    # bad, and it holds over the final stretch rather than being a lucky epoch.
    e_valid = np.asarray(history["energy_valid"], dtype=float)
    e_bad = np.asarray(history["energy_bad"], dtype=float)
    accuracy = np.asarray(history["accuracy"], dtype=float)
    tail = slice(max(len(e_valid) - 10, 0), None)
    tail_gap = float(np.mean(e_bad[tail] - e_valid[tail]))
    gap_final = float(e_bad[-1] - e_valid[-1])

    print(f"Wrote training curve -> {out_path}")
    print(f"  epochs           : {len(history['loss'])}")
    print(f"  final E_valid     : {e_valid[-1]:+.3f}")
    print(f"  final E_bad       : {e_bad[-1]:+.3f}")
    print(f"  final gap         : {gap_final:+.3f}")
    print(f"  mean gap (last 10): {tail_gap:+.3f}")
    print(f"  final accuracy    : {accuracy[-1]:.3f}")
    print(f"  display font      : {DISPLAY_FONT}")
    print(f"  mono font         : {MONO_FONT}")

    assert tail_gap > 0.5, (
        f"energy separation looks collapsed (mean last-10 gap {tail_gap:+.3f})"
    )
    assert accuracy[-1] > 0.9, f"ordering accuracy too low ({accuracy[-1]:.3f})"
    print("  confirmed: energy separation is real and sustained, not collapsed.")
    return out_path


if __name__ == "__main__":
    main()
