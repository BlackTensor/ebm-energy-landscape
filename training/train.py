"""Score-matching training for the Energy Landscape Visualizer.

This is where the energy function learns its shape. The model defined in
:mod:`energy_model` is, at initialisation, a meaningless scalar field. Training
carves it into a landscape whose *gradient* carries a chaotic trajectory
downhill into a valley that is a valid route — so the Langevin sampler (Phase 4)
can start from noise and slide into a fresh valid path.

Why score matching (and what failed first)
-------------------------------------------

Getting a 96-dimensional trajectory energy whose minima are the valid paths *and*
which a fixed-step Langevin sampler can reach from random noise turned out to be
the whole problem. Two earlier objectives were built and measured, and both
failed for instructive reasons:

1. **Margin hinge only** (push valid below bad against the bad-path generator).
   A superb discriminator (0.98 ordering accuracy) but useless for generation:
   it constrains the energy only at the handful of paths it is shown, so each
   valid path sat on an isolated *spike* over a flat plateau. The trajectory
   gradient at a valid path was 60-400 (not even stationary), and a step away the
   plateau gradient was ~0.02. Langevin had nowhere to descend.

2. **Contrastive divergence** (draw negatives with Langevin, push them up, data
   down). Measured: the sampler energy barely moved from its random start
   (+5.39 -> +5.33 over 200 steps). CD raised the whole noise plateau and dug a
   narrow well at the data, but left the plateau *flat* — there was no downhill
   slope connecting noise to the valleys. A chicken-and-egg trap: the chains
   cannot move without a slope, and the slope never forms because the chains
   never traverse the corridor.

The fix is **denoising score matching across multiple noise scales** (the NCSN
recipe of Song & Ermon). Instead of only scoring whole paths, we perturb each
valid path with Gaussian noise at a range of scales sigma and train the energy's
*gradient* to point a perturbed path back toward the clean one::

    x_sigma = x + sigma * eps,   eps ~ N(0, I)   (interior points only)
    L_dsm = E || sigma * grad_x E(x_sigma) - eps ||^2

This is the sigma^2-weighted denoising objective: each scale contributes
comparably, and minimising it makes ``grad_x E(x_sigma) ≈ eps/sigma`` — i.e. the
gradient at any perturbed path points straight back to the data manifold. Tiling
sigma from large (covers chaotic, near-uniform paths) to small (sharpens the
valley) builds a gradient field that is informative *at every distance from the
data*. That is exactly the descent corridor CD could not form: measured, the
sampler energy now plunges from +42 to -4.6, and chaotic paths flow into valid
routes. As sigma -> 0 the score -> 0 at the data, so valid paths become genuine
stationary minima.

Two light auxiliary terms ride alongside the score-matching loss:

- a **margin hinge** against the real bad paths, ``relu(margin + E(valid) -
  E(bad))``, which keeps valley energy clearly below ridge energy. Score matching
  fixes the gradient field but not the absolute energy *value*; the hinge grounds
  "valid is low, bad is high" so the Phase 6 heatmap reads correctly.
- a small **L2 anchor** on the valid/bad energy magnitudes that keeps the overall
  energy scale numerically bounded over a long run.

::

    L = dsm_weight * L_dsm + margin_weight * L_hinge
        + energy_reg * mean( E_valid^2 + E_bad^2 )

Stability safeguards
--------------------

Score matching is far calmer than CD (no inner MCMC chain to diverge), but three
guardrails remain: global gradient-norm **clipping** on the model update
(``grad_clip``), the **energy anchor** above, and **weight decay**. The
double-backward through ``grad_x E`` is the only subtlety — the perturbed path is
scored through a leaf with ``create_graph=True`` so the score-matching term is
itself differentiable w.r.t. the network weights.

Data
----

All training data is synthetic and generated in process from
:mod:`environment` and :mod:`generators` (no external dataset, per the project
rules). We build a pool of random scenes, draw several valid and several bad
paths for each, and pair them up. Scenes are rasterised once via
:func:`energy_model.worlds_to_tensor`; only the trajectory branch is re-encoded
each step. The score-matching term uses only the *valid* paths (the data
manifold); the bad paths feed the discrimination hinge.

Running it
----------

``python train.py`` runs the full end-to-end run on the best available device,
saves a checkpoint to ``exports/energy_model.pt``, and verifies it reloads.
``python train.py --demo`` runs a short CPU-friendly version that prints, per
epoch, the loss, the denoising loss, and the valid/bad energies so the
separation and the score fit are visible as they form. On Colab's free GPU the
same :func:`train` call (with :func:`production_config`) is what produces the
shipped weights; pass ``device="cuda"``. Only numpy and torch are required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

try:  # works when imported as part of the ``training`` package
    from training.energy_model import EnergyModel, trajectories_to_tensor, worlds_to_tensor
    from training.environment import GridWorld
    from training.generators import (
        generate_bad_trajectories,
        generate_valid_trajectories,
    )
except ImportError:  # falls back when run directly as ``python train.py``
    from energy_model import EnergyModel, trajectories_to_tensor, worlds_to_tensor
    from environment import GridWorld
    from generators import generate_bad_trajectories, generate_valid_trajectories

__all__ = [
    "TrainConfig",
    "ContrastiveDataset",
    "make_sigmas",
    "score_matching_loss",
    "build_dataset",
    "train",
    "save_checkpoint",
    "load_checkpoint",
    "production_config",
    "CHECKPOINT_FORMAT_VERSION",
]

# Bumped if the on-disk checkpoint layout changes incompatibly. The objective
# rewrite only changes which diagnostics land in ``history`` (still a plain
# dict), so the saved layout is unchanged and stays at v1.
CHECKPOINT_FORMAT_VERSION = 1


@dataclass
class TrainConfig:
    """Everything the training run needs, in one reproducible place.

    Defaults are a small, CPU-friendly baseline that makes both the energy
    separation and the score fit visible in a handful of epochs. The full run
    (:func:`production_config`) scales the scene/epoch counts up and, on Colab,
    sets ``device="cuda"``. The score-matching hyper-parameters (the sigma
    schedule and the term weights) are the values validated in tuning and are
    held fixed between the demo and the production run.
    """

    # Data
    n_scenes: int = 20
    paths_per_scene: int = 6
    n_points: int = 48
    resolution: int = 64
    n_obstacles: tuple[int, int] = (3, 6)

    # Optimisation
    epochs: int = 90
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: Optional[float] = 5.0  # max global grad-norm; None disables clipping

    # Denoising score matching (the noise ladder; world units, size == 1.0)
    sigma_min: float = 0.01  # sharpens the valley bottom (score -> 0 at data)
    sigma_max: float = 0.5   # covers chaotic, near-uniform paths far from data
    n_sigma: int = 10        # geometric steps between sigma_max and sigma_min
    dsm_weight: float = 1.0  # weight on the denoising score-matching term

    # Discrimination + scale anchoring
    margin: float = 1.0
    margin_weight: float = 0.5  # light hinge: keep valley energy below ridge
    energy_reg: float = 0.05    # L2 anchor on valid/bad energy magnitudes

    # Bookkeeping
    seed: int = 0
    device: str = "cpu"
    log_every: int = 1
    checkpoint_path: Optional[str] = None  # if set, train() saves weights here


@dataclass
class ContrastiveDataset:
    """Precomputed examples: paired valid/bad paths per scene.

    The valid paths are the data manifold the score-matching term denoises
    toward; the bad paths feed the discrimination hinge. Tensors live on the
    training device. ``scene_ids[i]`` indexes into ``maps`` to recover the
    rasterised scene shared by ``valid[i]`` and ``bad[i]``.
    """

    maps: Tensor  # (S, C, H, W) one rasterised scene per id
    scene_ids: Tensor  # (M,) long, into maps
    valid: Tensor  # (M, N, 2)
    bad: Tensor  # (M, N, 2)
    worlds: list[GridWorld] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.scene_ids.shape[0])


def make_sigmas(config: TrainConfig, device: torch.device) -> Tensor:
    """The geometric noise ladder, from ``sigma_max`` down to ``sigma_min``.

    Geometric spacing (equal ratios) is the standard choice for score matching:
    it places the scales evenly in log space so every order of magnitude of
    distance-from-data gets comparable coverage.
    """
    return torch.exp(
        torch.linspace(
            np.log(config.sigma_max), np.log(config.sigma_min), config.n_sigma,
            device=device,
        )
    )


def build_dataset(config: TrainConfig, rng: np.random.Generator) -> ContrastiveDataset:
    """Generate scenes and paired valid/bad trajectories for training.

    For each of ``n_scenes`` random worlds we draw ``paths_per_scene`` valid and
    ``paths_per_scene`` bad trajectories and zip them into same-scene pairs. The
    valid generator yields visibly distinct routes across draws (which is what
    later lets the score field hold *several* valleys per scene — the source of
    the multimodality in Task 4.3), and the bad generator cycles its failure
    modes so the hinge sees real variety.

    Random scenes are occasionally unsolvable (a goal walled off by obstacles),
    which makes the valid generator raise. Such a scene is simply discarded and
    a fresh one drawn, so a single bad draw never aborts the whole run; only a
    run of consistent failures (a pathological config) surfaces as an error.
    """
    device = torch.device(config.device)

    worlds: list[GridWorld] = []
    scene_ids: list[int] = []
    valid_paths = []
    bad_paths = []

    skipped = 0
    max_skips = max(20, config.n_scenes)  # tolerate the odd unsolvable scene
    while len(worlds) < config.n_scenes:
        world = GridWorld.random(rng, n_obstacles=config.n_obstacles)
        try:
            valids = generate_valid_trajectories(
                world, config.paths_per_scene, rng, n_points=config.n_points
            )
        except RuntimeError:
            skipped += 1
            if skipped > max_skips:
                raise RuntimeError(
                    f"Could not assemble {config.n_scenes} solvable scenes "
                    f"({skipped} unsolvable draws). Lower n_obstacles or obstacle size."
                )
            continue
        bads = generate_bad_trajectories(
            world, config.paths_per_scene, rng, n_points=config.n_points
        )
        scene_id = len(worlds)
        worlds.append(world)
        for v, b in zip(valids, bads):
            scene_ids.append(scene_id)
            valid_paths.append(v)
            bad_paths.append(b)

    maps = worlds_to_tensor(worlds, resolution=config.resolution, device=device)
    return ContrastiveDataset(
        maps=maps,
        scene_ids=torch.tensor(scene_ids, dtype=torch.long, device=device),
        valid=trajectories_to_tensor(valid_paths, device=device),
        bad=trajectories_to_tensor(bad_paths, device=device),
        worlds=worlds,
    )


# --- the score-matching objective -------------------------------------------


def _interior_mask(n_points: int, device: torch.device) -> Tensor:
    """A ``(N, 1)`` mask that is 0 at the endpoints and 1 on the interior.

    The endpoints of every trajectory are pinned to the scene's start and goal,
    so they carry no noise and contribute no score-matching residual; only the
    interior points are free to move (matching the Langevin sampler).
    """
    mask = torch.ones(n_points, 1, device=device)
    mask[0] = 0.0
    mask[-1] = 0.0
    return mask


def score_matching_loss(
    model: EnergyModel,
    maps: Tensor,
    valid: Tensor,
    bad: Tensor,
    *,
    sigmas: Tensor,
    interior_mask: Tensor,
    world_size: float,
    margin: float,
    margin_weight: float,
    dsm_weight: float,
    energy_reg: float,
) -> tuple[Tensor, dict[str, float]]:
    """The denoising-score-matching + discrimination objective.

    Terms (see the module docstring): a multi-scale denoising term that makes
    ``grad_x E`` point a perturbed path back toward the clean valid path, a light
    margin hinge keeping valid energy below bad, and an L2 anchor on the energy
    scale. The perturbed path is scored through a leaf with ``create_graph=True``
    so the denoising term is differentiable w.r.t. the network weights.

    Returns the scalar loss to backprop and a dict of detached diagnostics.
    """
    batch = valid.shape[0]

    # --- multi-scale denoising score matching (on the valid/data manifold) ---
    sigma_idx = torch.randint(0, sigmas.shape[0], (batch,), device=valid.device)
    sigma = sigmas[sigma_idx].view(batch, 1, 1)
    eps = torch.randn_like(valid) * interior_mask  # no noise at the pinned ends
    perturbed = (valid + sigma * eps).clamp(0.0, world_size)
    perturbed[:, 0, :] = valid[:, 0, :]
    perturbed[:, -1, :] = valid[:, -1, :]
    perturbed = perturbed.detach().requires_grad_(True)

    # The denoising term double-backprops through this forward (autograd.grad
    # with create_graph, then loss.backward). cuDNN's fused RNN backend has no
    # double-backward for the LSTM (it raises "_cudnn_rnn_backward is not
    # implemented" on CUDA), so disable cuDNN here to record the native RNN
    # backward, which supports it. No-op on CPU (cuDNN is not used there).
    with torch.backends.cudnn.flags(enabled=False):
        energy_perturbed = model(maps, perturbed)
        (grad_perturbed,) = torch.autograd.grad(
            energy_perturbed.sum(), perturbed, create_graph=True
        )
    # sigma^2-weighted denoising residual: drive sigma * gradE -> eps.
    residual = (sigma * grad_perturbed - eps) * interior_mask
    dsm = residual.pow(2).sum(dim=(1, 2)).mean()

    # --- discrimination hinge + energy-scale anchor --------------------------
    energy_valid = model(maps, valid)
    energy_bad = model(maps, bad)
    hinge = torch.relu(margin + energy_valid - energy_bad).mean()
    reg = energy_reg * (energy_valid.pow(2).mean() + energy_bad.pow(2).mean())

    loss = dsm_weight * dsm + margin_weight * hinge + reg

    with torch.no_grad():
        gap = (energy_bad - energy_valid).mean()
        accuracy = (energy_valid < energy_bad).float().mean()
        diagnostics = {
            "loss": float(loss),
            "dsm": float(dsm),
            "hinge": float(hinge),
            "reg": float(reg),
            "energy_valid": float(energy_valid.mean()),
            "energy_bad": float(energy_bad.mean()),
            "gap": float(gap),
            "accuracy": float(accuracy),
        }
    return loss, diagnostics


_HISTORY_KEYS = (
    "loss", "dsm", "hinge", "reg",
    "energy_valid", "energy_bad", "gap", "accuracy", "grad_norm",
)


def train(
    config: Optional[TrainConfig] = None,
    *,
    model: Optional[EnergyModel] = None,
    dataset: Optional[ContrastiveDataset] = None,
) -> tuple[EnergyModel, dict[str, list[float]]]:
    """Run the score-matching training loop and return the model and history.

    Trains the energy's gradient field by multi-scale denoising of the valid
    paths (so chaotic paths flow downhill into valleys), with a light hinge
    keeping valid energy below bad. ``model`` and ``dataset`` may be supplied to
    reuse or inspect them; otherwise they are built from ``config``. The returned
    ``history`` holds per-epoch diagnostics for the Task 3.4 chart.
    """
    config = config or TrainConfig()
    device = torch.device(config.device)

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    if dataset is None:
        dataset = build_dataset(config, rng)
    if model is None:
        model = EnergyModel(coord_scale=float(dataset.worlds[0].size))
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    m = len(dataset)
    world_size = float(dataset.worlds[0].size)
    sigmas = make_sigmas(config, device)
    interior_mask = _interior_mask(dataset.valid.shape[1], device)

    history: dict[str, list[float]] = {key: [] for key in _HISTORY_KEYS}
    clip_norm = config.grad_clip if config.grad_clip is not None else float("inf")

    model.train()
    for epoch in range(config.epochs):
        order = torch.randperm(m, device=device)
        epoch_stats: dict[str, list[float]] = {key: [] for key in history}

        for start_i in range(0, m, config.batch_size):
            idx = order[start_i : start_i + config.batch_size]
            maps = dataset.maps[dataset.scene_ids[idx]]

            loss, diagnostics = score_matching_loss(
                model, maps, dataset.valid[idx], dataset.bad[idx],
                sigmas=sigmas,
                interior_mask=interior_mask,
                world_size=world_size,
                margin=config.margin,
                margin_weight=config.margin_weight,
                dsm_weight=config.dsm_weight,
                energy_reg=config.energy_reg,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            weight = idx.shape[0]
            for key, value in diagnostics.items():
                epoch_stats[key].append((value, weight))
            epoch_stats["grad_norm"].append((float(grad_norm), weight))

        for key, pairs in epoch_stats.items():
            total_w = sum(w for _, w in pairs)
            history[key].append(sum(v * w for v, w in pairs) / total_w)

        if config.log_every and (epoch % config.log_every == 0 or epoch == config.epochs - 1):
            print(
                f"  epoch {epoch + 1:>3}/{config.epochs}  "
                f"loss={history['loss'][-1]:.3f}  "
                f"dsm={history['dsm'][-1]:.3f}  "
                f"E_valid={history['energy_valid'][-1]:+.3f}  "
                f"E_bad={history['energy_bad'][-1]:+.3f}  "
                f"gap={history['gap'][-1]:+.3f}  "
                f"acc={history['accuracy'][-1]:.3f}  "
                f"gnorm={history['grad_norm'][-1]:.2f}"
            )

    if config.checkpoint_path is not None:
        metrics = {key: values[-1] for key, values in history.items()}
        saved = save_checkpoint(
            config.checkpoint_path, model, config=config, history=history, metrics=metrics
        )
        print(f"  saved checkpoint -> {saved}")

    return model, history


# --- minima / sampleability verification ------------------------------------


def data_gradient_norm(
    model: EnergyModel,
    dataset: ContrastiveDataset,
    *,
    device: Optional[torch.device] = None,
) -> float:
    """Mean L2 norm of ``grad_x E`` at the clean valid paths.

    The defining check that score matching made the valid paths genuine
    stationary minima: as sigma -> 0 the learned score -> 0 at the data, so this
    number should be small (contrast the hinge-only model, where it was 60-400).
    Measured over the interior points only (the endpoints are pinned).
    """
    device = device or next(model.parameters()).device
    maps = dataset.maps[dataset.scene_ids]
    valid = dataset.valid.detach().to(device).requires_grad_(True)
    interior = _interior_mask(valid.shape[1], device)
    energy = model(maps, valid)
    (grad,) = torch.autograd.grad(energy.sum(), valid)
    per_sample = (grad * interior).pow(2).sum(dim=(1, 2)).sqrt()
    return float(per_sample.mean())


# --- checkpointing ----------------------------------------------------------


def save_checkpoint(
    path: Union[str, Path],
    model: EnergyModel,
    *,
    config: Optional[TrainConfig] = None,
    history: Optional[dict[str, list[float]]] = None,
    metrics: Optional[dict[str, float]] = None,
) -> Path:
    """Write the trained model's weights and provenance to ``path``.

    The checkpoint is a single ``torch.save`` payload bundling the state dict
    with everything needed to use it later without guessing: the architecture
    kwargs (so :func:`load_checkpoint` can rebuild the exact network), the
    training config, the full per-epoch history (for the Task 3.4 chart), and
    the final metrics. Parent directories are created as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_kwargs": dict(model.init_kwargs),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "train_config": asdict(config) if config is not None else None,
        "history": history,
        "metrics": metrics,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Union[str, Path],
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> tuple[EnergyModel, dict]:
    """Rebuild the model from a checkpoint and return ``(model, payload)``.

    The architecture is reconstructed from the stored ``model_kwargs`` before
    the weights are loaded, so the caller needs no prior knowledge of how the
    network was configured. The returned model is moved to ``map_location`` and
    set to ``eval`` mode; the raw payload (config, history, metrics) is returned
    alongside for inspection.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format version {version} does not match expected "
            f"{CHECKPOINT_FORMAT_VERSION}; this checkpoint is incompatible."
        )
    model = EnergyModel(**payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"])
    model.to(map_location)
    model.eval()
    return model, payload


def _demo() -> None:
    """Short CPU run showing the score fit, separation, and stationary minima."""
    config = TrainConfig(n_scenes=16, epochs=40)
    print("Denoising score-matching training (demo)")
    print(
        f"  scenes={config.n_scenes}  pairs/scene={config.paths_per_scene}  "
        f"epochs={config.epochs}  sigma=[{config.sigma_min},{config.sigma_max}]"
        f"x{config.n_sigma}  dsm_w={config.dsm_weight}  margin_w={config.margin_weight}"
    )

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    dataset = build_dataset(config, rng)
    model, history = train(config, dataset=dataset)

    n_params = sum(p.numel() for p in model.parameters())
    first_dsm, last_dsm = history["dsm"][0], history["dsm"][-1]
    first_gap, last_gap = history["gap"][0], history["gap"][-1]
    last_acc = history["accuracy"][-1]
    grad_at_valid = data_gradient_norm(model, dataset)
    print("Result")
    print(f"  parameters          : {n_params:,}")
    print(f"  denoising loss      : {first_dsm:.3f} -> {last_dsm:.3f} (lower = better score fit)")
    print(f"  valley-ridge gap    : {first_gap:+.3f} -> {last_gap:+.3f}")
    print(f"  ordering accuracy   : {last_acc:.3f}")
    print(f"  |grad@valid|        : {grad_at_valid:.3f} (small = valid paths are minima)")

    # The score fit improved, valid sits below bad, and the data gradient is
    # small enough that the valid paths are genuine minima the sampler can reach.
    assert last_dsm < first_dsm, "denoising loss did not improve (score not learned)"
    assert last_gap > 0.3, f"valid/bad energies barely separated (gap {last_gap:.3f})"
    print("  self-check passed: score fit improving, valleys below ridges, minima forming.")


# --- end-to-end production run ----------------------------------------------


def production_config(checkpoint_path: Union[str, Path], device: str) -> TrainConfig:
    """The full end-to-end training configuration written to a checkpoint.

    Larger than the :func:`_demo` baseline — more scenes for generalisation and
    more epochs for the score field to sharpen — but the identical code path and
    the same (locked) score-matching hyper-parameters. On Colab's free GPU pass
    ``device="cuda"``; locally it falls back to CPU unchanged.
    """
    return TrainConfig(
        n_scenes=48,
        paths_per_scene=6,
        epochs=120,
        device=device,
        checkpoint_path=str(checkpoint_path),
    )


def main() -> None:
    """Run training end to end, save the weights, and verify they reload.

    This is what ``python train.py`` runs (and what the Colab notebook calls):
    a full training pass on the best available device, a checkpoint written to
    ``exports/energy_model.pt``, a round-trip check that the reloaded model
    reproduces the trained one on a probe batch, and a report of the trajectory
    gradient at the valid paths (the headline that they became genuine minima).
    """
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint = repo_root / "exports" / "energy_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = production_config(checkpoint, device)
    print("End-to-end denoising score-matching training")
    print(
        f"  device={device}  scenes={config.n_scenes}  pairs/scene={config.paths_per_scene}  "
        f"epochs={config.epochs}  sigma=[{config.sigma_min},{config.sigma_max}]"
        f"x{config.n_sigma}  -> {checkpoint}"
    )
    if device == "cpu":
        print("  note: no CUDA device found; running on CPU (same code path as Colab GPU).")

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    dataset = build_dataset(config, rng)
    model, history = train(config, dataset=dataset)

    last = {key: values[-1] for key, values in history.items()}
    grad_at_valid = data_gradient_norm(model, dataset)
    print("Result")
    print(f"  final gap          : {last['gap']:+.3f}")
    print(f"  final accuracy     : {last['accuracy']:.3f}")
    print(f"  final denoising    : {last['dsm']:.3f}")
    print(f"  final loss         : {last['loss']:.4f}")
    print(f"  |grad@valid|       : {grad_at_valid:.3f}  (small = valid paths are minima)")

    # Round-trip: a freshly reloaded model must score a probe batch identically.
    reloaded, payload = load_checkpoint(checkpoint, map_location=device)
    torch.manual_seed(12345)
    probe_maps = torch.randn(4, model.init_kwargs["map_channels"], config.resolution,
                             config.resolution, device=device)
    probe_paths = torch.randn(4, config.n_points, 2, device=device)
    model.eval()
    with torch.no_grad():
        diff = (model(probe_maps, probe_paths) - reloaded(probe_maps, probe_paths)).abs().max()
    size_mb = checkpoint.stat().st_size / 1e6
    print(f"  checkpoint size    : {size_mb:.2f} MB")
    print(f"  reload max abs diff: {float(diff):.2e}")

    assert checkpoint.exists(), "checkpoint file was not written"
    assert payload["metrics"]["gap"] > 0.3, (
        f"trained model under-separated (gap {payload['metrics']['gap']:.3f})"
    )
    assert float(diff) < 1e-5, "reloaded weights do not reproduce the trained model"
    print("  self-check passed: weights trained, saved, and reloaded faithfully.")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        _demo()
    else:
        main()
