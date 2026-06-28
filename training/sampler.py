"""Langevin sampler for the Energy Landscape Visualizer (Task 4.1).

This is the generative half of the project. Training (Phase 3) carved the energy
function into a landscape: valid routes sit in low-energy valleys, bad routes on
high-energy ridges. Here we *use* that landscape to invent fresh trajectories.
We drop a chaotic random path into the field and let it slide downhill, nudged
by a little noise so it does not simply collapse onto the single nearest valley.
That noisy gradient descent is **Langevin dynamics**, and it is what turns a
discriminator of paths into a generator of them.

The update
----------

Starting from a random trajectory, we repeat for ``n_steps``::

    path <- path - step_size * grad(E)(path) + noise

term by term:

- ``grad(E)(path)`` is the gradient of the scalar energy with respect to the
  trajectory coordinates — the uphill direction. Subtracting it walks the path
  *downhill*, toward lower energy (a more plausible route). Phase 2.3 verified
  this gradient is correct, finite, and per-sample isolated.
- ``step_size`` (the learning rate ``lr``) sets how far each downhill step
  travels.
- ``noise`` is fresh Gaussian noise scaled by ``noise_scale``. Without it the
  path would settle deterministically into whatever valley it started nearest;
  the noise lets it explore, which is what makes *different random seeds* settle
  into *different* valid routes — the multimodality this whole project is built
  to show (verified in Task 4.3).

Two constraints keep the iterate a usable trajectory. The endpoints are
**pinned** to the world's start and goal (the route must begin and end in the
right places; only the interior is free to move), and every point is **clamped**
to the world rectangle each step (the path cannot wander off the map). Both mirror
the conventions the trajectory generators already follow, so a sampled path is
directly comparable to a generated one.

Scope of this task
------------------

Task 4.1 builds the mechanism: the random initialisation, the Langevin loop, the
endpoint/bounds handling, and the bookkeeping (energy trace and optional full
descent history, which the Phase 5 export layer animates). The *defaults* here
are sensible but not yet tuned — dialling ``step_size``, ``noise_scale``, and
``n_steps`` so chaotic paths reliably reach valid routes is Task 4.2, and
confirming multimodality across seeds is Task 4.3. The API is built to serve
both: ``sample`` runs many seeds at once and can hand back the per-step history.

Only numpy and torch are required, so this runs unchanged on the Colab free tier
and locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

try:  # works when imported as part of the ``training`` package
    from training.energy_model import EnergyModel, rasterize_world
    from training.environment import GridWorld
    from training.trajectory import Trajectory
except ImportError:  # falls back when run directly as ``python sampler.py``
    from energy_model import EnergyModel, rasterize_world
    from environment import GridWorld
    from trajectory import Trajectory

__all__ = [
    "LangevinConfig",
    "SampleResult",
    "random_initial_trajectory",
    "LangevinSampler",
]


@dataclass
class LangevinConfig:
    """Knobs for one Langevin sampling run.

    The defaults are the values tuned in Task 4.2 against the score-matching
    model: a moderate step with light exploration noise that *anneals to zero*
    over the run, so chaotic paths descend the corridor early and then settle
    cleanly into a valley instead of jittering interior points across an obstacle
    boundary. They are a sound default for the shipped weights; the tuning
    sweep that chose them lives in :mod:`tune_sampler`.

    The schedule
    ------------

    ``step_size`` and ``noise_scale`` are the values at the *first* step. If
    ``final_step_size`` / ``final_noise_scale`` are given, the live values
    interpolate linearly to them over ``n_steps`` (a settling schedule); if left
    ``None`` they stay constant for the whole run. Annealing the noise to ``0.0``
    is what lets a descended path relax exactly onto the valley floor.

    Attributes
    ----------
    n_steps:
        Number of Langevin updates to apply.
    step_size:
        The ``lr`` multiplying ``grad(E)`` at the first step — how far each
        downhill step moves.
    noise_scale:
        Standard deviation of the Gaussian noise added at the first step (in
        world units). Set to ``0.0`` for plain gradient descent (no exploration).
    final_step_size:
        Step size at the last step; ``None`` keeps ``step_size`` constant.
    final_noise_scale:
        Noise scale at the last step; ``None`` keeps ``noise_scale`` constant.
        Set the pair (``noise_scale`` > 0, ``final_noise_scale`` = 0.0) to
        explore early and settle late.
    grad_clip:
        If set, the per-sample gradient is rescaled so its L2 norm never exceeds
        this value. The energy can have rare gradient spikes (the same ones the
        training loop clips); bounding them here stops a single step from
        hurling the path off the landscape. ``None`` disables clipping.
    pin_endpoints:
        Hold the first/last points at the world's start/goal every step, so only
        the interior of the path is free to move.
    clamp_to_world:
        Clip every point back into ``[0, size] x [0, size]`` each step.
    record_history:
        Keep the full path at every step (for the Phase 5 descent animation). If
        ``False``, only the energy trace is retained, which is cheaper in memory.
    seed:
        Optional seed for the noise/initialisation RNG, for reproducible runs.
    """

    n_steps: int = 300
    step_size: float = 2e-2
    noise_scale: float = 6e-3
    final_step_size: Optional[float] = 5e-3
    final_noise_scale: Optional[float] = 0.0
    grad_clip: Optional[float] = 1.0
    pin_endpoints: bool = True
    clamp_to_world: bool = True
    record_history: bool = True
    seed: Optional[int] = None

    def step_size_at(self, step: int) -> float:
        """Step size at iteration ``step`` (linear schedule, or constant)."""
        if self.final_step_size is None or self.n_steps <= 1:
            return self.step_size
        frac = step / (self.n_steps - 1)
        return self.step_size + (self.final_step_size - self.step_size) * frac

    def noise_at(self, step: int) -> float:
        """Noise scale at iteration ``step`` (linear schedule, or constant)."""
        if self.final_noise_scale is None or self.n_steps <= 1:
            return self.noise_scale
        frac = step / (self.n_steps - 1)
        return self.noise_scale + (self.final_noise_scale - self.noise_scale) * frac


@dataclass
class SampleResult:
    """The output of a sampling run.

    Attributes
    ----------
    trajectories:
        The final sampled paths, one :class:`Trajectory` per seed.
    energy_trace:
        ``(n_steps + 1, n_samples)`` array of the energy of every path at every
        step, including the initial state (row 0) and the final state (last
        row). The headline evidence that descent worked: each column should fall.
    history:
        ``(n_steps + 1, n_samples, N, 2)`` array of the full path at every step,
        or ``None`` when ``record_history`` was disabled. Row 0 is the random
        initialisation; the last row matches :pyattr:`trajectories`. This is what
        the web app animates as the descent into the valleys.
    """

    trajectories: list[Trajectory]
    energy_trace: np.ndarray
    history: Optional[np.ndarray]

    @property
    def initial_energy(self) -> np.ndarray:
        """Energy of each path before any Langevin step, shape ``(n_samples,)``."""
        return self.energy_trace[0]

    @property
    def final_energy(self) -> np.ndarray:
        """Energy of each path after the final step, shape ``(n_samples,)``."""
        return self.energy_trace[-1]


def random_initial_trajectory(
    world: GridWorld,
    n_points: int,
    rng: np.random.Generator,
    *,
    pin_endpoints: bool = True,
) -> np.ndarray:
    """Draw one chaotic starting path: ``n_points`` uniform points in the world.

    The interior points are sampled uniformly across the whole world rectangle,
    so the path is genuinely disordered — exactly the "chaos" the landscape is
    meant to resolve into a route. When ``pin_endpoints`` is set (the default,
    matching the sampler), the first and last points are placed on the world's
    start and goal so the path begins life with the right endpoints.

    Returns a writable ``(n_points, 2)`` float array.
    """
    if n_points < 2:
        raise ValueError(f"A trajectory needs at least 2 points; got {n_points}.")
    points = rng.uniform(0.0, world.size, size=(n_points, 2))
    if pin_endpoints:
        points[0] = np.asarray(world.start, dtype=float)
        points[-1] = np.asarray(world.goal, dtype=float)
    return points


class LangevinSampler:
    """Generate trajectories by noisy gradient descent on a trained energy.

    Construct it with a trained :class:`EnergyModel` and a :class:`GridWorld`;
    the scene is rasterised once (it is the fixed conditioning context — only the
    trajectory carries gradient) and reused for every step and every seed. Call
    :meth:`sample` to run one or many descents.

    Parameters
    ----------
    model:
        A trained energy model. Put it in ``eval`` mode (the constructor does).
    world:
        The scene to sample routes through.
    resolution:
        Rasterisation resolution for the scene; must match what the model was
        trained at (the project default is 64).
    device:
        Where to run. Defaults to the model's current device.
    """

    def __init__(
        self,
        model: EnergyModel,
        world: GridWorld,
        *,
        resolution: int = 64,
        device: Optional[torch.device] = None,
    ):
        self.model = model.eval()
        self.world = world
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )

        # Rasterise the scene once: (1, C, H, W). Held with no grad — the map is
        # fixed conditioning, only the trajectory is differentiated.
        raster = rasterize_world(world, resolution=resolution)
        self._map = torch.from_numpy(raster).to(
            device=self.device, dtype=torch.float32
        ).unsqueeze(0)

        self._start = torch.tensor(
            np.asarray(world.start, dtype=np.float32), device=self.device
        )
        self._goal = torch.tensor(
            np.asarray(world.goal, dtype=np.float32), device=self.device
        )

    def sample(
        self,
        n_samples: int = 1,
        n_points: int = 48,
        *,
        config: Optional[LangevinConfig] = None,
        rng: Optional[np.random.Generator] = None,
        init: Optional[np.ndarray] = None,
    ) -> SampleResult:
        """Run ``n_samples`` Langevin descents in parallel and return the result.

        Each sample starts from its own random trajectory (or from ``init`` if
        supplied) and follows ``path <- path - step_size * grad(E) + noise`` for
        ``config.n_steps`` steps, with endpoints pinned and points clamped to the
        world. Running the seeds as one batch is what makes Task 4.3's
        multimodality check cheap: distinct seeds, identical scene, one forward
        pass per step.

        Parameters
        ----------
        n_samples:
            How many independent paths (seeds) to evolve at once.
        n_points:
            Length ``N`` of each trajectory. Ignored when ``init`` is given.
        config:
            Sampling hyper-parameters; a default :class:`LangevinConfig` is used
            if omitted.
        rng:
            Source of randomness for initialisation and noise. If omitted, one is
            built from ``config.seed``.
        init:
            Optional explicit starting batch, shape ``(n_samples, N, 2)``. Useful
            for reproducing a specific run or resuming from a known state.
        """
        config = config or LangevinConfig()
        if rng is None:
            rng = np.random.default_rng(config.seed)
        if n_samples < 1:
            raise ValueError(f"n_samples must be positive; got {n_samples}.")

        # Build the starting batch (n_samples, N, 2).
        if init is not None:
            init = np.asarray(init, dtype=np.float32)
            if init.ndim != 3 or init.shape[0] != n_samples or init.shape[2] != 2:
                raise ValueError(
                    f"init must have shape (n_samples, N, 2)=({n_samples}, N, 2); "
                    f"got {init.shape}."
                )
            start_batch = init
        else:
            start_batch = np.stack(
                [
                    random_initial_trajectory(
                        self.world, n_points, rng, pin_endpoints=config.pin_endpoints
                    )
                    for _ in range(n_samples)
                ],
                axis=0,
            ).astype(np.float32)

        path = torch.from_numpy(start_batch).to(self.device)
        maps = self._map.expand(n_samples, -1, -1, -1)

        # A torch generator seeded from the same numpy RNG keeps the noise stream
        # reproducible alongside the (numpy) initialisation.
        noise_gen = torch.Generator(device=self.device)
        noise_gen.manual_seed(int(rng.integers(0, 2**63 - 1)))

        n_steps = config.n_steps
        energy_trace = np.empty((n_steps + 1, n_samples), dtype=np.float32)
        history = (
            np.empty((n_steps + 1, n_samples, path.shape[1], 2), dtype=np.float32)
            if config.record_history
            else None
        )

        def _record(step: int, current: Tensor, energy: Tensor) -> None:
            energy_trace[step] = energy.detach().cpu().numpy()
            if history is not None:
                history[step] = current.detach().cpu().numpy()

        for step in range(n_steps):
            path = path.detach().requires_grad_(True)
            # The model is in eval mode, but cuDNN's fused LSTM backward refuses
            # to run unless the RNN is in training mode ("cudnn RNN backward can
            # only be called in training mode" on CUDA). Disable cuDNN so the
            # native RNN backward computes the Langevin gradient instead. No-op on
            # CPU, and the sampling math is unchanged. (Same cuDNN RNN limitation
            # fixed in train.py.)
            with torch.backends.cudnn.flags(enabled=False):
                energy = self.model(maps, path)
                (grad,) = torch.autograd.grad(energy.sum(), path)
            _record(step, path, energy)

            with torch.no_grad():
                grad = self._maybe_clip(grad, config.grad_clip)
                step_size = config.step_size_at(step)
                noise_scale = config.noise_at(step)
                noise = (
                    noise_scale
                    * torch.randn(
                        path.shape, generator=noise_gen, device=self.device
                    )
                    if noise_scale > 0.0
                    else torch.zeros_like(path)
                )
                path = path - step_size * grad + noise
                path = self._apply_constraints(path, config)

        # Final state after the last update.
        with torch.no_grad():
            final_energy = self.model(maps, path)
        _record(n_steps, path, final_energy)

        final = path.detach().cpu().numpy()
        trajectories = [Trajectory(final[i]) for i in range(n_samples)]
        return SampleResult(
            trajectories=trajectories, energy_trace=energy_trace, history=history
        )

    # --- internals ----------------------------------------------------------

    @staticmethod
    def _maybe_clip(grad: Tensor, grad_clip: Optional[float]) -> Tensor:
        """Rescale each sample's gradient so its L2 norm is at most ``grad_clip``.

        Operates per sample (over the ``N x 2`` coordinates) so one path's spike
        cannot affect another's step. A no-op when ``grad_clip`` is ``None``.
        """
        if grad_clip is None:
            return grad
        flat = grad.reshape(grad.shape[0], -1)
        norm = flat.norm(dim=1, keepdim=True)
        scale = (grad_clip / (norm + 1e-12)).clamp(max=1.0)
        return (flat * scale).reshape(grad.shape)

    def _apply_constraints(self, path: Tensor, config: LangevinConfig) -> Tensor:
        """Re-pin endpoints and clamp to the world, per the config."""
        if config.clamp_to_world:
            path = path.clamp(0.0, self.world.size)
        if config.pin_endpoints:
            # Clamp may have nudged the endpoints; pin after clamping so they
            # land exactly on start/goal.
            path = path.clone()
            path[:, 0, :] = self._start
            path[:, -1, :] = self._goal
        return path


def _demo() -> None:
    """Self-check: descend random paths on the trained model and show energy drop.

    Loads the Phase 3 checkpoint if present (``exports/energy_model.pt``);
    otherwise trains a tiny model on the fly so the script always runs. Confirms
    the core 4.1 contract: starting from chaos, the Langevin loop lowers the
    energy and returns well-formed trajectories with the right endpoints.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint = repo_root / "exports" / "energy_model.pt"

    rng = np.random.default_rng(0)
    if checkpoint.exists():
        from train import load_checkpoint  # local import; keeps module import light

        model, _ = load_checkpoint(checkpoint)
        # Reuse a world the trained model has the right scale for.
        world = GridWorld.random(rng, n_obstacles=(3, 5))
        source = f"checkpoint {checkpoint.name}"
    else:
        from train import TrainConfig, train  # local import for the fallback

        world = GridWorld.random(rng, n_obstacles=(3, 5))
        cfg = TrainConfig(n_scenes=12, epochs=15, log_every=0)
        model, _ = train(cfg)
        source = "freshly trained tiny model (no checkpoint found)"

    sampler = LangevinSampler(model, world)
    config = LangevinConfig(n_steps=200, seed=0)
    result = sampler.sample(n_samples=6, n_points=48, config=config)

    e0 = result.initial_energy
    e1 = result.final_energy
    dropped = int((e1 < e0).sum())

    print("Langevin sampler (Task 4.1)")
    print(f"  model            : {source}")
    print(f"  scene obstacles  : {len(world.obstacles)}")
    print(
        f"  config           : steps={config.n_steps} step_size={config.step_size} "
        f"noise={config.noise_scale} grad_clip={config.grad_clip}"
    )
    print(f"  samples          : {len(result.trajectories)}")
    print(f"  energy trace     : {result.energy_trace.shape}")
    if result.history is not None:
        print(f"  descent history  : {result.history.shape}")
    print(f"  mean energy      : {e0.mean():+.3f} -> {e1.mean():+.3f}")
    print(f"  samples lowered  : {dropped}/{len(result.trajectories)}")

    # Core 4.1 contract: the loop runs, produces valid containers with the right
    # endpoints, and the noisy descent lowers the energy on average.
    assert result.energy_trace.shape == (config.n_steps + 1, 6)
    assert result.history is not None
    assert result.history.shape == (config.n_steps + 1, 6, 48, 2)
    for traj in result.trajectories:
        assert traj.num_points == 48
        assert np.allclose(traj.start, world.start, atol=1e-5), "start not pinned"
        assert np.allclose(traj.goal, world.goal, atol=1e-5), "goal not pinned"
        assert bool(world.in_bounds(traj.points).all()), "path left the world"
    assert e1.mean() < e0.mean(), "Langevin descent did not lower the mean energy"
    print("  self-check passed: chaos descends into lower energy, endpoints held.")


if __name__ == "__main__":
    _demo()
