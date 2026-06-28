"""Gradient-flow verification for the energy network (Task 2.3).

Langevin sampling (Phase 4) generates trajectories by walking *downhill* in the
energy, following ``grad(E)`` with respect to the path itself. That only works
if the energy is genuinely differentiable in its trajectory input and the
gradients are correct, finite, and actually informative. This script confirms
all of that with dummy forward/backward passes:

- **Gradients flow.** ``E.backward()`` populates ``trajectory.grad`` with the
  right shape ``(B, N, 2)``, all finite and not identically zero.
- **Gradients are correct.** The autograd gradient matches a central
  finite-difference estimate (run in double precision) to tight tolerance.
- **Per-sample isolation.** In a batch, the energy of sample ``i`` depends only
  on trajectory ``i`` — its gradient w.r.t. every other sample is exactly zero.
  This is the payoff of using GroupNorm/LayerNorm instead of BatchNorm, and it
  is what lets the sampler refine one path without leaking through the batch.
- **Descent lowers energy.** A single ``path <- path - lr * grad(E)`` step (the
  Langevin update without noise) reduces the energy, so the gradient points the
  way the sampler will need.

Run ``python verify_gradients.py`` (or ``python -m training.verify_gradients``).
"""

from __future__ import annotations

import torch

try:  # works when imported as part of the ``training`` package
    from training.energy_model import MAP_CHANNELS, EnergyModel
except ImportError:  # falls back when run directly as ``python verify_gradients.py``
    from energy_model import MAP_CHANNELS, EnergyModel


def _verify_grad_flows() -> None:
    """A backward pass fills in a finite, non-zero trajectory gradient."""
    print("Gradient flow")
    torch.manual_seed(0)
    model = EnergyModel().eval()

    b, n, res = 4, 48, 64
    maps = torch.randn(b, MAP_CHANNELS, res, res)
    paths = torch.randn(b, n, 2, requires_grad=True)

    energy = model(maps, paths)
    energy.sum().backward()  # sum so every sample contributes its own gradient

    grad = paths.grad
    assert grad is not None, "no gradient reached the trajectory input"
    assert tuple(grad.shape) == (b, n, 2), f"grad shape {tuple(grad.shape)} != (B,N,2)"
    assert torch.all(torch.isfinite(grad)), "trajectory gradient has NaN/inf"
    assert grad.abs().sum() > 0, "trajectory gradient is identically zero"
    print(f"  ok  grad shape {tuple(grad.shape)}, finite, norm={grad.norm():.4f}")


def _verify_numerical() -> None:
    """Autograd matches central finite differences (double precision)."""
    print("Numerical gradient check")
    torch.manual_seed(1)
    # Small double-precision model + input: finite differences are expensive and
    # only meaningful at high precision.
    model = EnergyModel(map_dim=16, traj_dim=16, hidden=16).double().eval()

    b, n, res = 1, 6, 16
    maps = torch.randn(b, MAP_CHANNELS, res, res, dtype=torch.float64)
    paths = torch.randn(b, n, 2, dtype=torch.float64, requires_grad=True)

    energy = model(maps, paths)
    energy.backward()
    analytic = paths.grad.clone()

    eps = 1e-5
    numeric = torch.zeros_like(paths)
    with torch.no_grad():
        flat = paths.view(-1)
        for i in range(flat.numel()):
            orig = flat[i].item()
            flat[i] = orig + eps
            e_plus = model(maps, paths).item()
            flat[i] = orig - eps
            e_minus = model(maps, paths).item()
            flat[i] = orig
            numeric.view(-1)[i] = (e_plus - e_minus) / (2.0 * eps)

    max_abs = (analytic - numeric).abs().max().item()
    denom = analytic.abs().max().item() + 1e-12
    print(f"  ok  max |autograd - numeric| = {max_abs:.2e} (rel {max_abs / denom:.2e})")
    assert max_abs < 1e-5, f"autograd disagrees with finite differences ({max_abs:.2e})"


def _verify_sample_isolation() -> None:
    """Energy of sample i has zero gradient w.r.t. every other sample."""
    print("Per-sample isolation")
    torch.manual_seed(2)
    model = EnergyModel().eval()

    b, n, res = 4, 32, 64
    maps = torch.randn(b, MAP_CHANNELS, res, res)
    paths = torch.randn(b, n, 2, requires_grad=True)

    energy = model(maps, paths)
    energy[0].backward()  # only sample 0's energy

    grad = paths.grad
    others = grad[1:].abs().max().item()
    own = grad[0].abs().max().item()
    assert own > 0, "sample 0 has no gradient on its own trajectory"
    assert others == 0.0, f"energy[0] leaked gradient to other samples ({others:.2e})"
    print(f"  ok  own grad max={own:.4f}, cross-sample grad max={others:.2e}")


def _verify_descent_lowers_energy() -> None:
    """One noise-free Langevin step reduces the energy along -grad(E)."""
    print("Descent direction")
    torch.manual_seed(3)
    model = EnergyModel().eval()

    b, n, res = 8, 48, 64
    maps = torch.randn(b, MAP_CHANNELS, res, res)
    paths = torch.randn(b, n, 2, requires_grad=True)

    energy_before = model(maps, paths)
    energy_before.sum().backward()

    lr = 1e-2
    with torch.no_grad():
        stepped = paths - lr * paths.grad
    energy_after = model(maps, stepped)

    improved = int((energy_after < energy_before).sum())
    print(
        f"  ok  mean energy {energy_before.mean():.4f} -> {energy_after.mean():.4f}; "
        f"{improved}/{b} samples decreased"
    )
    assert energy_after.mean() < energy_before.mean(), "a -grad step did not lower energy"


def main() -> None:
    _verify_grad_flows()
    _verify_numerical()
    _verify_sample_isolation()
    _verify_descent_lowers_energy()
    print("Gradient verification passed: energy is differentiable in the trajectory.")


if __name__ == "__main__":
    main()
