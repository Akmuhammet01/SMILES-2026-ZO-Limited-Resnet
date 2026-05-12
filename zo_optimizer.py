"""
zo_optimizer.py — Zero-order optimizer (SPSA + multi-query + Adam).

Final solution overview
-----------------------
The skeleton's per-parameter 2-point central-difference estimator costs
``2 * numel`` forward passes per step, which is prohibitive even for a single
classification head (51,300 parameters → 102,600 forward passes per step).

This implementation replaces it with **SPSA** (Simultaneous Perturbation
Stochastic Approximation):

    g ≈ (f(θ + ε·u) - f(θ - ε·u)) / (2ε) · u,   u_i ~ Rademacher{-1, +1}

SPSA perturbs *all* selected parameters simultaneously, so only 2 forward
passes are required per gradient sample, independent of model size.

To reduce variance (SPSA's only weakness for high-dimensional problems), we
average ``num_queries`` independent SPSA samples per ``.step()`` on the *same*
mini-batch. Each step therefore costs ``1 + 2 * num_queries`` forward passes
(the leading ``1`` is the API-mandated ``loss_before`` call).

The update rule is **Adam** with bias correction, which adapts the
per-parameter step size and behaves robustly under noisy ZO gradient
estimates — when variance dominates, the update degenerates to a sign-like
step which is empirically very effective for SPSA.

Only ``fc.weight`` and ``fc.bias`` are tuned. With a 32-step budget any
deeper backbone tuning has too few gradient samples to converge through SPSA
noise; ImageNet features for the 32×32 → 224 upscaled CIFAR100 images are
strong enough that a well-tuned linear head dominates the achievable score.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Gradient-free optimizer using SPSA with multi-query averaging + Adam.

    Args:
        model:             The ``nn.Module`` to optimize.
        lr:                Adam learning rate.
        eps:               SPSA perturbation magnitude.
        perturbation_mode: ``"rademacher"`` (default, recommended for SPSA),
                           ``"gaussian"``, or ``"uniform"``.
        num_queries:       Number of independent SPSA samples averaged per
                           ``.step()`` on the same mini-batch. Higher reduces
                           variance at the cost of more forward passes.
        beta1:             Adam first-moment decay.
        beta2:             Adam second-moment decay.
        adam_eps:          Adam numerical stabiliser.
        weight_decay:      Decoupled weight decay (0 disables).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-3,
        eps: float = 1e-3,
        perturbation_mode: str = "rademacher",
        num_queries: int = 4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.num_queries = int(num_queries)
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay

        if perturbation_mode not in ("gaussian", "uniform", "rademacher"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian', 'uniform', or "
                f"'rademacher', got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode

        if self.num_queries < 1:
            raise ValueError("num_queries must be >= 1")

        # Tune only the freshly initialised classification head. The ImageNet
        # backbone already supplies strong features after the Resize(224)
        # upscaling; with a 32-step budget there is not enough signal to push
        # SPSA through the much higher-dimensional backbone layers.
        self.layer_names: list[str] = ["fc.weight", "fc.bias"]

        # Adam state — lazily allocated on first update so that we always
        # match each parameter's device/dtype.
        self._step_count: int = 0
        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        """Sample an SPSA perturbation vector with the same shape as ``param``.

        For ``"rademacher"`` (default), each entry is independently ±1, which
        is the standard SPSA choice: it has bounded inverse moments and yields
        an unbiased element-wise gradient estimator. The Gaussian and uniform
        modes are kept for compatibility / experimentation and are *not*
        re-normalised to unit length — for SPSA the per-coordinate scale is
        what matters, not the global norm.
        """
        if self.perturbation_mode == "rademacher":
            return (
                torch.randint(
                    0, 2, param.shape, device=param.device, dtype=torch.int8
                ).to(param.dtype)
                * 2.0
                - 1.0
            )
        if self.perturbation_mode == "gaussian":
            return torch.randn_like(param)
        # uniform
        return torch.rand_like(param) * 2.0 - 1.0

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """Multi-query SPSA pseudo-gradient on the current mini-batch.

        For each of ``num_queries`` independent samples:
            1. Draw u with the same shape as every active parameter.
            2. Perturb every active parameter to θ + ε·u, evaluate f_plus.
            3. Perturb every active parameter to θ - ε·u, evaluate f_minus.
            4. Restore every active parameter to its original value.
            5. Accumulate (f_plus - f_minus) / (2ε) * u into the running sum.

        The accumulator is averaged at the end, yielding a variance-reduced
        SPSA estimate. Each sample costs exactly 2 forward passes regardless
        of the number or size of active parameters.
        """
        grads: dict[str, torch.Tensor] = {
            name: torch.zeros_like(p) for name, p in params.items()
        }

        with torch.no_grad():
            for _ in range(self.num_queries):
                directions = {
                    name: self._sample_direction(p) for name, p in params.items()
                }

                # θ ← θ + ε·u
                for name, p in params.items():
                    p.data.add_(directions[name], alpha=self.eps)
                f_plus = loss_fn()

                # θ ← θ - ε·u (subtract 2ε·u from the perturbed state)
                for name, p in params.items():
                    p.data.add_(directions[name], alpha=-2.0 * self.eps)
                f_minus = loss_fn()

                # θ ← original
                for name, p in params.items():
                    p.data.add_(directions[name], alpha=self.eps)

                scale = (f_plus - f_minus) / (2.0 * self.eps)
                for name in params:
                    grads[name].add_(directions[name], alpha=scale)

            inv_q = 1.0 / self.num_queries
            for name in grads:
                grads[name].mul_(inv_q)

        return grads

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Adam update with bias correction.

        Under heavy gradient noise the Adam update degenerates to a
        ``lr * sign(m̂)``-like step, which is exactly the regime SPSA
        produces — empirically a much better fit than plain SGD here.
        """
        self._step_count += 1
        bc1 = 1.0 - self.beta1 ** self._step_count
        bc2 = 1.0 - self.beta2 ** self._step_count

        with torch.no_grad():
            for name, p in params.items():
                g = grads[name]

                if self.weight_decay > 0.0:
                    g = g.add(p.data, alpha=self.weight_decay)

                if name not in self._m:
                    self._m[name] = torch.zeros_like(p)
                    self._v[name] = torch.zeros_like(p)

                m = self._m[name]
                v = self._v[name]
                m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

                m_hat = m / bc1
                v_hat = v / bc2
                p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.adam_eps), value=-self.lr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Perform one SPSA + Adam optimisation step.

        Calls ``loss_fn`` ``1 + 2 * num_queries`` times in total — once to
        report the pre-update loss (also used as a progress signal in
        ``validate.py``) and twice per SPSA sample for the central-difference
        estimate. ``validate.py`` guarantees that every call within a single
        ``.step()`` uses the same fixed mini-batch.
        """
        params = self._active_params()

        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        return float(loss_before)
