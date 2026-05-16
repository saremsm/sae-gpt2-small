from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal, NamedTuple


@dataclass(frozen=True)
class SAEConfig:
    d_model: int
    n_features: int
    l1_coefficient: float = 5e-3
    # Optimizer defaults are the frontier recipe: AdamW at 4e-4 after a warmup of
    # 2% of the run.
    lr: float = 4e-4
    warmup_steps: int = 976
    normalize_decoder: bool = True
    # The SAE owns its input contract: raw residuals are multiplied by one.
    normalize_input: bool = True
    # Encoder nonlinearity. "relu": h = relu(pre), sparsity from the L1 penalty
    # (l1_coefficient).
    activation: Literal["relu", "topk"] = "relu"
    # Number of latents kept per token under "topk".
    k: int | None = None
    # AuxK (Gao et al. 2024, topk only, off by default): each training step, take
    # the top-aux_k pre-activations among currently DEAD features
    # (feature_activation_counts == 0 in the current window)
    aux_k: int = 0
    aux_coeff: float = 0.0
    # AdamW betas the training loop uses (train_sae passes them to the optimizer).
    # Default (0.9, 0.99) is the frontier-sweep recipe; the README's history rows
    # used torch's (0.9, 0.999).
    adam_betas: tuple[float, float] = (0.9, 0.99)

    def __post_init__(self) -> None:
        betas = tuple(float(b) for b in self.adam_betas)
        if len(betas) != 2 or not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(
                f"adam_betas must be two floats in [0, 1), got {self.adam_betas!r}"
            )
        object.__setattr__(self, "adam_betas", betas)
        if self.activation not in ("relu", "topk"):
            raise ValueError(
                f"activation must be 'relu' or 'topk', got {self.activation!r}"
            )
        if self.activation == "topk":
            if self.k is None:
                raise ValueError("activation='topk' requires k")
            if not (1 <= self.k <= self.n_features):
                raise ValueError(
                    f"k must be in [1, n_features={self.n_features}], "
                    f"got {self.k}"
                )
            if self.l1_coefficient != 0.0:
                warnings.warn(
                    f"activation='topk' ignores the L1 penalty; forcing "
                    f"l1_coefficient={self.l1_coefficient} to 0.0 (pass "
                    f"l1_coefficient=0.0 to silence this).",
                    stacklevel=3,
                )
                object.__setattr__(self, "l1_coefficient", 0.0)
        elif self.k is not None:
            raise ValueError(
                f"k={self.k} is only meaningful with activation='topk'"
            )
        if self.aux_k < 0 or self.aux_k > self.n_features:
            raise ValueError(
                f"aux_k must be in [0, n_features={self.n_features}], "
                f"got {self.aux_k}"
            )
        if self.aux_k > 0 and self.activation != "topk":
            raise ValueError("aux_k > 0 requires activation='topk'")
        if self.aux_coeff < 0.0:
            raise ValueError(f"aux_coeff must be >= 0, got {self.aux_coeff}")

    @property
    def expansion_factor(self) -> float:
        return self.n_features / self.d_model


class SAEOutput(NamedTuple):
    # Reconstruction in scaled space (x * input_scale)
    recon_scaled: torch.Tensor
    # postprocess(recon_scaled): raw residual space, spliceable back into the
    recon_raw: torch.Tensor
    # Feature activations: post-ReLU ("relu") or the ReLU'd top-k pre-activations.
    h: torch.Tensor
    # reconstruction_loss + l1_coefficient * sparsity_loss + aux_coeff * aux_loss.
    loss: torch.Tensor
    # MSE in scaled space.
    reconstruction_loss: torch.Tensor
    # sum_i |h_i| per token, averaged over the batch.
    sparsity_loss: torch.Tensor
    # AuxK MSE(residual, aux_recon) in scaled space, unweighted.
    aux_loss: torch.Tensor
    l0: torch.Tensor
    # Per-token reconstruction error (scaled space, detached) - resampling
    per_token_recon_error: torch.Tensor


class SparseAutoencoder(nn.Module):
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        d, f = config.d_model, config.n_features

        W_dec = torch.randn(f, d)
        if config.normalize_decoder:
            W_dec = W_dec / W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)

        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(f))
        self.b_dec = nn.Parameter(torch.zeros(d))

        self.register_buffer(
            "feature_activation_counts",
            torch.zeros(f, dtype=torch.long),
        )
        # Dataset-wide input scale (scalar).
        self.register_buffer(
            "input_scale",
            torch.ones((), dtype=torch.float32),
        )

    # Input contract
    def set_input_scale(self, value: float) -> None:
        """Set input_scale directly (e.g. to a previously calibrated value)."""
        value = float(value)
        if not (value > 0.0 and math.isfinite(value)):
            raise ValueError(
                f"input_scale must be positive and finite, got {value}"
            )
        with torch.no_grad():
            self.input_scale.fill_(value)

    def set_input_scale_from_activations(self, x: torch.Tensor) -> None:
        """Calibrate input_scale = sqrt(d_model) / mean_i ||x_i||_2 on a sample of
        raw activations, shape (n, d_model)."""
        with torch.no_grad():
            mean_norm = x.detach().float().norm(dim=-1).mean().item()
        if not (mean_norm > 0.0 and math.isfinite(mean_norm)):
            raise ValueError(
                f"cannot calibrate input_scale: mean activation norm is "
                f"{mean_norm}"
            )
        self.set_input_scale((self.config.d_model ** 0.5) / mean_norm)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Raw -> scaled space: x * input_scale."""
        if not self.config.normalize_input:
            return x
        return x * self.input_scale

    def postprocess(self, x_hat_scaled: torch.Tensor) -> torch.Tensor:
        """Scaled -> raw space: x_hat_scaled / input_scale."""
        if not self.config.normalize_input:
            return x_hat_scaled
        return x_hat_scaled / self.input_scale

    # Decoder constraint
    def normalize_decoder(self) -> None:
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True)
            self.W_dec.data /= norms.clamp(min=1e-8)

    def project_decoder_grad(self) -> None:
        if self.W_dec.grad is None:
            return
        with torch.no_grad():
            W = self.W_dec.data
            grad = self.W_dec.grad
            norms_sq = (W * W).sum(dim=1, keepdim=True).clamp(min=1e-8)
            inner = (grad * W).sum(dim=1, keepdim=True)
            grad.sub_((inner / norms_sq) * W)

    # Forward paths
    @property
    def activation(self) -> str:
        return self.config.activation

    @property
    def k(self) -> int | None:
        return self.config.k

    def pre_activations(self, x_scaled: torch.Tensor) -> torch.Tensor:
        """Encoder pre-activations (x_scaled - b_dec) @ W_enc + b_enc for inputs."""
        return (x_scaled - self.b_dec) @ self.W_enc + self.b_enc

    def apply_activation(self, pre: torch.Tensor) -> torch.Tensor:
        """Pre-activations -> feature activations h, same shape. Gradient flows only
        through the kept latents - the scatter of the top-k values."""
        if self.config.activation == "relu":
            return F.relu(pre)
        top_vals, top_idx = pre.topk(self.config.k, dim=-1)
        h = torch.zeros_like(pre)
        return h.scatter(-1, top_idx, F.relu(top_vals))

    def _encode_preprocessed(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_activation(self.pre_activations(x))

    def encode(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Feature activations for RAW residuals."""
        return self._encode_preprocessed(self.preprocess(x_raw))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Reconstruction in SCALED space; postprocess for raw space."""
        return h @ self.W_dec + self.b_dec

    def aux_loss(
        self,
        pre: torch.Tensor,
        x_scaled: torch.Tensor,
        recon_scaled: torch.Tensor,
    ) -> torch.Tensor:
        """AuxK term (Gao et al. 2024): MSE between the detached residual (x_scaled
        - recon_scaled) and its reconstruction from the top-aux_k pre-activations
        among currently dead features."""
        dead = self.feature_activation_counts == 0
        residual = (x_scaled - recon_scaled).detach()
        # Only dead features may be selected: alive ones go to -inf.
        pre_dead = pre.masked_fill(~dead, float("-inf"))
        top_vals, top_idx = pre_dead.topk(self.config.aux_k, dim=-1)
        h_aux = torch.zeros_like(pre).scatter(-1, top_idx, F.relu(top_vals))
        aux_recon = h_aux @ self.W_dec
        return F.mse_loss(aux_recon, residual) * dead.any().to(pre.dtype)

    def forward(self, x_raw: torch.Tensor) -> SAEOutput:
        """Raw residuals in. Dead-feature bookkeeping is the same under both:
        feature_activation_counts counts h > 0 per feature."""
        x = self.preprocess(x_raw)
        pre = self.pre_activations(x)
        h = self.apply_activation(pre)
        recon_scaled = self.decode(h)

        reconstruction_loss = F.mse_loss(recon_scaled, x)
        sparsity_loss = h.abs().sum(dim=-1).mean()
        loss = reconstruction_loss + self.config.l1_coefficient * sparsity_loss

        # Dead mask from the counts as they stand BEFORE this batch is added.
        if self.config.aux_k > 0 and self.training:
            aux_loss = self.aux_loss(pre, x, recon_scaled)
            loss = loss + self.config.aux_coeff * aux_loss
        else:
            aux_loss = loss.new_zeros(())

        l0 = (h > 0).float().sum(dim=-1).mean()
        per_token_recon_error = (
            (recon_scaled - x).pow(2).mean(dim=-1).detach()
        )

        if self.training:
            with torch.no_grad():
                self.feature_activation_counts += (h > 0).long().sum(dim=0)

        return SAEOutput(
            recon_scaled=recon_scaled,
            recon_raw=self.postprocess(recon_scaled),
            h=h,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            aux_loss=aux_loss,
            l0=l0,
            per_token_recon_error=per_token_recon_error,
        )

    # Checkpoint compatibility
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """A state_dict without `input_scale` (written before the scalar input scale
        existed) loads with input_scale = 1.0 and a warning instead of a missing-
        key error."""
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        key = prefix + "input_scale"
        if key in missing_keys:
            missing_keys.remove(key)
            warnings.warn(
                f"state_dict has no '{key}' (checkpoint predates the "
                f"dataset-wide input scale); defaulting input_scale to 1.0. "
                f"Recalibrate with set_input_scale_from_activations, or "
                f"set_input_scale, if the model was trained on scaled inputs.",
                stacklevel=2,
            )
            with torch.no_grad():
                self.input_scale.fill_(1.0)

    # Dead-feature handling

    def get_dead_features(self, threshold: int = 0) -> torch.Tensor:
        return (self.feature_activation_counts <= threshold).nonzero(
            as_tuple=True
        )[0]

    def resample_dead_features(
        self,
        dead_feature_indices: torch.Tensor,
        activations: torch.Tensor,
        errors: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Reinitialize dead features toward high-reconstruction-error examples
        (sampled proportional to error, not uniformly)."""
        if len(dead_feature_indices) == 0:
            return

        activations = self.preprocess(activations)
        n_dead = len(dead_feature_indices)

        error_weights = errors / errors.sum().clamp(min=1e-8)
        sampled_indices = torch.multinomial(
            error_weights,
            num_samples=n_dead,
            replacement=True,
        )
        sampled_activations = activations[sampled_indices]

        with torch.no_grad():
            directions = sampled_activations - self.b_dec
            directions = directions / (
                directions.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            )

            self.W_enc.data[:, dead_feature_indices] = directions.T * 0.2
            self.b_enc.data[dead_feature_indices] = 0.0
            self.W_dec.data[dead_feature_indices] = directions

            self.feature_activation_counts.zero_()

            if optimizer is not None:
                self._zero_optim_state(optimizer, dead_feature_indices)

    def _zero_optim_state(
        self,
        optimizer: torch.optim.Optimizer,
        dead: torch.Tensor,
    ) -> None:
        targets = [
            (self.W_enc, 1),
            (self.b_enc, 0),
            (self.W_dec, 0),
        ]
        for param, axis in targets:
            state = optimizer.state.get(param, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key not in state:
                    continue
                if axis == 0:
                    state[key][dead] = 0.0
                else:
                    state[key][:, dead] = 0.0
