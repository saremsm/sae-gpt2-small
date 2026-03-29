from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class SAEConfig:
    d_model: int
    n_features: int
    l1_coefficient: float = 5e-3
    lr: float = 2e-4
    warmup_steps: int = 100
    normalize_decoder: bool = True
    # The SAE owns its input contract: raw residuals are multiplied by one.
    normalize_input: bool = True

    @property
    def expansion_factor(self) -> float:
        return self.n_features / self.d_model


class SAEOutput(NamedTuple):
    # Reconstruction in scaled space (x * input_scale)
    recon_scaled: torch.Tensor
    # postprocess(recon_scaled): raw residual space, spliceable back into the
    recon_raw: torch.Tensor
    # Feature activations (post-ReLU).
    h: torch.Tensor
    loss: torch.Tensor
    # MSE in scaled space.
    reconstruction_loss: torch.Tensor
    sparsity_loss: torch.Tensor
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
    def _encode_preprocessed(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        pre_activations = x_centered @ self.W_enc + self.b_enc
        return F.relu(pre_activations)

    def encode(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Feature activations for RAW residuals."""
        return self._encode_preprocessed(self.preprocess(x_raw))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Reconstruction in SCALED space; postprocess for raw space."""
        return h @ self.W_dec + self.b_dec

    def forward(self, x_raw: torch.Tensor) -> SAEOutput:
        """raw residuals in; loss/MSE in scaled space, recon_raw mapped back to raw."""
        x = self.preprocess(x_raw)
        h = self._encode_preprocessed(x)
        recon_scaled = self.decode(h)

        reconstruction_loss = F.mse_loss(recon_scaled, x)
        sparsity_loss = h.abs().sum(dim=-1).mean()
        loss = reconstruction_loss + self.config.l1_coefficient * sparsity_loss

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
