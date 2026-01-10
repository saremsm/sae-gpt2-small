from __future__ import annotations

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
    warmup_steps: int = 1000
    normalize_decoder: bool = True

    @property
    def expansion_factor(self) -> float:
        return self.n_features / self.d_model


class SAEOutput(NamedTuple):
    reconstructed: torch.Tensor
    features: torch.Tensor
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    l0: torch.Tensor


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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        pre_activations = x_centered @ self.W_enc + self.b_enc
        return F.relu(pre_activations)

    def encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: normalize raw residuals to norm sqrt(d_model), then encode."""
        x = (
            x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            * (self.config.d_model ** 0.5)
        )
        return self.encode(x)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        h = self.encode(x)
        x_reconstructed = self.decode(h)

        reconstruction_loss = F.mse_loss(x_reconstructed, x)
        sparsity_loss = h.abs().sum(dim=-1).mean()
        loss = reconstruction_loss + self.config.l1_coefficient * sparsity_loss

        l0 = (h > 0).float().sum(dim=-1).mean()

        if self.training:
            with torch.no_grad():
                self.feature_activation_counts += (h > 0).long().sum(dim=0)

        return SAEOutput(
            reconstructed=x_reconstructed,
            features=h,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            l0=l0,
        )

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
