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
    l1_coefficient: float = 8e-4
    lr: float = 2e-4
    warmup_steps: int = 1000

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

        self.W_enc = nn.Parameter(torch.empty(d, f))
        nn.init.kaiming_uniform_(self.W_enc)
        self.W_dec = nn.Parameter(torch.randn(f, d))
        self.b_enc = nn.Parameter(torch.zeros(f))
        self.b_dec = nn.Parameter(torch.zeros(d))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        pre_activations = x_centered @ self.W_enc + self.b_enc
        return F.relu(pre_activations)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        h = self.encode(x)
        x_reconstructed = self.decode(h)

        reconstruction_loss = F.mse_loss(x_reconstructed, x)
        sparsity_loss = h.abs().sum(dim=-1).mean()
        loss = reconstruction_loss + self.config.l1_coefficient * sparsity_loss

        l0 = (h > 0).float().sum(dim=-1).mean()

        return SAEOutput(
            reconstructed=x_reconstructed,
            features=h,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            l0=l0,
        )
