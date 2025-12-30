import torch
import pytest

from sparse_autoencoder import SparseAutoencoder, SAEConfig

D_MODEL = 64
N_FEATURES = 256
BATCH_SIZE = 16


@pytest.fixture(scope="class")
def sae_no_l1() -> SparseAutoencoder:
    """l1=0 so loss == recon. class-scoped: forward-pass tests don't mutate the model."""
    config = SAEConfig(
        d_model=D_MODEL,
        n_features=N_FEATURES,
        l1_coefficient=0.0,
        lr=2e-4,
        warmup_steps=10,
    )
    return SparseAutoencoder(config)


class TestForwardPass:
    def test_reconstructed_shape(self, sae_no_l1):
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae_no_l1(x)
        assert out.reconstructed.shape == (BATCH_SIZE, D_MODEL)

    def test_features_shape(self, sae_no_l1):
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae_no_l1(x)
        assert out.features.shape == (BATCH_SIZE, N_FEATURES)

    def test_loss_is_scalar(self, sae_no_l1):
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae_no_l1(x)
        assert out.loss.shape == torch.Size([])

    def test_loss_is_finite(self, sae_no_l1):
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae_no_l1(x)
        assert out.loss.isfinite().item()

    def test_features_non_negative(self, sae_no_l1):
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae_no_l1(x)
        assert (out.features >= 0).all().item()

    def test_l0_finite_and_non_negative(self, sae_no_l1):
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae_no_l1(x)
        assert out.l0.isfinite().item() and out.l0.item() >= 0
