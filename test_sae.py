import torch
import pytest

from sparse_autoencoder import SparseAutoencoder, SAEConfig

D_MODEL = 64
N_FEATURES = 256
BATCH_SIZE = 16


@pytest.fixture
def sae() -> SparseAutoencoder:
    config = SAEConfig(
        d_model=D_MODEL,
        n_features=N_FEATURES,
        l1_coefficient=8e-4,
        lr=2e-4,
        warmup_steps=10,
        normalize_decoder=True,
    )
    return SparseAutoencoder(config)


@pytest.fixture(scope="class")
def sae_no_l1() -> SparseAutoencoder:
    """l1=0 so loss == recon. class-scoped: forward-pass tests don't mutate the model."""
    config = SAEConfig(
        d_model=D_MODEL,
        n_features=N_FEATURES,
        l1_coefficient=0.0,
        lr=2e-4,
        warmup_steps=10,
        normalize_decoder=True,
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


class TestInit:
    def test_decoder_unit_norm_at_init(self, sae):
        norms = sae.W_dec.norm(dim=1)
        assert torch.allclose(norms, torch.ones(N_FEATURES), atol=1e-5)

    def test_encoder_tied_to_decoder_transpose_at_init(self, sae):
        """tied init: W_enc = W_dec.T at construction time"""
        assert torch.allclose(sae.W_enc, sae.W_dec.T, atol=1e-6)

    def test_no_normalize_when_flag_off(self):
        config = SAEConfig(
            d_model=D_MODEL,
            n_features=N_FEATURES,
            normalize_decoder=False,
        )
        sae = SparseAutoencoder(config)
        norms = sae.W_dec.norm(dim=1)
        assert not torch.allclose(norms, torch.ones(N_FEATURES), atol=1e-2)


class TestNormalizeDecoder:
    def test_unit_norm_after_optimizer_step(self, sae):
        optimizer = torch.optim.AdamW(sae.parameters(), lr=1e-3)
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae(x)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()
        sae.normalize_decoder()

        norms = sae.W_dec.norm(dim=1)
        assert torch.allclose(norms, torch.ones(N_FEATURES), atol=1e-5)


class TestProjectDecoderGrad:
    def test_grad_perpendicular_after_projection(self, sae):
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae(x)
        out.loss.backward()

        # pre-condition: parallel component generally != 0
        W = sae.W_dec.data
        grad = sae.W_dec.grad
        before = (grad * W).sum(dim=1).abs().max().item()
        assert before > 0  # else the test proves nothing

        sae.project_decoder_grad()

        after = (sae.W_dec.grad * sae.W_dec.data).sum(dim=1).abs().max().item()
        assert after < 1e-5

    def test_no_op_when_grad_is_none(self, sae):
        # no backward yet -> no-op, no raise
        sae.project_decoder_grad()
        assert sae.W_dec.grad is None

    def test_norms_preserved_under_sgd(self, sae):
        """projection + SGD preserves row norms (no Adam drift)"""
        sgd = torch.optim.SGD([sae.W_dec], lr=1e-3)
        norms_before = sae.W_dec.norm(dim=1).clone()

        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae(x)
        sgd.zero_grad()
        out.loss.backward()
        sae.project_decoder_grad()
        sgd.step()

        norms_after = sae.W_dec.norm(dim=1)
        # |W + lr*g_perp| = sqrt(1 + lr^2 |g_perp|^2) -> ~1 not exactly 1
        assert torch.allclose(norms_after, norms_before, atol=1e-3)
