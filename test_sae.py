import torch
import pytest

from sparse_autoencoder import SparseAutoencoder, SAEConfig
from analysis import (
    FeatureCache,
    find_max_activating_examples,
)

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


class TestResampleDeadFeatures:
    def test_get_dead_features_finds_correct_indices(self, sae):
        sae.feature_activation_counts[:10] = torch.arange(1, 11, dtype=torch.long)
        sae.feature_activation_counts[10:] = 0

        dead_indices = sae.get_dead_features(threshold=0)
        assert dead_indices.tolist() == list(range(10, N_FEATURES))

    def test_all_counts_zeroed_after_resample(self, sae):
        sae.feature_activation_counts[:10] = torch.arange(1, 11, dtype=torch.long)
        sae.feature_activation_counts[10:] = 0

        dead_indices = sae.get_dead_features(threshold=0)
        activations = torch.randn(BATCH_SIZE, D_MODEL)
        errors = torch.rand(BATCH_SIZE)

        sae.resample_dead_features(
            dead_feature_indices=dead_indices,
            activations=activations,
            errors=errors,
        )

        assert (sae.feature_activation_counts == 0).all().item()

    def test_resample_zeros_optimizer_state(self, sae):
        """resampled slices must zero adam moments. otherwise stale momentum drags
        the reinit'd direction back to zero on step 1."""
        optimizer = torch.optim.AdamW(sae.parameters(), lr=1e-3)

        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae(x)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()

        sae.feature_activation_counts.zero_()
        sae.feature_activation_counts[:10] = 5

        dead_indices = sae.get_dead_features(threshold=0)
        activations = torch.randn(BATCH_SIZE, D_MODEL)
        errors = torch.rand(BATCH_SIZE)

        sae.resample_dead_features(
            dead_feature_indices=dead_indices,
            activations=activations,
            errors=errors,
            optimizer=optimizer,
        )

        state_enc = optimizer.state[sae.W_enc]
        assert (state_enc["exp_avg"][:, dead_indices] == 0).all().item()
        assert (state_enc["exp_avg_sq"][:, dead_indices] == 0).all().item()

        state_dec = optimizer.state[sae.W_dec]
        assert (state_dec["exp_avg"][dead_indices] == 0).all().item()
        assert (state_dec["exp_avg_sq"][dead_indices] == 0).all().item()


class TestFindMaxActivatingExamples:
    @pytest.fixture
    def minimal_cache(self) -> FeatureCache:
        feature_acts = torch.tensor(
            [[0.5], [9.9], [0.1],
             [1.0], [0.2]],
            dtype=torch.float32,
        )
        return FeatureCache(
            feature_acts=feature_acts,
            text_offsets=torch.tensor([0, 3, 5], dtype=torch.long),
            token_strings=[["hello", "world", "foo"], ["bar", "baz"]],
            texts=["hello world foo", "bar baz"],
        )

    def test_returns_results(self, minimal_cache):
        results = find_max_activating_examples(
            minimal_cache, feature_idx=0, top_k=3
        )
        assert len(results) >= 1

    def test_rank0_activation_value(self, minimal_cache):
        results = find_max_activating_examples(
            minimal_cache, feature_idx=0, top_k=3
        )
        assert abs(results[0]["activation"] - 9.9) < 1e-4

    def test_rank0_peak_token(self, minimal_cache):
        results = find_max_activating_examples(
            minimal_cache, feature_idx=0, top_k=3
        )
        assert results[0]["peak_token"] == "world"

    def test_peak_position_in_context(self, minimal_cache):
        results = find_max_activating_examples(
            minimal_cache, feature_idx=0, top_k=3, context_window=1
        )
        top = results[0]
        assert top["context"][top["peak_position_in_context"]] == "world"

    def test_descending_order(self, minimal_cache):
        results = find_max_activating_examples(
            minimal_cache, feature_idx=0, top_k=3
        )
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i]["activation"] >= results[i + 1]["activation"]

    def test_peak_at_text_boundary(self):
        """regression: peak at flat_pos == offsets[k] for k > 0. right=False ->
        searchsorted returns k, text_idx = k-1 (wrong). right=True -> returns"""
        feature_acts = torch.tensor(
            [[0.5], [0.1], [0.2],
             [9.9], [0.3]],
            dtype=torch.float32,
        )
        cache = FeatureCache(
            feature_acts=feature_acts,
            text_offsets=torch.tensor([0, 3, 5], dtype=torch.long),
            token_strings=[["a", "b", "c"], ["X", "Y"]],
            texts=["a b c", "X Y"],
        )
        results = find_max_activating_examples(cache, feature_idx=0, top_k=1)
        assert results[0]["peak_token"] == "X", (
            f"expected 'X' (first token of text 1), "
            f"got '{results[0]['peak_token']}'"
        )
