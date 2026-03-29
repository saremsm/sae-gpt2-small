import dataclasses

import torch
import pytest

from sparse_autoencoder import SparseAutoencoder, SAEConfig
from analysis import (
    FeatureCache,
    feature_activation_stats,
    find_interesting_features,
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
    """l1=0 so loss == recon. class-scoped and in eval mode: forward in train mode
    mutates feature_activation_counts."""
    config = SAEConfig(
        d_model=D_MODEL,
        n_features=N_FEATURES,
        l1_coefficient=0.0,
        lr=2e-4,
        warmup_steps=10,
        normalize_decoder=True,
    )
    sae = SparseAutoencoder(config)
    sae.eval()
    return sae


class TestForwardPass:
    def test_recon_scaled_shape(self, sae_no_l1):
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae_no_l1(x)
        assert out.recon_scaled.shape == (BATCH_SIZE, D_MODEL)

    def test_recon_raw_shape_and_is_postprocessed(self, sae_no_l1):
        """recon_raw is postprocess(recon_scaled): same shape, raw space."""
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae_no_l1(x)
        assert out.recon_raw.shape == (BATCH_SIZE, D_MODEL)
        assert torch.allclose(
            out.recon_raw, sae_no_l1.postprocess(out.recon_scaled)
        )

    def test_h_shape(self, sae_no_l1):
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae_no_l1(x)
        assert out.h.shape == (BATCH_SIZE, N_FEATURES)

    def test_loss_is_scalar(self, sae_no_l1):
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae_no_l1(x)
        assert out.loss.shape == torch.Size([])

    def test_loss_is_finite(self, sae_no_l1):
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae_no_l1(x)
        assert out.loss.isfinite().item()

    def test_h_non_negative(self, sae_no_l1):
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae_no_l1(x)
        assert (out.h >= 0).all().item()

    def test_l0_finite_and_non_negative(self, sae_no_l1):
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae_no_l1(x)
        assert out.l0.isfinite().item() and out.l0.item() >= 0

    def test_counts_not_mutated_in_eval_mode(self, sae_no_l1):
        before = sae_no_l1.feature_activation_counts.clone()
        sae_no_l1(torch.randn(BATCH_SIZE, D_MODEL))
        assert torch.equal(sae_no_l1.feature_activation_counts, before)

    def test_per_token_recon_error_shape_and_grad(self, sae_no_l1):
        """per_token_recon_error must be per-token (B,) and detached."""
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae_no_l1(x)
        assert out.per_token_recon_error.shape == (BATCH_SIZE,)
        assert not out.per_token_recon_error.requires_grad
        assert (out.per_token_recon_error >= 0).all().item()


class TestInputNormalization:
    """The SAE owns its input contract: encode/forward multiply raw inputs by ONE."""
    def test_default_scale_is_one_and_identity(self, sae):
        assert sae.input_scale.item() == 1.0
        assert sae.input_scale.dtype == torch.float32
        x = torch.randn(BATCH_SIZE, D_MODEL) * 150.0
        assert torch.equal(sae.preprocess(x), x)

    def test_calibrated_mean_norm_is_sqrt_d(self, sae):
        """set_input_scale_from_activations: mean ||preprocess(x)|| == sqrt(d) to
        1e-4 relative on the calibration sample itself."""
        x = torch.randn(1000, D_MODEL) * 37.0
        sae.set_input_scale_from_activations(x)
        mean_norm = sae.preprocess(x).norm(dim=-1).mean().item()
        target = D_MODEL ** 0.5
        assert abs(mean_norm - target) / target < 1e-4

    def test_preprocess_is_linear_not_per_token(self, sae):
        """preprocess(2x) == 2 preprocess(x), and two tokens with different norms
        keep their norm ratio - no per-token equalisation."""
        sae.set_input_scale(0.37)
        x = torch.randn(BATCH_SIZE, D_MODEL)
        assert torch.allclose(sae.preprocess(2.0 * x), 2.0 * sae.preprocess(x))

        a = torch.randn(1, D_MODEL) * 10.0
        b = torch.randn(1, D_MODEL) * 150.0
        ratio_raw = (a.norm() / b.norm()).item()
        ratio_scaled = (
            sae.preprocess(a).norm() / sae.preprocess(b).norm()
        ).item()
        assert abs(ratio_raw - ratio_scaled) <= 1e-5 * ratio_raw

    def test_postprocess_inverts_preprocess(self, sae):
        sae.set_input_scale_from_activations(torch.randn(256, D_MODEL) * 150.0)
        x = torch.randn(BATCH_SIZE, D_MODEL) * 150.0
        # atol=1e-6 plus default rtol: float32 x*s/s round-trips to ~1 ulp
        assert torch.allclose(sae.postprocess(sae.preprocess(x)), x, atol=1e-6)

    def test_encode_applies_input_scale(self, sae):
        """encode(raw) with the flag on and scale s == encode(raw * s) with the flag
        off."""
        sae.set_input_scale_from_activations(torch.randn(256, D_MODEL) * 150.0)
        cfg_off = dataclasses.replace(sae.config, normalize_input=False)
        sae_off = SparseAutoencoder(cfg_off)
        sae_off.load_state_dict(sae.state_dict())

        x = torch.randn(BATCH_SIZE, D_MODEL) * 150.0
        assert torch.allclose(
            sae.encode(x), sae_off.encode(x * sae.input_scale), atol=1e-5
        )

    def test_flag_off_is_identity(self):
        """normalize_input=False: preprocess/postprocess ignore input_scale
        entirely."""
        cfg = SAEConfig(
            d_model=D_MODEL,
            n_features=N_FEATURES,
            normalize_input=False,
        )
        sae = SparseAutoencoder(cfg)
        sae.set_input_scale(0.2)
        x = torch.randn(BATCH_SIZE, D_MODEL) * 150.0
        assert torch.equal(sae.preprocess(x), x)
        assert torch.equal(sae.postprocess(x), x)

    def test_set_input_scale_rejects_degenerate_values(self, sae):
        with pytest.raises(ValueError):
            sae.set_input_scale(0.0)
        with pytest.raises(ValueError):
            sae.set_input_scale(float("inf"))
        with pytest.raises(ValueError):
            sae.set_input_scale_from_activations(torch.zeros(4, D_MODEL))

    def test_input_scale_survives_state_dict_round_trip(self, sae):
        sae.set_input_scale(0.1234)
        fresh = SparseAutoencoder(sae.config)
        fresh.load_state_dict(sae.state_dict())
        assert torch.equal(fresh.input_scale, sae.input_scale)

    def test_missing_input_scale_defaults_to_one_with_warning(self, sae):
        """a pre-input_scale checkpoint loads (strict=True) with a warning and
        input_scale reset to 1.0 - reset, not merely left alone."""
        legacy = {k: v for k, v in sae.state_dict().items() if k != "input_scale"}
        fresh = SparseAutoencoder(sae.config)
        fresh.set_input_scale(5.0)
        with pytest.warns(UserWarning, match="input_scale"):
            fresh.load_state_dict(legacy)
        assert fresh.input_scale.item() == 1.0

    def test_forward_recon_raw_matches_input(self):
        """perfectly reconstructing SAE: W_dec = [I; -I], W_enc = W_dec.T, zero
        biases, so decode(encode(x)) = relu(x) - relu(-x) = x. Then recon_raw ==
        x_raw whatever input_scale is, and recon_scaled == preprocess(x_raw)."""
        d = 8
        cfg = SAEConfig(d_model=d, n_features=2 * d, l1_coefficient=0.0)
        sae = SparseAutoencoder(cfg).eval()
        with torch.no_grad():
            eye = torch.eye(d)
            sae.W_dec.copy_(torch.cat([eye, -eye], dim=0))
            sae.W_enc.copy_(sae.W_dec.T)
            sae.b_enc.zero_()
            sae.b_dec.zero_()

        x_raw = torch.randn(BATCH_SIZE, d) * 150.0
        sae.set_input_scale_from_activations(x_raw)
        assert sae.input_scale.item() != 1.0  # else the test proves nothing

        out = sae(x_raw)
        assert torch.allclose(out.recon_raw, x_raw, atol=1e-4)
        assert torch.allclose(out.recon_scaled, sae.preprocess(x_raw), atol=1e-4)
        assert out.reconstruction_loss.item() < 1e-8


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

    def test_resampled_decoder_rows_unit_norm_from_raw_inputs(self, sae):
        """resample_dead_features preprocesses internally, so raw-scale activations
        still yield unit-norm rows in the scaled space W_dec lives in."""
        sae.feature_activation_counts.zero_()
        sae.feature_activation_counts[:10] = 5
        dead = sae.get_dead_features(threshold=0)

        raw_activations = torch.randn(BATCH_SIZE, D_MODEL) * 150.0
        errors = torch.rand(BATCH_SIZE)
        sae.resample_dead_features(dead, raw_activations, errors)

        norms = sae.W_dec.data[dead].norm(dim=1)
        assert torch.allclose(norms, torch.ones(len(dead)), atol=1e-5)

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


class TestFindInterestingFeatures:
    @pytest.fixture
    def two_feature_cache(self) -> FeatureCache:
        """feature 0: rate 10%, mean-when-active 2.0 (in-band, strong); feature 1:
        rate 50%, mean-when-active 0.1 (doubly out-of-band)."""
        n_tokens = 100
        fa = torch.zeros(n_tokens, 2)
        fa[:10, 0] = 2.0
        fa[:50, 1] = 0.1
        return FeatureCache(
            feature_acts=fa,
            text_offsets=torch.tensor([0, n_tokens], dtype=torch.long),
            token_strings=[["tok"] * n_tokens],
            texts=["tok " * n_tokens],
        )

    def test_in_band_feature_selected(self, two_feature_cache):
        result = find_interesting_features(two_feature_cache, n_features=2)
        assert 0 in result

    def test_out_of_band_feature_rejected(self, two_feature_cache):
        result = find_interesting_features(two_feature_cache, n_features=2)
        assert 1 not in result

    def test_chunked_reduce_matches_full_reduce(self, two_feature_cache):
        small = find_interesting_features(
            two_feature_cache, n_features=2, chunk_size=3
        )
        full = find_interesting_features(
            two_feature_cache, n_features=2, chunk_size=1000
        )
        assert small == full

    def test_shape_mismatch_raises(self, two_feature_cache):
        with pytest.raises(ValueError):
            find_interesting_features(two_feature_cache, n_features=7)

    def test_returns_data_not_prints(self, two_feature_cache, capsys):
        """design principle, executable: queries return data, never print."""
        find_interesting_features(two_feature_cache, n_features=2)
        assert capsys.readouterr().out == ""

    def test_activation_stats_values(self, two_feature_cache):
        """rate and mean-when-active are what the fixture was built to have, and the
        chunked reduction agrees with a single-chunk one."""
        stats = feature_activation_stats(two_feature_cache, chunk_size=3)
        assert torch.allclose(stats.rate, torch.tensor([0.1, 0.5]))
        assert torch.allclose(stats.mean_when_active, torch.tensor([2.0, 0.1]))

        full = feature_activation_stats(two_feature_cache, chunk_size=1000)
        assert torch.allclose(stats.rate, full.rate)
        assert torch.allclose(stats.mean_when_active, full.mean_when_active)

    def test_ranked_by_mean_when_active_desc(self):
        """three in-band features (rates 10%/5%/15%) with means 1.0/3.0/2.0 come
        back strongest first: [1, 2, 0]; n_return truncates the head."""
        n_tokens = 100
        fa = torch.zeros(n_tokens, 3)
        fa[:10, 0] = 1.0
        fa[:5, 1] = 3.0
        fa[:15, 2] = 2.0
        cache = FeatureCache(
            feature_acts=fa,
            text_offsets=torch.tensor([0, n_tokens], dtype=torch.long),
            token_strings=[["tok"] * n_tokens],
            texts=["tok " * n_tokens],
        )
        assert find_interesting_features(cache, n_features=3) == [1, 2, 0]
        assert find_interesting_features(cache, n_features=3, n_return=2) == [1, 2]


class TestTrainSaeCalibration:
    """train_sae calibrates input_scale on the first buffer fill."""

    class _SyntheticSource:
        """yields n_chunks of (chunk_tokens, d) raw activations."""
        def __init__(self, d: int, chunk_tokens: int, n_chunks: int, scale: float):
            self.d, self.chunk_tokens, self.n_chunks, self.scale = (
                d, chunk_tokens, n_chunks, scale,
            )

        def __iter__(self):
            g = torch.Generator().manual_seed(0)
            for _ in range(self.n_chunks):
                yield torch.randn(self.chunk_tokens, self.d, generator=g) * self.scale

    def test_input_scale_calibrated_before_training(self):
        from training import train_sae

        d, scale = 16, 37.0
        cfg = SAEConfig(d_model=d, n_features=32, l1_coefficient=1e-3, warmup_steps=2)
        sae = SparseAutoencoder(cfg)
        src = self._SyntheticSource(d=d, chunk_tokens=300, n_chunks=20, scale=scale)

        history = train_sae(
            sae, src, n_training_tokens=1024, resample_interval=1000,
            log_interval=1, device="cpu", calibration_tokens=1000,
        )

        assert history["loss"], "training ran no steps"
        assert sae.input_scale.item() != 1.0
        # every chunk is iid, so scale * E||x|| == sqrt(d) up to sampling noise.
        mean_norm = torch.cat(list(src)).norm(dim=-1).mean().item()
        assert abs(sae.input_scale.item() * mean_norm - d ** 0.5) / d ** 0.5 < 0.02

    def test_calibration_tokens_zero_leaves_scale_alone(self):
        from training import train_sae

        d = 16
        cfg = SAEConfig(d_model=d, n_features=32, l1_coefficient=1e-3, warmup_steps=2)
        sae = SparseAutoencoder(cfg)
        sae.set_input_scale(0.25)
        src = self._SyntheticSource(d=d, chunk_tokens=300, n_chunks=10, scale=37.0)

        train_sae(
            sae, src, n_training_tokens=512, resample_interval=1000,
            log_interval=1, device="cpu", calibration_tokens=0,
        )
        assert sae.input_scale.item() == 0.25
