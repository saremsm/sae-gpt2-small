import dataclasses
import json
import os

import numpy as np
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


class TestTopK:
    """activation="topk" (Gao et al. 2024): per token keep the k largest pre-
    activations (ReLU'd), zero the rest; loss is MSE only."""

    K = 8

    def _topk_sae(self, **overrides) -> SparseAutoencoder:
        kw = dict(
            d_model=D_MODEL, n_features=N_FEATURES, l1_coefficient=0.0,
            lr=2e-4, warmup_steps=10, activation="topk", k=self.K,
        )
        kw.update(overrides)
        return SparseAutoencoder(SAEConfig(**kw))

    def test_config_validation(self):
        """topk needs k in [1, n_features]; k is rejected under relu; aux_k needs
        topk; a non-zero l1_coefficient under topk warns and is forced to 0 (so
        the checkpointed config is honest)."""
        with pytest.raises(ValueError, match="requires k"):
            SAEConfig(d_model=D_MODEL, n_features=N_FEATURES, activation="topk")
        with pytest.raises(ValueError, match="k must be"):
            SAEConfig(d_model=D_MODEL, n_features=N_FEATURES, activation="topk",
                      k=N_FEATURES + 1)
        with pytest.raises(ValueError, match="k must be"):
            SAEConfig(d_model=D_MODEL, n_features=N_FEATURES, activation="topk", k=0)
        with pytest.raises(ValueError, match="only meaningful"):
            SAEConfig(d_model=D_MODEL, n_features=N_FEATURES, k=8)
        with pytest.raises(ValueError, match="requires activation"):
            SAEConfig(d_model=D_MODEL, n_features=N_FEATURES, aux_k=4)
        with pytest.raises(ValueError, match="activation must be"):
            SAEConfig(d_model=D_MODEL, n_features=N_FEATURES, activation="jumprelu")
        with pytest.warns(UserWarning, match="l1_coefficient"):
            cfg = SAEConfig(d_model=D_MODEL, n_features=N_FEATURES,
                            activation="topk", k=8, l1_coefficient=5e-3)
        assert cfg.l1_coefficient == 0.0
        # the default is still relu, k None: existing callers unchanged
        default = SAEConfig(d_model=D_MODEL, n_features=N_FEATURES)
        assert default.activation == "relu" and default.k is None
        assert default.aux_k == 0 and default.aux_coeff == 0.0

    def test_relu_path_unchanged(self, sae):
        """the default SAE still returns relu(pre): the refactor into."""
        x = torch.randn(BATCH_SIZE, D_MODEL)
        pre = sae.pre_activations(sae.preprocess(x))
        assert torch.equal(sae.encode(x), torch.relu(pre))
        assert sae.activation == "relu" and sae.k is None

    def test_l0_equals_k_exactly_when_enough_positive(self):
        """with every pre-activation positive (b_enc pushed up) every token has
        EXACTLY k non-zeros, and h agrees with the top-k of the pre-activations;
        the same input through eval-mode forward gives l0 == k."""
        sae = self._topk_sae().eval()
        x = torch.randn(BATCH_SIZE, D_MODEL)
        with torch.no_grad():
            sae.b_enc.fill_(100.0)
        h = sae.encode(x)
        assert h.shape == (BATCH_SIZE, N_FEATURES)
        assert (h >= 0).all()
        assert ((h > 0).sum(dim=-1) == self.K).all(), (h > 0).sum(dim=-1)

        pre = sae.pre_activations(sae.preprocess(x))
        assert (pre > 0).all()  # else the test proves nothing
        top_vals, top_idx = pre.topk(self.K, dim=-1)
        assert torch.allclose(h.gather(-1, top_idx), top_vals)
        # nothing outside the top k survives
        mask = torch.zeros_like(h, dtype=torch.bool).scatter(-1, top_idx, True)
        assert (h[~mask] == 0).all()

        out = sae(x)
        assert out.l0.item() == pytest.approx(self.K)
        assert torch.equal(out.h, h)

        with torch.no_grad():
            sae.b_enc.fill_(-100.0)
        assert (sae.encode(x) == 0).all()
        assert sae(x).l0.item() == 0.0

    def test_topk_is_per_token(self):
        """the top k is taken over the feature dim of every token independently: two
        tokens with different rankings keep different latents."""
        sae = self._topk_sae().eval()
        with torch.no_grad():
            sae.b_enc.fill_(50.0)
        x = torch.randn(3, D_MODEL)
        x = torch.cat([x, x[:1]])
        h = sae.encode(x)
        active = [set((h[i] > 0).nonzero(as_tuple=True)[0].tolist()) for i in range(4)]
        assert all(len(a) == self.K for a in active)
        assert active[0] == active[3]
        assert active[0] != active[1]

    def test_gradient_flows_only_through_selected_latents(self):
        """d(sum h^2)/d(pre) is non-zero exactly on the kept latents; through a full
        forward, W_enc columns / W_dec rows of features that were never selected
        in the batch get zero gradient."""
        sae = self._topk_sae()
        with torch.no_grad():
            sae.b_enc.fill_(50.0)  # all pre-activations positive
        x = torch.randn(BATCH_SIZE, D_MODEL)

        pre = sae.pre_activations(sae.preprocess(x))
        pre.retain_grad()
        h = sae.apply_activation(pre)
        (h ** 2).sum().backward()
        selected = h > 0
        assert selected.sum(dim=-1).eq(self.K).all()
        assert (pre.grad[selected] != 0).all()
        assert (pre.grad[~selected] == 0).all()

        sae.zero_grad(set_to_none=True)
        out = sae(x)
        out.loss.backward()
        ever_selected = selected.any(dim=0)
        assert 0 < int(ever_selected.sum()) < N_FEATURES  # both sides populated
        assert (sae.W_enc.grad[:, ~ever_selected] == 0).all()
        assert (sae.b_enc.grad[~ever_selected] == 0).all()
        assert (sae.W_dec.grad[~ever_selected] == 0).all()
        assert (sae.W_dec.grad[ever_selected] != 0).any()

    def test_loss_is_mse_only(self):
        """topk without AuxK: loss == reconstruction MSE, sparsity_loss is still
        reported (L1 of the kept latents) but carries weight 0."""
        sae = self._topk_sae()
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae(x)
        assert torch.equal(out.loss, out.reconstruction_loss)
        assert out.aux_loss.item() == 0.0
        assert out.sparsity_loss.item() > 0

    def test_checkpoint_round_trip_carries_activation_and_k(self, tmp_path):
        """the main.py checkpoint layout (asdict(config) + state_dict) through
        eval.load_checkpoint: activation, k, aux_k."""
        from eval import load_checkpoint

        sae = self._topk_sae(aux_k=16, aux_coeff=1 / 32).eval()
        sae.set_input_scale(0.2281)
        path = tmp_path / "topk.pt"
        torch.save({"sae_state_dict": sae.state_dict(),
                    "config": dataclasses.asdict(sae.config), "layer": 8}, path)
        loaded, ckpt = load_checkpoint(str(path), "cpu")
        assert ckpt["config"]["activation"] == "topk" and ckpt["config"]["k"] == self.K
        assert loaded.config == sae.config
        assert loaded.activation == "topk" and loaded.k == self.K
        assert loaded.config.aux_k == 16 and loaded.config.aux_coeff == 1 / 32
        assert loaded.config.l1_coefficient == 0.0
        x = torch.randn(BATCH_SIZE, D_MODEL) * 150.0
        assert torch.equal(loaded.encode(x), sae.encode(x))
        assert ((loaded.encode(x) > 0).sum(dim=-1) <= self.K).all()

    def test_aux_loss_zero_without_dead_positive_with_dead(self):
        """train mode, aux_k > 0: no dead feature -> aux_loss 0 and loss == MSE;
        some dead -> aux_loss > 0 and loss == MSE + aux_coeff * aux."""
        coeff = 1 / 32
        sae = self._topk_sae(aux_k=4, aux_coeff=coeff).train()
        x = torch.randn(BATCH_SIZE, D_MODEL)

        with torch.no_grad():
            sae.feature_activation_counts.fill_(1)
        out = sae(x)
        assert out.aux_loss.item() == 0.0
        assert torch.equal(out.loss, out.reconstruction_loss)

        with torch.no_grad():
            sae.feature_activation_counts.fill_(1)
            sae.feature_activation_counts[:8] = 0
        out = sae(x)
        assert out.aux_loss.item() > 0.0
        assert out.loss.item() == pytest.approx(
            out.reconstruction_loss.item() + coeff * out.aux_loss.item(), rel=1e-6
        )
        assert out.aux_loss.requires_grad

        with torch.no_grad():
            sae.feature_activation_counts.fill_(1)
            sae.feature_activation_counts[:8] = 0
            sae.W_dec[:8] = 0.0
        out = sae(x)
        residual = sae.preprocess(x) - out.recon_scaled
        assert out.aux_loss.item() == pytest.approx(
            residual.pow(2).mean().item(), rel=1e-5
        )

    def test_aux_gradient_reaches_only_dead_features(self):
        """the aux term's gradient lands on the dead features' W_dec rows and W_enc
        columns and nowhere else (the residual is detached, so the live
        dictionary gets nothing from it). aux_k (16) > the 8 dead features."""
        sae = self._topk_sae(aux_k=16, aux_coeff=1.0).train()
        with torch.no_grad():
            sae.b_enc.fill_(50.0)
            sae.feature_activation_counts.fill_(1)
            sae.feature_activation_counts[:8] = 0
        x = torch.randn(BATCH_SIZE, D_MODEL)
        pre = sae.pre_activations(sae.preprocess(x))
        recon = sae.decode(sae.apply_activation(pre))
        aux = sae.aux_loss(pre, sae.preprocess(x), recon)
        aux.backward()
        assert torch.isfinite(aux).item() and aux.item() > 0
        for p in (sae.W_dec, sae.W_enc, sae.b_enc):
            assert torch.isfinite(p.grad).all()
        assert (sae.W_dec.grad[8:] == 0).all()
        assert (sae.W_dec.grad[:8] != 0).any()
        assert (sae.W_enc.grad[:, 8:] == 0).all()
        assert (sae.b_enc.grad[8:] == 0).all()

    def test_aux_is_training_only(self):
        """in eval mode the AuxK term is 0 whatever the counts say."""
        sae = self._topk_sae(aux_k=4, aux_coeff=1.0).eval()
        x = torch.randn(BATCH_SIZE, D_MODEL)
        assert (sae.feature_activation_counts == 0).all()
        out = sae(x)
        assert out.aux_loss.item() == 0.0
        assert torch.equal(out.loss, out.reconstruction_loss)

    def test_counts_and_resampling_work_under_topk(self):
        """feature_activation_counts accumulate (h > 0) under topk exactly as under
        relu, and resample_dead_features runs unchanged on them: counts zeroed,
        resampled decoder rows unit norm, adam state reset."""
        sae = self._topk_sae().train()
        optimizer = torch.optim.AdamW(sae.parameters(), lr=1e-3)
        x = torch.randn(BATCH_SIZE, D_MODEL)
        out = sae(x)
        assert sae.feature_activation_counts.sum().item() == int((out.h > 0).sum())
        assert 0 < sae.feature_activation_counts.sum().item() <= BATCH_SIZE * self.K
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()

        dead = sae.get_dead_features(threshold=0)
        assert len(dead) > 0  # 16 tokens x 8 latents cannot cover 256 features
        sae.resample_dead_features(dead, x * 150.0, torch.rand(BATCH_SIZE),
                                   optimizer=optimizer)
        assert (sae.feature_activation_counts == 0).all()
        assert torch.allclose(sae.W_dec.data[dead].norm(dim=1),
                              torch.ones(len(dead)), atol=1e-5)
        assert (optimizer.state[sae.W_enc]["exp_avg"][:, dead] == 0).all()
        # and the SAE still encodes exactly k per token afterwards
        with torch.no_grad():
            sae.b_enc.fill_(50.0)
        assert ((sae.encode(x) > 0).sum(dim=-1) == self.K).all()

    def test_train_sae_topk_five_steps(self):
        """training smoke: 5 steps of train_sae under topk + AuxK on a synthetic
        source (CPU, hermetic), resampling firing on the way."""
        from training import train_sae

        cfg = SAEConfig(d_model=D_MODEL, n_features=N_FEATURES, l1_coefficient=0.0,
                        warmup_steps=2, activation="topk", k=self.K,
                        aux_k=2 * self.K, aux_coeff=1 / 32)
        sae = SparseAutoencoder(cfg)
        src = TestTrainSaeCalibration._SyntheticSource(
            d=D_MODEL, batch_tokens=64, n_chunks=8, scale=37.0)
        history = train_sae(
            sae, src, n_training_tokens=5 * 64, resample_interval=3,
            log_interval=1, device="cpu", calibration_tokens=128,
        )
        assert history["step"] == [1, 2, 3, 4, 5]
        assert all(np.isfinite(history["loss"]))
        assert all(l0 <= self.K + 1e-6 for l0 in history["l0"])
        assert len(history["aux_loss"]) == 5 and all(v >= 0 for v in history["aux_loss"])
        assert sae.input_scale.item() != 1.0
        assert ((sae.eval().encode(next(iter(src))) > 0).sum(dim=-1) <= self.K).all()


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

    def test_topk_feature_cache_queries_work(self):
        """the analysis queries make no ReLU-sparsity assumption: on a FeatureCache
        built from a topk SAE (exactly k non-zeros per token)
        find_interesting_features returns in-band features and"""
        from analysis import build_feature_cache, ActivationCache

        k = 8
        torch.manual_seed(0)
        cfg = SAEConfig(d_model=D_MODEL, n_features=N_FEATURES, l1_coefficient=0.0,
                        activation="topk", k=k)
        sae = SparseAutoencoder(cfg)
        with torch.no_grad():
            sae.b_enc.fill_(1.0)  # ensure the kept latents are > 0.5
        acts = [torch.randn(40, D_MODEL), torch.randn(25, D_MODEL)]
        cache = build_feature_cache(
            sae,
            ActivationCache(activations=acts,
                            token_strings=[[f"a{i}" for i in range(40)],
                                           [f"b{i}" for i in range(25)]],
                            texts=["A", "B"]),
        )
        assert cache.feature_acts.shape == (65, N_FEATURES)
        assert ((cache.feature_acts > 0).sum(dim=-1) == k).all()

        stats = feature_activation_stats(cache)
        assert stats.rate.sum().item() == pytest.approx(k)  # sum of rates == k
        interesting = find_interesting_features(cache, n_features=N_FEATURES)
        assert interesting  # k / n_features = 3% mean rate: inside the band
        for f in interesting:
            assert 0.001 < stats.rate[f].item() < 0.2
        top = find_max_activating_examples(cache, interesting[0], top_k=3)
        assert top and all(e["activation"] > 0 for e in top)
        # attribution round-trips: the peak token string is the one at (text,
        for e in top:
            assert e["peak_token"][0] in ("a", "b")


class TestTrainSaeCalibration:
    """train_sae calibrates input_scale on the first batches the loader yields."""

    class _SyntheticSource:
        """yields n_chunks batches of (batch_tokens, d) raw activations."""
        def __init__(self, d: int, batch_tokens: int, n_chunks: int, scale: float):
            self.d, self.batch_tokens, self.n_chunks, self.scale = (
                d, batch_tokens, n_chunks, scale,
            )

        def __iter__(self):
            g = torch.Generator().manual_seed(0)
            for _ in range(self.n_chunks):
                yield torch.randn(self.batch_tokens, self.d, generator=g) * self.scale

    def test_input_scale_calibrated_before_training(self):
        from training import train_sae

        d, scale = 16, 37.0
        cfg = SAEConfig(d_model=d, n_features=32, l1_coefficient=1e-3, warmup_steps=2)
        sae = SparseAutoencoder(cfg)
        src = self._SyntheticSource(d=d, batch_tokens=300, n_chunks=20, scale=scale)

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
        src = self._SyntheticSource(d=d, batch_tokens=300, n_chunks=10, scale=37.0)

        train_sae(
            sae, src, n_training_tokens=512, resample_interval=1000,
            log_interval=1, device="cpu", calibration_tokens=0,
        )
        assert sae.input_scale.item() == 0.25

    def test_calibration_batches_are_trained_on_then_source_continues(self):
        """the calibration batches are not discarded: with 4 batches of 256 in the
        source and calibration on 512, a 1024-token run does 4 steps."""
        from training import train_sae

        d = 16
        cfg = SAEConfig(d_model=d, n_features=32, l1_coefficient=1e-3, warmup_steps=2)
        sae = SparseAutoencoder(cfg)
        src = self._SyntheticSource(d=d, batch_tokens=256, n_chunks=4, scale=37.0)

        history = train_sae(
            sae, src, n_training_tokens=2048, resample_interval=1000,
            log_interval=1, device="cpu", calibration_tokens=512,
        )
        assert history["step"] == [1, 2, 3, 4]


# data pipeline: shards, tokenization, ActivationLoader

BOS = 50256
SHARD_SEQS = 64
SHARD_SEQ_LEN = 32


def _write_shard(dir_path, tokens: np.ndarray, seq_len: int):
    """write a (n_seqs, seq_len) uint16 array as a shard (bin + sidecar) and open
    it."""
    from data import TokenShard, shard_meta_path

    bin_path = os.path.join(dir_path, "shard.bin")
    tokens = np.ascontiguousarray(tokens, dtype=np.uint16)
    with open(bin_path, "wb") as f:
        f.write(tokens.tobytes())
    meta = {
        "n_tokens": int(tokens.size), "n_seqs": int(tokens.shape[0]),
        "seq_len": seq_len, "seed": 0, "dataset": "<test>", "split": "train",
        "text_field": "text", "doc_range": [0, 0], "n_docs": 0,
        "tokenizer": "gpt2", "bos_id": BOS, "eos_id": BOS, "dtype": "uint16",
    }
    with open(shard_meta_path(bin_path), "w") as f:
        json.dump(meta, f)
    return TokenShard(bin_path)


def _random_shard_tokens(n_seqs: int, seq_len: int, seed: int = 0) -> np.ndarray:
    """BOS at position 0; column 1 distinct per row so every (row, position >= 1)
    prefix - hence every non-BOS activation."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, 50256, size=(n_seqs, seq_len), dtype=np.uint16)
    tokens[:, 0] = BOS
    tokens[:, 1] = 1000 + np.arange(n_seqs)
    return tokens


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def loader_shard(tmp_path_factory):
    d = tmp_path_factory.mktemp("shard")
    return _write_shard(str(d), _random_shard_tokens(SHARD_SEQS, SHARD_SEQ_LEN), SHARD_SEQ_LEN)


@pytest.fixture(scope="session")
def tiny_model():
    """random-weight HookedTransformer with GPT-2's vocab (so shard ids are valid)
    but 3 tiny layers: exercises the exact loader code path hermetically."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=3, d_model=32, d_head=8, n_heads=4, d_mlp=64, d_vocab=50257,
        n_ctx=64, act_fn="gelu_new", normalization_type="LN", seed=0,
        device="cpu",  # TL defaults to CUDA when present; these tests are CPU
    )
    return HookedTransformer(cfg).eval()


@pytest.fixture(scope="session")
def gpt2_model():
    """the real GPT-2-small. Skips (does not fail) when weights cannot load."""
    from transformer_lens import HookedTransformer

    try:
        # and these tests build CPU loaders.
        #   device="cpu": from_pretrained otherwise picks CUDA when present,
        return HookedTransformer.from_pretrained("gpt2", device="cpu").eval()
    except OSError as exc:
        pytest.skip(f"GPT-2 weights unavailable offline: {exc}")


@pytest.fixture(params=["tiny", "gpt2"])
def model_and_hook(request):
    """(model, hook_name): the hermetic tiny model at its layer 1."""
    from data import resid_post_hook

    if request.param == "tiny":
        return request.getfixturevalue("tiny_model"), resid_post_hook(1)
    return request.getfixturevalue("gpt2_model"), resid_post_hook(8)


def _reference_activations(model, hook_name: str, tokens: torch.Tensor) -> torch.Tensor:
    """(n_seqs, seq_len, d) activations at hook_name from a plain run_with_cache
    WITHOUT stop_at_layer."""
    outs = []
    with torch.no_grad():
        for start in range(0, tokens.shape[0], 16):
            _, cache = model.run_with_cache(tokens[start : start + 16], names_filter=hook_name)
            outs.append(cache[hook_name].float())
    return torch.cat(outs, dim=0)


class TestTokenizeCorpus:
    DOCS = [
        f"Document number {i}. "
        + "The quick brown fox jumps over the lazy dog. " * (1 + i % 4)
        for i in range(30)
    ]

    @staticmethod
    def _stream(doc_ids: list[list[int]]) -> np.ndarray:
        """the packed body stream: every doc followed by EOS."""
        return np.concatenate([np.array(ids + [BOS]) for ids in doc_ids])

    @staticmethod
    def _rows(path: str, seq_len: int) -> np.ndarray:
        raw = np.fromfile(path, dtype=np.uint16)
        return raw.reshape(-1, seq_len)

    def test_packing_from_iterable(self, tmp_path, tokenizer):
        """BOS at position 0 of every row, then a contiguous stream of docs
        separated by EOS; uint16 on disk; n_tokens rounded down to whole rows;
        sidecar carries the contract."""
        from data import tokenize_corpus

        seq_len, n_tokens = 16, 20 * 16 + 5
        out = tmp_path / "train.bin"
        result = tokenize_corpus(
            [{"text": t} for t in self.DOCS], out_path=str(out),
            n_tokens=n_tokens, seq_len=seq_len,
        )
        meta = result["train"]
        assert result["holdout"] is None
        assert meta["n_seqs"] == 20 and meta["n_tokens"] == 320
        assert meta["dtype"] == "uint16" and meta["seq_len"] == seq_len
        assert meta["bos_id"] == BOS and meta["tokenizer"] == "gpt2"
        assert os.path.getsize(out) == 320 * 2  # two bytes per token
        assert json.load(open(tmp_path / "train.json")) == meta

        rows = self._rows(str(out), seq_len)
        assert rows.shape == (20, seq_len)
        assert (rows[:, 0] == BOS).all()
        expected = self._stream([tokenizer(t)["input_ids"] for t in self.DOCS])
        assert (rows[:, 1:].reshape(-1) == expected[: 20 * (seq_len - 1)]).all()

    def test_holdout_docs_disjoint_via_load_dataset(self, tmp_path, tokenizer, monkeypatch):
        """with a dataset name, load_dataset is streamed and shuffled with (seed,
        buffer_size=10_000)"""
        import data

        docs = self.DOCS
        calls = {}

        class _FakeStream:
            def __init__(self, items):
                self._items = items

            def shuffle(self, seed, buffer_size):
                calls["shuffle"] = (seed, buffer_size)
                order = np.random.default_rng(seed).permutation(len(self._items))
                return _FakeStream([self._items[i] for i in order])

            def __iter__(self):
                return iter(self._items)

        def fake_load_dataset(name, split, streaming):
            calls["load"] = (name, split, streaming)
            return _FakeStream([{"content": t} for t in docs])

        monkeypatch.setattr(data, "load_dataset", fake_load_dataset)
        seq_len, holdout_docs = 16, 4
        result = data.tokenize_corpus(
            "fake/dataset", split="train", out_path=str(tmp_path / "train.bin"),
            n_tokens=8 * seq_len, seq_len=seq_len, seed=7, text_field="content",
            holdout_docs=holdout_docs, holdout_path=str(tmp_path / "holdout.bin"),
        )
        assert calls["load"] == ("fake/dataset", "train", True)
        assert calls["shuffle"] == (7, 10_000)

        order = np.random.default_rng(7).permutation(len(docs))
        shuffled_ids = [tokenizer(docs[i])["input_ids"] for i in order]
        hold_stream = self._stream(shuffled_ids[:holdout_docs])
        train_stream = self._stream(shuffled_ids[holdout_docs:])

        hold_rows = self._rows(str(tmp_path / "holdout.bin"), seq_len)
        train_rows = self._rows(str(tmp_path / "train.bin"), seq_len)
        assert (hold_rows[:, 0] == BOS).all() and (train_rows[:, 0] == BOS).all()
        n_hold = len(hold_stream) // (seq_len - 1)
        assert hold_rows.shape[0] == n_hold  # everything the docs give
        assert (hold_rows[:, 1:].reshape(-1) == hold_stream[: n_hold * (seq_len - 1)]).all()
        assert train_rows.shape[0] == 8
        assert (train_rows[:, 1:].reshape(-1) == train_stream[: 8 * (seq_len - 1)]).all()

        hm, tm = result["holdout"], result["train"]
        assert hm["doc_range"] == [0, holdout_docs]
        assert tm["doc_range"][0] == holdout_docs
        assert hm["n_docs"] == holdout_docs
        assert hm["seed"] == tm["seed"] == 7 and tm["dataset"] == "fake/dataset"
        assert tm["text_field"] == "content"

    @pytest.mark.parametrize("streaming", [True, False])
    def test_real_datasets_objects_and_cli(self, tmp_path, tokenizer, monkeypatch, streaming):
        """against real `datasets` objects (IterableDataset.shuffle takes
        buffer_size, Dataset.shuffle does not) through the CLI entry point; the
        split is a seeded permutation of the documents either way."""
        import data
        from datasets import Dataset

        base = Dataset.from_dict({"text": self.DOCS})

        def fake_load_dataset(name, split, streaming):
            return base.to_iterable_dataset() if streaming else base

        monkeypatch.setattr(data, "load_dataset", fake_load_dataset)
        argv = [
            "tokenize", "--dataset", "fake/ds", "--n-tokens", "160", "--seq-len", "16",
            "--holdout-docs", "5", "--out", str(tmp_path / "t.bin"),
            "--holdout-out", str(tmp_path / "h.bin"), "--seed", "3",
        ]
        if not streaming:
            argv.append("--no-streaming")
        data.main(argv)

        train, hold = data.TokenShard(tmp_path / "t.bin"), data.TokenShard(tmp_path / "h.bin")
        assert train.n_seqs == 10 and hold.n_seqs >= 1
        assert hold.meta["doc_range"] == [0, 5] and train.meta["doc_range"][0] == 5
        # the held-out docs are a strict subset of the corpus and none of them starts.
        first_hold_doc = tokenizer.decode(hold[0][1:8].tolist())
        first_train_doc = tokenizer.decode(train[0][1:8].tolist())
        assert first_hold_doc.startswith("Document number")
        assert first_train_doc.startswith("Document number")
        assert first_hold_doc != first_train_doc

    def test_exhausted_stream_writes_what_it_has(self, tmp_path, tokenizer, capsys):
        from data import tokenize_corpus

        result = tokenize_corpus(
            self.DOCS[:5], out_path=str(tmp_path / "t.bin"),
            n_tokens=10_000, seq_len=16,
        )
        total = sum(len(tokenizer(t)["input_ids"]) + 1 for t in self.DOCS[:5])
        assert result["train"]["n_seqs"] == total // 15 < 10_000 // 16
        assert "WARNING: stream exhausted" in capsys.readouterr().out

    def test_argument_validation(self, tmp_path):
        from data import tokenize_corpus

        with pytest.raises(ValueError):
            tokenize_corpus(self.DOCS, out_path=str(tmp_path / "t.bin"),
                            n_tokens=64, seq_len=16, holdout_docs=2)
        with pytest.raises(ValueError):
            tokenize_corpus(self.DOCS, out_path=str(tmp_path / "t.bin"),
                            n_tokens=8, seq_len=16)


class TestTokenShard:
    def test_round_trip_written_ids_equal_read_ids(self, tmp_path):
        tokens = _random_shard_tokens(10, 8, seed=3)
        shard = _write_shard(str(tmp_path), tokens, 8)
        assert shard.n_seqs == 10 and shard.n_tokens == 80 and len(shard) == 10
        assert shard.seq_len == 8

        rows = shard[[2, 5, 9]]
        assert rows.dtype == torch.long and rows.shape == (3, 8)
        assert np.array_equal(rows.numpy(), tokens[[2, 5, 9]])
        assert np.array_equal(shard[4].numpy(), tokens[4])
        assert np.array_equal(shard[torch.tensor([0, 1])].numpy(), tokens[:2])
        assert np.array_equal(shard[3:6].numpy(), tokens[3:6])

    def test_iter_batches_covers_each_row_once_per_epoch(self, tmp_path):
        tokens = _random_shard_tokens(10, 8, seed=1)
        shard = _write_shard(str(tmp_path), tokens, 8)
        batches = list(shard.iter_batches(4, shuffle=True, seed=0, epochs=2))
        assert [b.shape[0] for b in batches] == [4, 4, 2, 4, 4, 2]
        seen = torch.cat(batches).numpy()
        # column 1 is distinct per row: identifies rows
        ids_epoch1 = sorted(seen[:10, 1].tolist())
        ids_epoch2 = sorted(seen[10:, 1].tolist())
        assert ids_epoch1 == ids_epoch2 == sorted(tokens[:, 1].tolist())
        assert seen[:10, 1].tolist() != seen[10:, 1].tolist()  # reshuffled
        ordered = torch.cat(list(shard.iter_batches(3, shuffle=False)))
        assert np.array_equal(ordered.numpy(), tokens)

    def test_size_mismatch_raises(self, tmp_path):
        from data import TokenShard

        shard = _write_shard(str(tmp_path), _random_shard_tokens(4, 8), 8)
        with open(shard.path, "ab") as f:
            f.write(b"\x00\x00")
        with pytest.raises(ValueError):
            TokenShard(shard.path)


class TestActivationLoader:
    def _loader(self, model, hook_name, shard, **kw):
        from data import ActivationLoader

        kwargs = dict(
            batch_seqs=8, batch_tokens=128, buffer_tokens=512, device="cpu",
            seed=0, log_every=0,
        )
        kwargs.update(kw)
        return ActivationLoader(model, shard, hook_name, **kwargs)

    def test_batch_shape_dtype_and_bos_excluded(self, model_and_hook, loader_shard):
        """batches are (batch_tokens, d_model) float32 and no yielded row is the
        position-0 activation."""
        model, hook_name = model_and_hook
        loader = self._loader(model, hook_name, loader_shard)
        bos_act = _reference_activations(model, hook_name, loader_shard[[0]])[0, 0]

        for _ in range(3):
            batch = next(loader)
            assert batch.shape == (128, model.cfg.d_model)
            assert batch.dtype == torch.float32
            dist = (batch - bos_act).norm(dim=-1)
            assert (dist > 1e-3 * bos_act.norm()).all()
        assert loader.tokens_yielded == 3 * 128
        assert loader.throughput_tok_s() > 0

    def test_gpt2_position0_outlier_is_gone(self, gpt2_model, loader_shard):
        """on the real model the position-0 residual has norm ~3141 vs ~120
        elsewhere (README notes): with position 0 in, a 128-row batch averages >
        200 from the outlier alone."""
        from data import resid_post_hook

        loader = self._loader(gpt2_model, resid_post_hook(8), loader_shard)
        for _ in range(4):
            assert next(loader).norm(dim=-1).mean().item() < 500.0

    def test_exclude_bos_false_keeps_position0(self, model_and_hook, loader_shard):
        model, hook_name = model_and_hook
        loader = self._loader(model, hook_name, loader_shard, exclude_bos=False, epochs=1)
        bos_act = _reference_activations(model, hook_name, loader_shard[[0]])[0, 0]
        rows = torch.cat(list(loader))
        assert rows.shape[0] == SHARD_SEQS * SHARD_SEQ_LEN  # 2048 = 16 x 128
        n_bos = ((rows - bos_act).norm(dim=-1) < 1e-3 * bos_act.norm()).sum().item()
        assert n_bos == SHARD_SEQS

    def test_epoch_yields_each_activation_once_and_matches_full_forward(
        self, model_and_hook, loader_shard
    ):
        """one epoch through a buffer far smaller than the shard (so it refills
        several times): every yielded row is a non-BOS activation of the shard."""
        model, hook_name = model_and_hook
        loader = self._loader(model, hook_name, loader_shard, epochs=1)
        ref = _reference_activations(model, hook_name, loader_shard[np.arange(SHARD_SEQS)])
        ref = ref[:, 1:].reshape(-1, model.cfg.d_model)  # drop position 0

        batches = list(loader)
        assert len(batches) == (SHARD_SEQS * (SHARD_SEQ_LEN - 1)) // 128 == 15
        assert loader.n_refills >= 3
        assert loader.tokens_yielded == 15 * 128
        assert loader.tokens_forwarded == SHARD_SEQS * SHARD_SEQ_LEN

        rows = torch.cat(batches)
        dists = torch.cdist(rows, ref)
        min_dist, nearest = dists.min(dim=1)
        # loader and reference use different batch compositions.
        assert (min_dist <= 2e-2 * ref[nearest].norm(dim=-1)).all(), min_dist.max()
        assert nearest.unique().numel() == rows.shape[0], "an activation was yielded twice"

    def test_epochs_none_cycles(self, tiny_model, loader_shard):
        from data import resid_post_hook

        loader = self._loader(tiny_model, resid_post_hook(1), loader_shard, epochs=None)
        n = 0
        for batch in loader:
            n += batch.shape[0]
            if n > 3 * SHARD_SEQS * (SHARD_SEQ_LEN - 1):
                break
        assert loader.n_chunks > 3 * (SHARD_SEQS // 8)

    def test_argument_validation(self, tiny_model, loader_shard):
        from data import ActivationLoader, layer_from_hook_name, resid_post_hook

        with pytest.raises(ValueError):
            ActivationLoader(tiny_model, loader_shard, resid_post_hook(1),
                             batch_seqs=8, batch_tokens=300, buffer_tokens=512, device="cpu")
        with pytest.raises(ValueError):
            ActivationLoader(tiny_model, loader_shard, "hook_embed",
                             batch_seqs=8, batch_tokens=64, buffer_tokens=512, device="cpu")
        assert layer_from_hook_name("blocks.8.hook_resid_post") == 8
        assert resid_post_hook(8) == "blocks.8.hook_resid_post"


class TestTrainSaeFromShard:
    def test_five_steps_on_cpu(self, model_and_hook, loader_shard):
        """train_sae smoke: 5 steps from a shard through the loader, with
        calibration and a resample checkpoint on the way."""
        from data import ActivationLoader
        from training import train_sae

        model, hook_name = model_and_hook
        loader = ActivationLoader(
            model, loader_shard, hook_name, batch_seqs=8, batch_tokens=128,
            buffer_tokens=512, device="cpu", seed=0, log_every=0,
        )
        cfg = SAEConfig(d_model=model.cfg.d_model, n_features=4 * model.cfg.d_model,
                        l1_coefficient=1e-3, warmup_steps=2)
        sae = SparseAutoencoder(cfg)
        history = train_sae(
            sae, loader, n_training_tokens=5 * 128, resample_interval=3,
            log_interval=1, device="cpu", calibration_tokens=256,
        )
        assert history["step"] == [1, 2, 3, 4, 5]
        assert all(np.isfinite(history["loss"]))
        assert sae.input_scale.item() != 1.0
        assert loader.tokens_yielded == 5 * 128

    def test_five_steps_topk_on_cpu(self, model_and_hook, loader_shard):
        """the same 5-step smoke with activation=topk (+ AuxK) through the real
        loader path: MSE-only loss, L0 <= k at every logged step, resample
        checkpoint at step 3 taken."""
        from data import ActivationLoader
        from training import train_sae

        model, hook_name = model_and_hook
        loader = ActivationLoader(
            model, loader_shard, hook_name, batch_seqs=8, batch_tokens=128,
            buffer_tokens=512, device="cpu", seed=0, log_every=0,
        )
        k = 16
        cfg = SAEConfig(d_model=model.cfg.d_model, n_features=4 * model.cfg.d_model,
                        l1_coefficient=0.0, warmup_steps=2, activation="topk", k=k,
                        aux_k=2 * k, aux_coeff=1 / 32)
        sae = SparseAutoencoder(cfg)
        history = train_sae(
            sae, loader, n_training_tokens=5 * 128, resample_interval=3,
            log_interval=1, device="cpu", calibration_tokens=256,
        )
        assert history["step"] == [1, 2, 3, 4, 5]
        assert all(np.isfinite(history["loss"]))
        assert all(l0 <= k for l0 in history["l0"])
        assert loader.tokens_yielded == 5 * 128


@pytest.fixture(scope="module")
def random_hf_and_tl():
    """a random-weight GPT-2-config HF model and the TL model built FROM those same
    weights with TL's default processing (fold_ln, center_writing_weights, ...)"""
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformer_lens import HookedTransformer

    cfg = GPT2Config(n_layer=12, n_embd=768, n_head=12, vocab_size=50257,
                     n_positions=1024)
    torch.manual_seed(0)
    hf = GPT2LMHeadModel(cfg).eval()
    tl = HookedTransformer.from_pretrained("gpt2", hf_model=hf, device="cpu").eval()
    return hf, tl


class TestHFResidualBackend:
    """data.HFResidualModel must produce TransformerLens' hook_resid_post exactly."""

    def test_hf_backend_matches_transformerlens_resid_post(self, random_hf_and_tl, loader_shard):
        """HF-after-block-8 minus per-token mean == TL blocks.8.hook_resid_post to
        1e-4, and without the centring they differ - so the test is sensitive."""
        from data import HFResidualModel, resid_post_hook

        hf, tl = random_hf_and_tl
        backend = HFResidualModel(hf, center=True)
        assert backend.cfg.d_model == tl.cfg.d_model == 768

        tokens = loader_shard[np.arange(8)]
        with torch.no_grad():
            _, cache = tl.run_with_cache(tokens, names_filter=resid_post_hook(8),
                                         stop_at_layer=9)
            tl_resid = cache[resid_post_hook(8)]
            hf_resid = backend.resid_post(tokens, 8)
            hf_uncentered = HFResidualModel(hf, center=False).resid_post(tokens, 8)
        scale = tl_resid.norm(dim=-1).mean()
        assert (hf_resid - tl_resid).abs().max() < 1e-4 * scale
        assert (hf_uncentered - tl_resid).abs().max() > 1e-3 * scale

    def test_loader_yields_same_activations_either_backend(self, random_hf_and_tl, loader_shard):
        """the ActivationLoader driven by the HF backend and by TL (same seed)
        yields the same batches in the same order."""
        from data import ActivationLoader, HFResidualModel, resid_post_hook

        hf, tl = random_hf_and_tl
        kw = dict(batch_seqs=8, batch_tokens=128, buffer_tokens=512, device="cpu",
                  seed=0, log_every=0)
        a = ActivationLoader(HFResidualModel(hf), loader_shard, resid_post_hook(8), **kw)
        b = ActivationLoader(tl, loader_shard, resid_post_hook(8), **kw)
        for _ in range(3):
            xa, xb = next(a), next(b)
            assert xa.shape == xb.shape == (128, 768)
            assert (xa - xb).abs().max() < 1e-4 * xb.norm(dim=-1).mean()

    def test_real_gpt2_hf_backend_matches(self, gpt2_model, loader_shard):
        """same check against the real weights (skips offline like the other GPT-2
        tests): the shipped `--forward hf` path is exact."""
        from data import HFResidualModel, resid_post_hook

        try:
            backend = HFResidualModel.from_pretrained("gpt2", device="cpu")
        except OSError as exc:
            pytest.skip(f"HF GPT-2 weights unavailable offline: {exc}")
        tokens = loader_shard[np.arange(8)]
        with torch.no_grad():
            _, cache = gpt2_model.run_with_cache(tokens, names_filter=resid_post_hook(8),
                                                 stop_at_layer=9)
            tl_resid = cache[resid_post_hook(8)]
            hf_resid = backend.resid_post(tokens, 8)
        # position 0 is the ~3000-norm outlier; compare relative to each row
        rel = (hf_resid - tl_resid).norm(dim=-1) / tl_resid.norm(dim=-1)
        assert rel.max() < 1e-4

# held-out evaluator (eval.py)


def _exact_sae(d: int, input_scale: float = 0.3) -> SparseAutoencoder:
    """W_dec = [I; -I], W_enc = W_dec.T, zero biases: decode(encode(x)) == relu(x) -
    relu(-x) == x, so recon_raw == x for any input_scale."""
    cfg = SAEConfig(d_model=d, n_features=2 * d, l1_coefficient=0.0)
    sae = SparseAutoencoder(cfg).eval()
    with torch.no_grad():
        eye = torch.eye(d)
        sae.W_dec.copy_(torch.cat([eye, -eye], dim=0))
        sae.W_enc.copy_(sae.W_dec.T)
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    sae.set_input_scale(input_scale)
    return sae


def _zero_sae(d: int, n_features: int = 64) -> SparseAutoencoder:
    """h == 0 on every input and b_dec == 0: recon_raw is all zeros, so the
    reconstruction splice IS a zero-ablation of the same positions."""
    sae = SparseAutoencoder(SAEConfig(d_model=d, n_features=n_features)).eval()
    with torch.no_grad():
        sae.W_enc.zero_()
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    sae.set_input_scale(0.3)
    return sae


def _meta(dataset="pile", split="train", seed=0, doc_range=(0, 0)) -> dict:
    return {"dataset": dataset, "split": split, "seed": seed,
            "doc_range": list(doc_range)}


@pytest.fixture(scope="session")
def gpt2_text_shard(tmp_path_factory, tokenizer):
    """16 rows x 64 of packed real text (tokenize_corpus over in-memory docs) for
    the real-GPT-2 evaluator tests: zero-ablating layer 8 on English text raises."""
    from data import TokenShard, tokenize_corpus

    docs = [
        f"Paragraph {i}. The committee met on Tuesday to discuss the budget "
        f"for the coming year, and after a long debate the members agreed "
        f"to postpone the decision until more information was available. "
        * (2 + i % 3)
        for i in range(40)
    ]
    d = tmp_path_factory.mktemp("gpt2_eval_shard")
    tokenize_corpus(docs, out_path=str(d / "holdout.bin"), n_tokens=16 * 64, seq_len=64)
    shard = TokenShard(d / "holdout.bin")
    assert shard.n_seqs == 16
    return shard


class TestCheckHoldoutDisjoint:
    def test_overlap_raises(self):
        from eval import check_holdout_disjoint

        with pytest.raises(ValueError, match="overlaps"):
            check_holdout_disjoint(_meta(doc_range=(0, 100)), _meta(doc_range=(50, 500)))
        with pytest.raises(ValueError, match="overlaps"):
            check_holdout_disjoint(_meta(doc_range=(0, 100)), _meta(doc_range=(0, 100)))
        # containment is overlap too
        with pytest.raises(ValueError, match="overlaps"):
            check_holdout_disjoint(_meta(doc_range=(10, 20)), _meta(doc_range=(0, 100)))

    def test_disjoint_passes(self):
        from eval import check_holdout_disjoint

        # the tokenize_corpus layout: holdout first, train after it
        check_holdout_disjoint(_meta(doc_range=(0, 100)), _meta(doc_range=(100, 500)))
        # empty ranges (the test shards) never overlap
        check_holdout_disjoint(_meta(doc_range=(0, 0)), _meta(doc_range=(0, 0)))
        # different corpora are disjoint by construction, whatever the ranges
        check_holdout_disjoint(_meta(dataset="owt", doc_range=(0, 100)),
                               _meta(dataset="pile", doc_range=(0, 100)))

    def test_same_dataset_other_seed_or_split_cannot_be_certified(self):
        from eval import check_holdout_disjoint

        with pytest.raises(ValueError, match="cannot be certified"):
            check_holdout_disjoint(_meta(seed=1, doc_range=(0, 10)), _meta(seed=0, doc_range=(10, 20)))
        with pytest.raises(ValueError, match="cannot be certified"):
            check_holdout_disjoint(_meta(split="test", doc_range=(0, 10)), _meta(doc_range=(10, 20)))
        with pytest.raises(ValueError, match="doc_range"):
            check_holdout_disjoint({"dataset": "pile"}, _meta())

    def test_evaluate_refuses_overlapping_shards_before_running(self, tiny_model, loader_shard):
        """evaluate(train_meta=...) with an overlapping doc_range raises before any
        forward: the model never sees a hook."""
        from data import resid_post_hook
        from eval import evaluate

        holdout_meta = dict(loader_shard.meta, doc_range=[0, 10])
        train_meta = dict(loader_shard.meta, doc_range=[5, 50])
        loader_shard.meta = holdout_meta  # session fixture: restore below
        try:
            with pytest.raises(ValueError, match="overlaps"):
                evaluate(_exact_sae(tiny_model.cfg.d_model), tiny_model, loader_shard,
                         resid_post_hook(1), n_tokens=64, batch_seqs=8, device="cpu",
                         train_meta=train_meta)
        finally:
            loader_shard.meta = dict(loader_shard.meta, doc_range=[0, 0])
        assert not any(hp.fwd_hooks for hp in tiny_model.hook_dict.values())


class TestEvaluate:
    HOOK = "blocks.1.hook_resid_post"

    def _eval(self, sae, model, shard, **kw):
        from eval import evaluate

        kwargs = dict(n_tokens=64 * 31, batch_seqs=8, device="cpu", log_every=0)
        kwargs.update(kw)
        return evaluate(sae, model, shard, self.HOOK, **kwargs)

    def test_exact_sae_fvu_zero_and_identity_loss_recovered_one(self, tiny_model, loader_shard):
        """constructed exact reconstruction: fvu ~ 0, mse ~ 0, ce_recon == ce_clean
        so loss_recovered ~ 1."""
        from eval import METRIC_KEYS

        d = tiny_model.cfg.d_model
        m = self._eval(_exact_sae(d), tiny_model, loader_shard)
        assert set(METRIC_KEYS) <= set(m)
        assert m["n_tokens"] == 64 * 31 and m["n_seqs"] == 64
        assert m["fvu"] < 1e-8 and m["mse_raw"] < 1e-8
        assert abs(m["variance_explained"] - 1.0) < 1e-8
        assert abs(m["ce_recon"] - m["ce_clean"]) < 1e-5
        assert abs(m["loss_recovered"] - 1.0) < 1e-3
        assert m["ce_zero"] != m["ce_clean"]  # else loss_recovered proves nothing
        assert m["l0"] == pytest.approx(d)  # relu(x) - relu(-x): one of each pair
        assert m["dead_frac_eval"] == 0.0 and not m["identity"] and m["exclude_bos"]

        for sae in (_exact_sae(d), SparseAutoencoder(SAEConfig(d_model=d, n_features=4 * d)).eval()):
            mi = self._eval(sae, tiny_model, loader_shard, identity=True)
            assert mi["identity"] and mi["fvu"] == 0.0 and mi["mse_raw"] == 0.0
            assert abs(mi["loss_recovered"] - 1.0) <= 1e-3
            assert mi["ce_recon"] == pytest.approx(m["ce_clean"], abs=1e-6)
        # ...while the random SAE itself does not reconstruct
        assert self._eval(sae, tiny_model, loader_shard)["fvu"] > 0.1

    def test_fvu_l0_ce_match_two_pass_reference(self, tiny_model, loader_shard):
        """streaming (Welford) FVU over batches == a two-pass FVU over the whole
        eval set from run_with_cache; L0 == the SAE's own; ce_clean ==
        model(tokens, return_type='loss')"""
        d = tiny_model.cfg.d_model
        torch.manual_seed(1)
        sae = SparseAutoencoder(SAEConfig(d_model=d, n_features=4 * d)).eval()
        sae.set_input_scale(0.3)
        m = self._eval(sae, tiny_model, loader_shard, n_tokens=1000)
        assert m["n_tokens"] == 5 * 248 and m["n_seqs"] == 40

        tokens = loader_shard[np.arange(40)]
        with torch.no_grad():
            _, cache = tiny_model.run_with_cache(tokens, names_filter=self.HOOK)
            x = cache[self.HOOK][:, 1:].reshape(-1, d).double()
            out = sae(x.float())
            fvu = ((x - out.recon_raw.double()) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
            ce = tiny_model(tokens, return_type="loss").item()
        assert m["fvu"] == pytest.approx(fvu.item(), rel=1e-6)
        assert m["mse_raw"] == pytest.approx(
            ((x - out.recon_raw.double()) ** 2).mean().item(), rel=1e-6)
        assert m["mse_scaled"] == pytest.approx(m["mse_raw"] * 0.3 ** 2, rel=1e-6)
        assert m["l0"] == pytest.approx(out.l0.item(), rel=1e-5)
        assert m["ce_clean"] == pytest.approx(ce, abs=1e-5)
        n_dead = int(((out.h > 0).sum(0) == 0).sum())
        assert m["n_dead_eval"] == n_dead
        assert m["dead_frac_eval"] == pytest.approx(n_dead / (4 * d))

    def test_zero_sae_equals_zero_ablation_and_bos_is_untouched(self, tiny_model, loader_shard):
        """an SAE whose reconstruction is all zeros splices exactly what the zero-
        ablation writes at exactly the same positions: ce_recon == ce_zero,
        loss_recovered == 0, l0 == 0, every feature dead on eval."""
        d = tiny_model.cfg.d_model
        m = self._eval(_zero_sae(d), tiny_model, loader_shard)
        assert m["ce_recon"] == pytest.approx(m["ce_zero"], abs=1e-6)
        assert abs(m["loss_recovered"]) < 1e-3
        assert m["l0"] == 0.0 and m["dead_frac_eval"] == 1.0
        assert m["fvu"] > 0.5  # zeros explain nothing (about the mean, less)

        m_all = self._eval(_zero_sae(d), tiny_model, loader_shard, exclude_bos=False)
        assert not m_all["exclude_bos"]
        assert m_all["n_tokens"] == 64 * 32  # position 0 counted now
        assert m_all["ce_recon"] == pytest.approx(m_all["ce_zero"], abs=1e-6)
        assert m_all["ce_clean"] == pytest.approx(m["ce_clean"], abs=1e-6)
        assert abs(m_all["ce_zero"] - m["ce_zero"]) > 1e-4  # BOS zeroed too

    def test_argument_validation(self, tiny_model, loader_shard):
        from eval import evaluate

        d = tiny_model.cfg.d_model
        with pytest.raises(ValueError):
            evaluate(_exact_sae(d), tiny_model, loader_shard, self.HOOK,
                     n_tokens=0, batch_seqs=8, device="cpu")
        with pytest.raises(ValueError):
            evaluate(_exact_sae(d), tiny_model, loader_shard, self.HOOK,
                     n_tokens=10, batch_seqs=0, device="cpu")
        with pytest.raises(ValueError, match="d_model"):
            evaluate(_exact_sae(d + 1), tiny_model, loader_shard, self.HOOK,
                     n_tokens=10, batch_seqs=8, device="cpu")

    def test_real_gpt2_zero_ablation_ce_above_clean(self, gpt2_model, gpt2_text_shard):
        """real GPT-2, 16 seqs of packed English text, layer-8 hook: zero- ablating
        the residual raises the loss (by nats, not noise), the identity splice
        recovers it exactly (loss_recovered == 1 to 1e-3, fvu == 0)"""
        from data import resid_post_hook
        from eval import evaluate

        hook = resid_post_hook(8)
        d = gpt2_model.cfg.d_model
        m = evaluate(_exact_sae(d, input_scale=0.228), gpt2_model, gpt2_text_shard, hook,
                     n_tokens=16 * 63, batch_seqs=8, device="cpu", identity=True)
        assert m["n_tokens"] == 16 * 63 and m["n_seqs"] == 16
        assert m["ce_zero"] > m["ce_clean"] + 0.5
        assert m["fvu"] == 0.0
        assert abs(m["loss_recovered"] - 1.0) <= 1e-3
        with torch.no_grad():
            ce = gpt2_model(gpt2_text_shard[np.arange(16)], return_type="loss").item()
        assert m["ce_clean"] == pytest.approx(ce, abs=1e-4)
        assert m["ce_clean"] < 6.0  # English text, not random ids


class TestRunRecord:
    def _fake_metrics(self) -> dict:
        from eval import METRIC_KEYS

        vals = {k: 0.5 for k in METRIC_KEYS}
        vals.update(n_tokens=1000, n_seqs=8, n_dead_eval=3, exclude_bos=True,
                    identity=False, hook_name="blocks.8.hook_resid_post")
        return vals

    def test_metrics_json_schema(self, tmp_path):
        """results/<run>/metrics.json: every RUN_RECORD_KEYS key, config + all
        METRIC_KEYS + ISO-8601 UTC timestamps + git SHA (None or hex), round-
        trips through JSON."""
        import datetime
        from eval import (METRIC_KEYS, RUN_RECORD_KEYS, make_run_record,
                          utc_now_iso, write_json)

        started = utc_now_iso()
        record = make_run_record(
            run="test-run", config={"sae": {"d_model": 8}, "args": {"n_tokens": 5}},
            metrics=self._fake_metrics(), started_at=started,
            train_shard=_meta(doc_range=(10, 20)), holdout_shard=_meta(doc_range=(0, 10)),
            checkpoint="sae.pt", training={"steps": 3},
        )
        path = write_json(tmp_path / "results" / "test-run" / "metrics.json", record)
        loaded = json.load(open(path))
        assert set(RUN_RECORD_KEYS) <= set(loaded)
        assert loaded["run"] == "test-run"
        assert loaded["config"]["sae"]["d_model"] == 8
        assert set(METRIC_KEYS) <= set(loaded["metrics"])
        assert loaded["metrics"]["n_tokens"] == 1000
        for key in ("started_at", "finished_at"):
            ts = datetime.datetime.fromisoformat(loaded[key])
            assert ts.tzinfo is not None and ts.utcoffset().total_seconds() == 0
        assert loaded["finished_at"] >= loaded["started_at"]
        sha = loaded["git_sha"]
        assert sha is None or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))
        assert loaded["train_shard"]["doc_range"] == [10, 20]
        assert loaded["checkpoint"] == "sae.pt" and loaded["training"] == {"steps": 3}

    def test_missing_metric_key_raises_and_none_is_allowed(self):
        from eval import make_run_record, utc_now_iso

        metrics = self._fake_metrics()
        del metrics["loss_recovered"]
        with pytest.raises(ValueError, match="loss_recovered"):
            make_run_record("r", {}, metrics, utc_now_iso())
        record = make_run_record("r", {}, None, utc_now_iso())
        assert record["metrics"] is None

    def test_load_checkpoint_round_trip(self, tmp_path):
        """the main.py checkpoint layout (plain-dict config, state_dict with
        input_scale) loads under weights_only=True."""
        import dataclasses
        from eval import load_checkpoint

        cfg = SAEConfig(d_model=D_MODEL, n_features=N_FEATURES, l1_coefficient=1e-3)
        sae = SparseAutoencoder(cfg)
        sae.set_input_scale(0.2281)
        path = tmp_path / "ckpt.pt"
        torch.save({"sae_state_dict": sae.state_dict(), "config": dataclasses.asdict(cfg),
                    "layer": 8, "train_shard": _meta()}, path)
        loaded, ckpt = load_checkpoint(str(path), "cpu")
        assert loaded.config == cfg and not loaded.training
        assert loaded.input_scale.item() == pytest.approx(0.2281)
        assert torch.equal(loaded.W_dec, sae.W_dec)
        assert ckpt["layer"] == 8 and ckpt["train_shard"]["doc_range"] == [0, 0]

    def test_eval_cli_identity_mode(self, tmp_path, tiny_model, loader_shard, monkeypatch, capsys):
        """`python -m eval --identity` end to end on the tiny model (its
        from_pretrained monkeypatched away; --device cpu because the tiny model
        lives on CPU even on a CUDA box): loads the checkpoint."""
        import dataclasses
        import eval as eval_mod
        import transformer_lens

        monkeypatch.setattr(
            transformer_lens.HookedTransformer, "from_pretrained",
            classmethod(lambda cls, *a, **k: tiny_model),
        )
        cfg = SAEConfig(d_model=tiny_model.cfg.d_model, n_features=64)
        sae = SparseAutoencoder(cfg)
        sae.set_input_scale(0.3)
        ckpt = tmp_path / "ckpt.pt"
        torch.save({"sae_state_dict": sae.state_dict(), "config": dataclasses.asdict(cfg),
                    "layer": 1, "train_shard": dict(loader_shard.meta, doc_range=[100, 200])},
                   ckpt)
        out_json = tmp_path / "m.json"
        eval_mod.main(["--checkpoint", str(ckpt), "--holdout", str(loader_shard.path),
                       "--n-tokens", "500", "--batch-seqs", "8", "--identity",
                       "--json", str(out_json), "--device", "cpu"])
        printed = capsys.readouterr().out
        assert "identity check       : OK" in printed
        m = json.load(open(out_json))
        assert m["identity"] and m["fvu"] == 0.0 and abs(m["loss_recovered"] - 1) <= 1e-3
        assert m["hook_name"] == "blocks.1.hook_resid_post" and m["n_tokens"] == 3 * 248

        # a held-out shard whose sidecar says docs [0, 10) vs a checkpoint trained.
        from data import shard_meta_path

        overlap = _write_shard(str(tmp_path), _random_shard_tokens(16, SHARD_SEQ_LEN), SHARD_SEQ_LEN)
        with open(shard_meta_path(overlap.path), "w") as f:
            json.dump(dict(overlap.meta, doc_range=[0, 10]), f)
        torch.save({"sae_state_dict": sae.state_dict(), "config": dataclasses.asdict(cfg),
                    "layer": 1, "train_shard": dict(overlap.meta, doc_range=[0, 5])},
                   ckpt)
        with pytest.raises(ValueError, match="overlaps"):
            eval_mod.main(["--checkpoint", str(ckpt), "--holdout", str(overlap.path),
                           "--n-tokens", "500", "--batch-seqs", "8", "--device", "cpu"])
