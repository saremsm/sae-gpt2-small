# sae-gpt2

A sparse autoencoder for GPT-2-small residual streams. Trains a dictionary, handles dead features by resampling them onto high-error examples, and lets you query what each feature fires on across a corpus. Input normalization lives inside the SAE, so the checkpoint cannot be mis-used by skipping preprocessing.

## running

```
pip install -r requirements.txt
pytest -v
python main.py
```

`main.py` trains a 3072-feature SAE at layer 8 for 500K tokens, saves the checkpoint, plots curves, and prints analysis for the most interpretable features it finds.

## bugs fixed

One critical inconsistency and several config-level bugs, caught after the first full training runs. Each fix is described with its mechanism, matching the pattern in the notes section below.

1. **The analysis pipeline fed the SAE the exact input this README warned "will produce nonsense."** Training normalized activations to norm sqrt(d_model) inside the loop only, so `analysis.py` encoded raw layer-8 residuals (norm ~150) with a model trained at ~27.7 - rankings survived (a monotone per-feature rescale) while every absolute statistic was wrong. Fix: the normalization now lives inside the SAE (`SAEConfig.normalize_input`, applied in `encode`/`forward`/`resample_dead_features`), so training, analysis, and evaluation share one contract by construction; `test_encode_scale_invariant` pins it.
2. **The entire run happened inside LR warmup.** 500K tokens / 512-token batches is ~976 steps, against `warmup_steps=1000` - the learning rate ramped toward 2e-4 and the run ended before reaching it, so every number in the table below was produced at a fraction of the configured LR. Fix: `warmup_steps=100`, plus a clamp-with-warning in `train_sae` so this failure mode can't recur silently.
3. **Resampling never fired, and the dead-feature window was broken.** `resample_interval=1000` > 976 total steps, so the resampling machinery was never exercised in the headline run; and `feature_activation_counts` was only zeroed when a resample actually happened, so after any all-alive checkpoint the counting window grew without bound ("dead" degraded to "never fired since step 0"). Fix: interval lowered to 250 and the counter is zeroed at every resample checkpoint, giving true fixed-window semantics.
4. **Resampling drew candidates from a single 512-token batch.** With many dead features that reinitializes near-duplicate directions from one narrow slice of data. Fix: a rolling pool of the last 8 batches' activations and per-token errors (~12 MB) feeds the sampler; `SAEOutput` now carries `per_token_recon_error` so the loop never compares the normalized-space reconstruction against raw inputs.
5. **Variance explained wasn't FVU.** The old script compared flattened `.var()` ratios (deviation from the *global scalar* mean) - close to, but not, the FVU the literature reports. `variance_explained.py` now computes 1 - FVU properly (residual sum of squares over sum of squares about the per-dimension mean, in the SAE's normalized space).
6. Smaller items: the checkpoint stores `asdict(config)` so `torch.load(..., weights_only=True)` works (pickling the frozen dataclass forced the unsafe flag on every consumer); `find_interesting_features` takes `n_features: int` instead of the whole SAE and returns data instead of printing; a redundant re-sort after `topk` removed; the `sae_no_l1` fixture actually doesn't mutate now (`.eval()` - forward in train mode updates activation counts, so the old "doesn't mutate" comment was false); repeat data passes reshuffle document order; `feature_token_projection` documents its deliberate ln_final omission.

## numbers

Measured on a Lambda A10 (24 GB): ~32s wall-clock for 500K tokens, plus ~30s for analysis on 100 texts.

| | |
|---|---|
| training tokens | 500K |
| final reconstruction MSE | 0.45 |
| final L0 | 43 |
| dead features at end | 0 / 3072 (but see review item 3 - the window was broken) |
| variance explained | 55% (pre-FVU metric; see review item 5) |

Three features that came out interpretable enough to describe in a sentence:

- **feature 679** - sentence-boundary / discourse-connective detector. Fires on `.` followed by newline; the decoder row's logit-lens projection top-promotes *However*, *Furthermore*, *Moreover*, *Additionally*, *Therefore*. Reads as "end of sentence, next sentence will continue or contrast the argument."
- **feature 2107** - Q&A-format detector. Fires on the newline immediately after `Q:\n`. Logit lens promotes *What*, *How*, *Why*, *Hello*, *Answer*. The SAE has learned that this exact pattern precedes a question.
- **feature 2256** - UTF-8 encoding-artifact detector. Fires on the `\xef` byte that appears when a curly apostrophe or em-dash gets decoded with the wrong codec and then BPE-tokenized. Useful as a corpus-cleanliness signal.

## notes

`find_max_activating_examples` uses `searchsorted` to map a flat token index back to (text, position). Default `right=False` is off-by-one when a peak lands exactly on a text boundary - text k's first token gets attributed to text k-1. Caught by spot-checking analysis output. Regression in `test_peak_at_text_boundary`.

Dead-feature resampling kept producing features that died again on the next step. AdamW's first and second moments still carried momentum from when the feature had been alive (and then went dead), and that momentum dragged the reinitialized direction back toward zero on the very first step. Fix is `_zero_optim_state`; covered by `test_resample_zeros_optimizer_state`.

The decoder is constrained to unit-norm rows. Naive approach: renormalize after each optimizer step. That fights the optimizer - every step nudges rows off the unit sphere and we yank them back. Cleaner: project the component of `grad(W_dec)` parallel to each row out before stepping, so the optimizer never moves rows off the sphere in the first place (Anthropic, *Towards Monosemanticity*). Adam's element-wise update can still produce small drift, so I kept a renormalize call every 100 steps as a safety net. `project_decoder_grad` and `test_grad_perpendicular_after_projection`.

The encoder/decoder are tied at init: `W_dec` is random unit-norm rows, `W_enc = W_dec.T`. The earlier version used `kaiming_uniform_` on `W_enc`, which infers `fan_in` from `tensor.size(1)` - but here the actual fan-in is `size(0)`, so the init scale was wrong on the wrong axis. Tied init sidesteps the issue and is what the SAE reference implementations do anyway.

Earlier version of the training loop filled an 8-chunk buffer (~30K tokens), sampled 512 tokens per step, and discarded the rest. About 1.5% utilization. Replaced with a 64K-token shuffled buffer walked through batch-by-batch, refilled when a batch wouldn't fit. ~100% now. (Remaining simplification: the buffer is walked fully then refilled, so consecutive batches come from one shuffled corpus window; refill-at-half-with-reshuffle would decorrelate further.)

GPT-2's pad token id equals BOS/EOS, so a `tokens != pad_id` filter would also drop legitimate BOS positions. The activation source tokenizes once with HF's tokenizer (real attention mask), prepends BOS, then uses the mask to drop pad-position activations in one boolean-index op over the `(B, T, d_model)` activation tensor.

First training run came out with L0=866 - the SAE was using a quarter of its 3072 features on every token. Layer 8 residuals have norm ~150, so the L1 penalty `l1_coefficient * sum(|h_i|)` was tiny relative to MSE and sparsity wasn't being optimized. Standard fix: normalize activations to a fixed norm (`sqrt(d_model)`), which decouples `l1_coefficient` from the layer's activation scale. After normalizing and bumping `l1_coefficient` from 8e-4 to 5e-3, L0 settled at ~43. That normalization originally lived in the training loop only - which is how the analysis-path bug (item 1 above) happened; it now lives inside the SAE, where the checkpoint's input contract belongs.

The first analysis pass had every "interesting" feature firing on `<|endoftext|>` because BOS has an outlier activation that monopolises every per-feature top-k. Stripping it from the analysis corpus (one slice in `build_activation_cache`) was enough to surface the features above.

## known issues

L1 SAEs are not state of the art anymore. JumpReLU (DeepMind) and top-k (Gao et al, OpenAI) both do better on the L0/MSE frontier. I picked L1 because it's the simplest thing that demonstrates the rest of this; in production I'd reach for top-k. Variance explained at this token budget reflects that - vanilla L1 at 500K tokens isn't competitive with current methods, but both training longer and switching the encoder are addressable.

Layer 8 is a guess from probing literature on GPT-2-small. A real choice would grid-search 6-10 at fixed token budget and pick the layer with the lowest reconstruction at a target L0.

Single-process: model forward, SAE forward, and optimizer step run sequentially. The `ActivationSource` protocol is shaped so a producer/consumer split is contained - the trainer doesn't have to know whether the source is in-process or behind a socket.

In-memory feature cache. `(total_tokens, n_features)` in RAM is fine for 100 texts. The layout (flat token array plus per-text offsets) maps cleanly onto Parquet sharded by feature index when this becomes the bottleneck.
