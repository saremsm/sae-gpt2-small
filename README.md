# sae-gpt2

A sparse autoencoder for GPT-2-small residual streams. Trains a dictionary, handles dead features by resampling them onto high-error examples, and lets you query what each feature fires on across a corpus. Input scaling - one dataset-wide scalar calibrated from the data - lives inside the SAE and its checkpoint, so the model cannot be mis-used by skipping preprocessing, and reconstructions map back to raw residual space for splicing into the model.

## running

```
pip install -r requirements.txt
pytest -v
python main.py
```

`main.py` trains a 3072-feature SAE at layer 8 for 500K tokens (calibrating the input scale first), saves the checkpoint, plots curves, and prints analysis for the most interpretable features it finds. `variance_explained.py` reports raw-space FVU for the saved checkpoint. `profile_sae.py` profiles one SAE step under `torch.profiler` (CUDA only).

## reporting conventions

Every number in this file lives in one of two spaces, and the table below labels each row with its contract. Read nothing across rows without matching contract, position-0 policy, and L0.

- **Loss and MSE are in scaled space**: `x * input_scale`, where `input_scale = sqrt(d_model) / mean ||x||` over the calibration sample. Under per-token normalization (the earlier contract) every token had norm sqrt(d); under the scalar contract only the *mean* does. MSE is not comparable across the two.
- **FVU is in raw residual space** from `recon_raw` (residual sum of squares over sum of squares about the per-dimension mean). It is scale-invariant - numerator and denominator both pick up `input_scale`^2 - so it is the one number that survives a change of contract; it is *not* invariant to whether position 0 is in the eval set (see the notes section: with position 0 in, FVU is meaningless).
- **L0** is active features per token, space-independent.
- **tok/s** is tokens through GPT-2 forward + SAE step per second of wall clock, on the stated GPU.

## bugs fixed

One critical inconsistency and several config-level bugs, caught after the first full training runs. Each fix is described with its mechanism, matching the pattern in the notes section below.

1. **The analysis pipeline fed the SAE the exact input this README warned "will produce nonsense."** Training normalized activations to norm sqrt(d_model) inside the loop only, so `analysis.py` encoded raw layer-8 residuals with a model trained at ~27.7 - rankings survived (a monotone per-feature rescale) while every absolute statistic was wrong. Fix: the input scaling now lives inside the SAE (applied in `encode`/`forward`/`resample_dead_features`), so training, analysis, and evaluation share one contract by construction; `test_encode_applies_input_scale` pins it. The first version of that fix normalized every token to norm sqrt(d_model); it has since been replaced by a single dataset-wide scalar (`input_scale`, see the notes section) so reconstructions can be mapped back to raw space.
2. **The entire run happened inside LR warmup.** 500K tokens / 512-token batches is ~976 steps, against `warmup_steps=1000` - the learning rate ramped toward 2e-4 and the run ended before reaching it, so every number in the pre-fix row below was produced at a fraction of the configured LR. Fix: `warmup_steps=100`, plus a clamp-with-warning in `train_sae` so this failure mode can't recur silently.
3. **Resampling never fired, and the dead-feature window was broken.** `resample_interval=1000` > 976 total steps, so the resampling machinery was never exercised in the headline run; and `feature_activation_counts` was only zeroed when a resample actually happened, so after any all-alive checkpoint the counting window grew without bound ("dead" degraded to "never fired since step 0"). Fix: interval lowered to 250 and the counter is zeroed at every resample checkpoint, giving true fixed-window semantics.
4. **Resampling drew candidates from a single 512-token batch.** With many dead features that reinitializes near-duplicate directions from one narrow slice of data. Fix: a rolling pool of the last 8 batches' activations and per-token errors (~12 MB) feeds the sampler; `SAEOutput` now carries `per_token_recon_error` so the loop never compares the scaled-space reconstruction against raw inputs.
5. **Variance explained wasn't FVU.** The old script compared flattened `.var()` ratios (deviation from the *global scalar* mean) - close to, but not, the FVU the literature reports. `variance_explained.py` now computes 1 - FVU properly (residual sum of squares over sum of squares about the per-dimension mean), in raw residual space from `recon_raw`. But see the position-0 note below: the script still includes position 0 in its eval set, which makes its headline number look far better than it is; the corrected figure is in the table.
6. Smaller items: the checkpoint stores `asdict(config)` so `torch.load(..., weights_only=True)` works (pickling the frozen dataclass forced the unsafe flag on every consumer); `find_interesting_features` takes `n_features: int` instead of the whole SAE and returns ranked indices instead of printing (the per-feature rate / mean-when-active it used to print now come from `feature_activation_stats`, and `main.py` does the printing); a redundant re-sort after `topk` removed; the `sae_no_l1` fixture actually doesn't mutate now (`.eval()` - forward in train mode updates activation counts, so the old "doesn't mutate" comment was false); repeat data passes reshuffle document order; `feature_token_projection` documents its deliberate ln_final omission; the `trust_remote_code=True` kwarg is gone from both `load_dataset` calls (`datasets>=3` ignores it and printed a warning on every iterator restart).

## numbers

All at 500K training tokens, 3072 features (4x), layer 8, `l1_coefficient=5e-3`, on a Lambda A10 (24 GB). One row per input contract; see reporting conventions.

| run | input contract | position 0 | MSE | L0 | FVU (1 - FVU) | dead | wall / tok/s |
|---|---|---|---|---|---|---|---|
| pre-fix (items 1-5 above not yet applied) | per-token norm sqrt(d), applied in the training loop only | in train and eval | 0.45 (per-token space) | 43 | 55% VE by the pre-FVU metric (item 5) - not an FVU | 0 / 3072 (window broken, item 3) | ~32 s / ~15K |
| per-token contract, items 1-6 applied (from the run notes; never recorded here before) | per-token norm sqrt(d), inside the SAE | in train and eval | - | ~27 | 1 - FVU 0.35 | - | ~15K |
| **the first run: scalar contract** | `input_scale = 0.1896` calibrated on 102,132 tokens **including position 0** | in train, calibration and eval | 0.269 (scaled space) | 16.7 at end of training; 16.2 on the eval set | **0.683 excluding position 0** (0.933 VE = 0.067 FVU with it in - see notes; that figure is an artifact) | 0 / 3072 | 22 s / ~22.7K |
| Data path: scalar contract, position 0 excluded everywhere | pending | out of train, calibration and eval | - | - | - | - | target >= 100K on A10 |

Caveats on the first-run row, all fixed by the data-path rebuild: the eval set for the FVU/L0 columns was the first 80 documents of `NeelNanda/pile-10k` (9,882 tokens), which the training stream also walked, so it is not held out; and the calibration mean of 146 was inflated by position 0 (ordinary tokens average 121), so `input_scale` is ~20% smaller than the sqrt(d)/121 = 0.228 it should be, which loosened the effective L1 pressure and is part of why L0 came out at 16.7 rather than nearer 20. Do not quote the 0.067.

The pre-fix and per-token rows are kept for history and cannot be compared to the scalar rows: different space for MSE, different L0, and (for the pre-fix row) a different metric entirely.

## features

Five features from the first run checkpoint (top of the 0.1%-20% activation-rate band, ranked by mean activation when active). All five are legible from their top contexts and logit-lens projections; none is a confirmed result in the sense of having been checked against a second seed or a held-out corpus.

- **feature 1869** - Romance-language subword text. Fires on Spanish / Portuguese word pieces (` que`, `unta`, `ente`, `ando`, `ado`); the decoder row promotes ` é`, ` la`, `ó`, `és`, `à`.
- **feature 1854** - newline inside source code (after `;`, before `///` doc-comments, around `@end`); promotes runs of spaces and `posted` / `Posted`.
- **feature 2844** - runs of indentation whitespace in XML- / JSON-like text; promotes ` `, ` ]`, ` )`, ` |` and underscore rules.
- **feature 2219** - the curly-apostrophe UTF-8 artifact: fires on the mangled byte in `don�t`, `it�s`, `wasn�t` and promotes replacement-character byte tokens. **This is the pre-fix README's feature 2256** (below), rediscovered at a new index after retraining under a different input contract from a different random init - a small but real stability signal.
- **feature 2731** - first-person subject, predict the verb: fires on ` I` / `I` / ` i` and promotes `'ve`, ` think`, ` know`, ` believe`, ` want`, ` understand`.

The ranking key is mean-when-active, which is why token-identity features (whitespace, byte fragments) sit at the top; that is a knob in `find_interesting_features`, not a finding.

For history, the three features written up from the **pre-fix** checkpoint - identified under the mis-scaled analysis path (bug 1) and never re-verified: **679**, a sentence-boundary / discourse-connective detector (fires on `.` + newline; promotes *However*, *Furthermore*, *Moreover*); **2107**, a Q&A-format detector (fires on the newline after `Q:`; promotes *What*, *How*, *Why*); and **2256**, the UTF-8 encoding-artifact detector that reappeared as 2219 above.

## notes

**Position 0 is a 3141-norm outlier and must be excluded by position, not by id.** At layer 8, the residual at position 0 (the BOS / attention-sink position) has mean norm 3141; every other position averages 121 - a 26x gap - and the position-0 vector is nearly constant across sequences. Three consequences, all found in the first run run. (a) In raw space it carries roughly 90% of the sum of squares about the per-dimension mean, and the SAE reconstructs it almost perfectly (near-constant vector = `b_dec` plus one feature), so FVU over all positions came out at 0.067 while FVU on the same checkpoint with position 0 removed was 0.683 - the first number is a metric artifact. (b) It inflated the calibration mean (146 vs 121), so `input_scale` was ~20% too small and the L1 pressure correspondingly loose. (c) In scaled space it has norm ~600 against ~23 for ordinary tokens, so a non-trivial share of the loss, the gradient, and the dictionary went to reconstructing a constant. The exclusion must be by *position*: packed sequences also carry id 50256 as EOS between documents, and those mid-sequence tokens are ordinary (a `tokens != pad_id` filter is exactly the mistake described further down). The analysis cache has stripped position 0 since the first analysis pass (see the last note in this section); as of the first run the training buffer, calibration sample and eval set do not. the data path drops it everywhere and passes it through untouched when splicing. This also corrects a number this README repeated for a long time: "layer 8 residuals have norm ~150" was the mean *with* position 0; ordinary tokens are ~120.

`find_max_activating_examples` uses `searchsorted` to map a flat token index back to (text, position). Default `right=False` is off-by-one when a peak lands exactly on a text boundary - text k's first token gets attributed to text k-1. Caught by spot-checking analysis output. Regression in `test_peak_at_text_boundary`.

Dead-feature resampling kept producing features that died again on the next step. AdamW's first and second moments still carried momentum from when the feature had been alive (and then went dead), and that momentum dragged the reinitialized direction back toward zero on the very first step. Fix is `_zero_optim_state`; covered by `test_resample_zeros_optimizer_state`.

The decoder is constrained to unit-norm rows. Naive approach: renormalize after each optimizer step. That fights the optimizer - every step nudges rows off the unit sphere and we yank them back. Cleaner: project the component of `grad(W_dec)` parallel to each row out before stepping, so the optimizer never moves rows off the sphere in the first place (Anthropic, *Towards Monosemanticity*). Adam's element-wise update can still produce small drift, so I kept a renormalize call every 100 steps as a safety net. `project_decoder_grad` and `test_grad_perpendicular_after_projection`.

The encoder/decoder are tied at init: `W_dec` is random unit-norm rows, `W_enc = W_dec.T`. The earlier version used `kaiming_uniform_` on `W_enc`, which infers `fan_in` from `tensor.size(1)` - but here the actual fan-in is `size(0)`, so the init scale was wrong on the wrong axis. Tied init sidesteps the issue and is what the SAE reference implementations do anyway.

Earlier version of the training loop filled an 8-chunk buffer (~30K tokens), sampled 512 tokens per step, and discarded the rest. About 1.5% utilization. Replaced with a 64K-token shuffled buffer walked through batch-by-batch, refilled when a batch wouldn't fit. ~100% now. (Remaining simplification: the buffer is walked fully then refilled, so consecutive batches come from one shuffled corpus window; refill-at-half-with-reshuffle would decorrelate further.)

GPT-2's pad token id equals BOS/EOS, so a `tokens != pad_id` filter would also drop legitimate BOS positions. The activation source tokenizes once with HF's tokenizer (real attention mask), prepends BOS, then uses the mask to drop pad-position activations in one boolean-index op over the `(B, T, d_model)` activation tensor.

First training run came out with L0=866 - the SAE was using a quarter of its 3072 features on every token. Ordinary layer-8 residuals have norm ~120 (~150 if you average position 0 in), so the L1 penalty `l1_coefficient * sum(|h_i|)` was tiny relative to MSE and sparsity wasn't being optimized. Fix: scale the inputs so the L1 coefficient is decoupled from the layer's activation scale. The SAE now carries one dataset-wide scale factor, `input_scale = sqrt(d_model) / mean ||x||`, computed once from a calibration sample (`set_input_scale_from_activations`; `train_sae` does this on its first buffer fill, >= 100K tokens by default) and stored as a buffer in the checkpoint. `preprocess(x) = x * input_scale` is linear - no per-token normalization, no clamping - so norm ratios between tokens survive, and `postprocess` divides the scale back out: `forward` returns both `recon_scaled` (the space the loss lives in) and `recon_raw`, which can be spliced straight back into the model's residual stream. An earlier version normalized *each token* to norm sqrt(d_model); that equalised per-token norms, which is information the residual stream carries, and made reconstructions unmappable to raw space. After scaling and bumping `l1_coefficient` from 8e-4 to 5e-3, L0 settled at ~43 under the per-token contract and ~17 under the scalar one (see known issues). The scaling originally lived in the training loop only - which is how the analysis-path bug (item 1 above) happened; it now lives inside the SAE, where the checkpoint's input contract belongs. A checkpoint written before `input_scale` existed loads with a warning and `input_scale = 1.0`.

The first analysis pass had every "interesting" feature firing on `<|endoftext|>` because BOS has an outlier activation that monopolises every per-feature top-k. Stripping it from the analysis corpus (one slice in `build_activation_cache`) was enough to surface the features above. That was the same 3141-norm position-0 outlier described at the top of this section, seen from the analysis side.

## known issues

L1 SAEs are not state of the art anymore. JumpReLU (DeepMind) and top-k (Gao et al, OpenAI) both do better on the L0/MSE frontier. I picked L1 because it's the simplest thing that demonstrates the rest of this; in production I'd reach for top-k. Variance explained at this token budget reflects that - vanilla L1 at 500K tokens isn't competitive with current methods, but both training longer and switching the encoder are addressable.

`l1_coefficient=5e-3` was tuned under the per-token contract, where it gave L0 ~27-43. Under the scalar contract it gives L0 ~17 (and the data path calibration change - position 0 out of the mean - will shift it again). Getting back to a target L0 for an apples-to-apples row needs a small sweep, cheap at ~22 s per run on the A10; not done yet.

Position 0 is still in the training buffer, the calibration sample, and `variance_explained.py`'s eval set (see the first note above). Being fixed in the data path together with the data pipeline; until then, the FVU that script prints is not quotable and the table carries the corrected figure.

The analysis corpus and the FVU eval set come from `load_dataset` text, not from a held-out shard, so neither is document-disjoint from training by construction (the first run eval set was in fact the head of the training stream). the data path introduces a document-disjoint holdout shard; moving `build_activation_cache` onto it is a follow-up because it needs strings for `token_strings` and packed windows re-tokenized with `prepend_bos` would double-BOS.

Layer 8 is a guess from probing literature on GPT-2-small. A real choice would grid-search 6-10 at fixed token budget and pick the layer with the lowest reconstruction at a target L0.

Single-process: model forward, SAE forward, and optimizer step run sequentially, and text is tokenized and activations shuffled on the CPU on the fly - ~22.7K tok/s on an A10, GPT-2-forward-bound (the SAE step itself is ~2.6 ms per 512 tokens, ~200K tok/s, per `profile_sae.py`). the data path replaces this with a pre-tokenized shard and a GPU-resident shuffle buffer.

In-memory feature cache. `(total_tokens, n_features)` in RAM is fine for 100 texts. The layout (flat token array plus per-text offsets) maps cleanly onto Parquet sharded by feature index when this becomes the bottleneck.

Cosmetic, visible in the logs: `HookedTransformer.from_pretrained` is deprecated in transformer_lens 3.x in favour of `TransformerBridge` (the hook name is unaffected); and `training_history.png` shows the dead-feature count spiking to 3072 at steps 250/500/750 - the count is logged right after the fixed-window counter is zeroed at a resample step, so that is a logging-order artifact, not features dying.
