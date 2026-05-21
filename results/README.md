# results/

Every `results/<run>/` (and `results/<sweep>/<run>/`) directory is written by
`main.py`: `config.json`, `train_log.jsonl`, `training_history.png`, and
`metrics.json` (config, held-out metrics, provenance). Nothing in this file is
computed here - every number below names the `metrics.json` it was read from,
and `python -m eval --checkpoint <run>/checkpoint.pt --holdout data/holdout.bin
--n-tokens 2000000 --compare <run>/metrics.json` regenerates any of them from
the checkpoint (the reproducibility contract, README top level).

## MVP: single 200M-token run (L1, 8x) with held-out FVU/CE

**What ran.** One ReLU + L1 SAE, `l1_coefficient` 5e-3 (the frontier's mid
lambda), 6144 features (8x), on GPT-2-small `blocks.8.hook_resid_post`,
200,003,584 training tokens (48,828 optimizer steps at batch 4096) from the
220M-token `monology/pile-uncopyrighted` shard (`data/train.bin`, docs
20,000-151,072), lr 4e-4 (warmup 1000 steps and resample every 5000 steps,
pinned by `sweeps/mvp.json` to the frontier's schedule), AdamW betas
(0.9, 0.99), seed 42, 2M-row on-device shuffle buffer, `--forward hf`,
`input_scale` 0.22823 calibrated on the first >= 100K position-0-free tokens.
Evaluated with `eval.evaluate` on 2,007,616 tokens of the document-disjoint
held-out shard (`data/holdout.bin`, docs 0-20,000; fp32 forwards, position 0
excluded).

**Hardware.** One NVIDIA A10 (24 GB), torch 2.7.0, Training wall
35 min 31 s at 93,184 tok/s end-to-end (GPT-2 forward + SAE step, 200 buffer
refills); checkpoint written 2026-05-20; the 2M-token eval ran
at 9.66K tok/s (~3.5 min, three fp32 forwards per batch).

**The numbers** (all from `results/mvp/metrics.json`, `metrics` block;
`git_sha` is null in that record - provenance is the `config` block in the
same file, and `python -m eval` re-derives every row from the checkpoint):

| metric | value |
|---|---|
| 1 - FVU (raw space, held out) | **0.7254** (FVU 0.2746) |
| loss recovered | **0.9539** |
| CE clean / recon / zero (nats) | 3.8590 / 4.3182 / 13.8133 |
| L0 (held out) | 14.1 |
| dead features at eval | 79 / 6144 (1.29%) |
| MSE (raw / scaled) | 2.9812 / 0.1553 |
| final training loss / L0 | 0.2701 / 14.1 |

**Kill point: PASS.** 1-FVU 0.7254 >= 0.65 and loss recovered 0.9539 >= 0.85,
with the pipeline gate re-verified on this box the same day
(`bench_pipeline.py`: 141-143K tok/s loader steady state, 95.1K loader + SAE
step at 8x, 110.9K at 4x >= the 100K A10 gate - within noise of the recorded
default-recipe numbers, 95.2K / 111.7K). No diagnosis checklist required.

**Prediction check.** Before this run was recorded, this section sourced
`results/a6_default/metrics.json` - the identical configuration under the
*derived* schedule (warmup 976 / resample 6103) - and predicted the MVP would
match it to < 0.001 of 1-FVU. Measured: 0.7254 vs 0.7249 (delta 0.0005,
inside the seed-replicate noise band), loss recovered 0.9539 vs 0.9533, L0
14.1 vs 14.01, dead 79 vs 81. The pinned-vs-derived schedule equivalence the recipe commit
claimed is now measured on the MVP run itself, and both reproduce the
frontier's `relu_x8_l1-mid` cell (0.725 / 0.954 / L0 14.0). Two more
contract checks passed in the same run: `python -m eval --checkpoint
results/mvp/checkpoint.pt` reproduced every number above from the checkpoint
exactly, and the shards this run trained on were re-tokenized from scratch
that morning (190 s) with sidecars identical to the ones recorded inside
every earlier run's `metrics.json` (train docs [20000, 151072], 1,718,750
seqs; holdout docs [0, 20000], 252,754 seqs) - shard regeneration is
deterministic, as documented.

**Context.** This is the L1 baseline at the default recipe, not the best run
in this directory: the frontier's TopK rows beat it at every width
(`results/frontier/`, `figures/tables.md`, `results/SUMMARY.md` - best 8x run
under L0 <= 40 is `topk_x8_k32` at 1-FVU 0.835 / loss recovered 0.985). The
top-of-repo README headline carries the frontier's measured values (verified
against `results/frontier/topk_x4_k32/metrics.json` and
`results/frontier/topk_x16_k64/metrics.json`); this run does not change it.

## Directory map

- `mvp/` - the run above (MVP: `sweeps/mvp.json` through `main.py`;
  commit its `config.json` + `metrics.json`).
- `a6_default/` - the default-recipe check: the identical configuration
  under the derived schedule, which the prediction check above measures
  against.
- `frontier/` - the 18-run frontier (3 lambda + 3 k, each at 4x / 8x / 16x,
  200M tokens each; `index.jsonl` has status and wall time per run). Per run,
  `config.json`, `metrics.json` and `sweep_entry.json` are committed;
  `train_log.jsonl`, `train.log`, `training_history.png` and `checkpoint.pt`
  are gitignored regenerated artifacts that live on the machines only.
- `seeds/` - seed replicates (seeds 1 and 2) of `topk_x4_k32` and
  `relu_x4_l1-low`.
- `a4-topk32/` - the TopK-comparison record (500K tokens, TopK k=32, 1-FVU 0.622 /
  loss recovered 0.924): pre-frontier history, the only machine-readable
  record of the TopK comparison, not part of any sweep. (Smoke-sweep output
  and two duplicate timestamp-named 500K-run records were deleted in the results
  cleanup - SUMMARY.md, restore round-trip.)
- `SUMMARY.md` - best config per width, the frontier read, anomalies
  (incl. measured resample churn from the train logs), and the
