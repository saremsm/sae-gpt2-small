# Sweep summary

Sources: `results/frontier/index.jsonl` (18 runs, status + wall),
`results/frontier/<run>/metrics.json` (every metric), `results/seeds/` (4 seed
replicates), `results/mvp/metrics.json` (the MVP run) and
`results/a6_default/metrics.json` (the default-recipe check it reproduces),
`figures/tables.md` (the same numbers tabulated by `plot.py`; regenerated from
`results/frontier/` and confirmed byte-identical to the checked-in file).
Nothing here is recomputed - every number names its file. Hardware: one
NVIDIA A10; the frontier and seed `metrics.json` files predate the provenance schema
so they do not carry a `gpu_name` field - the A10 attribution is from the
sweep run (`index.jsonl` commands/timestamps, README frontier section);
`results/a6_default/metrics.json` records `NVIDIA A10`, torch 2.7.0 explicitly.

## Best config per width (L0 <= 40, `figures/tables.md` best-per-width table)

| width | best run | 1-FVU | loss recovered | L0 | source |
|---|---|---|---|---|---|
| 4x (3072) | `topk_x4_k32` | 0.808 | 0.981 | 32.0 | `results/frontier/topk_x4_k32/metrics.json` |
| 8x (6144) | `topk_x8_k32` | 0.835 | 0.985 | 32.0 | `results/frontier/topk_x8_k32/metrics.json` |
| 16x (12288) | `topk_x16_k32` | 0.856 | 0.988 | 32.0 | `results/frontier/topk_x16_k32/metrics.json` |

k=32 wins both metrics at every width under the cap (it is the only run just
under L0 40). Without the cap the k=64 rows win trivially: 0.846 / 0.988,
0.868 / 0.990, and 0.885 / 0.992 at 4x / 8x / 16x - the last
(`topk_x16_k64/metrics.json`) is the best point without the cap.

## The frontier, in two sentences

TopK dominates ReLU + L1 at every width and sparsity: at matched L0 it is
0.04-0.08 of 1-FVU ahead (k=16 at L0 16: 0.765-0.819 vs lambda 5e-3 at L0
13-15: 0.688-0.753; k=32 at L0 32: 0.808-0.856 vs lambda 2.5e-3 at L0 35-37:
0.764-0.821), and at half the L0 it matches L1 (`topk_x4_k16` 0.765 at L0 16
vs `relu_x4_l1-low` 0.764 at L0 34.9). Width helps monotonically and by about
the same step everywhere - ~0.02-0.03 of 1-FVU per doubling at every k
(k=32: 0.808 -> 0.835 -> 0.856; k=64: 0.846 -> 0.868 -> 0.885) - with no sign
of saturation at 16x.

## Anomalies

- **Failed runs: none.** 18 / 18 frontier runs `status: ok`
  (`results/frontier/index.jsonl`), 4 / 4 seed runs ok
  (`results/seeds/index.jsonl`). Total frontier wall 792.6 min = 13 h 13 min
  (span 2026-05-08 21:32:06 -> 2026-05-09 10:44:40 UTC); seeds 138.6 min.
- **lambda = 1e-2 is the collapse regime, and the resampler measurably never
  touched it.** L0 4.3-4.4 with 37.0% / 43.3% / 44.3% of features dead on the
  2M held-out tokens at 4x / 8x / 16x (`relu_x*_l1-high/metrics.json`:
  `dead_frac_eval`), 1-FVU 0.49-0.50. Measured from the per-run
  `train_log.jsonl` files now present in the tree (restored from the GPU host;
  HEAD's `.gitignore` deliberately ignores them as regenerated artifacts -
  dead count at the last logged step of each
  5000-step window = features resampled at that checkpoint):
  `relu_x4_l1-high` resampled 1 feature across all 9 checkpoints,
  `relu_x8_l1-high` 3, so the thousands of eval-dead features fired just
  often enough in each 20.5M-token counting window to dodge the resampler
  and still never fire on 2M held-out tokens - the eval-dead mass is
  window-alive by measurement, not inference. Resample churn (a feature
  resampled repeatedly) is therefore ~zero everywhere: 19 of 22 runs
  (all TopK, all seeds, all l1-low/l1-mid except x16) resampled 0 features
  at every checkpoint.
- **`relu_x16_l1-mid` is the one run where resampling was actually working,
  and it was losing.** 77 features resampled in total, accelerating through
  training (0, 0, 0, 0, 2, 1, 8, 30, 36 per checkpoint,
  `results/frontier/relu_x16_l1-mid/train_log.jsonl`), against 835 (6.8%)
  dead at eval - features were dying faster than the 5000-step window could
  flag them. Together with the width trend at fixed lambda (5 -> 64 -> 835
  eval-dead at 4x / 8x / 16x), this is the concrete thing to fix before
  training L1 wider: a shorter counting window or AuxK-style pressure.
- **AuxK did engage, transiently.** The TopK runs' `aux_loss` column is
  nonzero on a small share of logged rows well inside counting windows
  (not just the one-step-after-reset artifact): `topk_x16_k16` 94 of 8,138
  rows more than 10 steps past a reset (max 0.79), `topk_x16_k64` 31,
  `topk_x4_k32` 22. So features do momentarily go window-dead under TopK
  and AuxK pulls them back - consistent with 0 resamples on every TopK run
  and only `topk_x16_k16` keeping any eval-dead features (169, 1.4%). A
  pure-TopK ablation is still needed to attribute that survival to AuxK.
- **The L1 bracket landed low** (README, frontier section): L0 35 / 13-15 / 4.3 instead of
  the targeted 15-60, so the L1 curve covers L0 4-37. The fill-in sweep
  (`sweeps/frontier_fill.json`, lambda 1.6e-3 and 3.5e-3 per width) has NOT
  run - no such runs exist under `results/frontier/`.
- **Throughput falls with width** (expected; the SAE step becomes the
  bottleneck at 16x): ReLU 102.6K -> 88.0K -> 68.3K tok/s, TopK 91.0K ->
  73.9K -> 53.2K at 4x / 8x / 16x (`training.loader_tok_s` per run).
- **Seed noise is tiny.** `topk_x4_k32` 1-FVU 0.8077 / 0.8076 / 0.8072
  (seeds 42 / 1 / 2, range 0.0005), loss recovered 0.9809 / 0.9810 / 0.9810;
  `relu_x4_l1-low` 0.7643 / 0.7638 / 0.7645, loss recovered 0.9686 / 0.9685 /
  0.9679, L0 34.85-34.99 (`results/frontier/*` + `results/seeds/*`). Every
  gap the readings above lean on is 30-100x that. The one near-tie -
  `relu_x16_l1-low` 0.821 vs `topk_x16_k16` 0.819 - is ~3x the noise: read as
  "about equal".
- **Restore round-trip.** The working tree briefly became the
  union of every state that ever existed (PC rm -rf'd -> restored from the
  box, which never had deletions applied -> box rm -rf'd -> restored from
  the workstation). Resurrected and re-deleted in the cleanup: an older
  `.gitignore` that had overwritten HEAD's (restoring HEAD's re-hides the
  per-run `train_log.jsonl` / `train.log` files), `variance_explained.py`
  (deleted in the eval.py commit, README known-issues; nothing imports it),
  two byte-identical timestamp-named copies of the 500K-run record
  (`results/20260816-*`; its numbers live in the README history table),
  the 200K-token smoke-sweep output (`results/smoke/`, `figures/smoke/`;
  1-FVU negative by design, regenerable in minutes), and three GPU-host run
  logs. Kept and newly committed: `results/a4-topk32/` (the only
  machine-readable TopK-comparison record) and the frontier runs' `sweep_entry.json`
  (parity with the seeds runs, which already committed theirs). The checkpoints
  and token shards survived on the machines (gitignored, and excluded only
  from the archived copy - see below); nothing irreplaceable was lost.

## Key numbers and source files

| claim | exact value | source file |
|---|---|---|
| 18 runs, all ok | 18 lines, `status: ok` | `results/frontier/index.jsonl` |
| 13.2 h on one A10 | 792.6 min total wall (span 13 h 12 min 34 s) | `results/frontier/index.jsonl` (sum of `wall_min` / timestamps) |
| 200M tokens per run | 200,003,584 | any `results/frontier/<run>/metrics.json`, `training.tokens` |
| 1-FVU 0.885 | 0.8852518672 | `results/frontier/topk_x16_k64/metrics.json`, `metrics.variance_explained` |
| loss recovered 0.992 | 0.9924200015 | same file, `metrics.loss_recovered` |
| CE 3.86 -> 3.93 vs 13.81 | 3.8589 / 3.9343 / 13.8135 | same file, `metrics.ce_clean` / `ce_recon` / `ce_zero` |
| L0 64, held out | 64.0 on 2,007,616 tokens | same file, `metrics.l0` / `metrics.n_tokens` |
| seed variance < 0.001 | 1-FVU range 0.0005 (topk), 0.0007 (relu) | `results/frontier/topk_x4_k32/metrics.json`, `results/seeds/{topk_x4_k32,relu_x4_l1-low}_s{1,2}/metrics.json` |


## figures/tables.md

Confirmed: the checked-in `figures/tables.md` is byte-identical to
`python plot.py --results results/frontier --out <dir>` output (verified on
this revision). After the `plot.py` fix (recursive discovery), the
documented `python plot.py --results results/ --out figures/` folds in
everything under `results/` - on the cleaned tree that is 24 evaluated
runs (25 once `results/mvp` is committed): 18 frontier + 4 seeds +
`a6_default` + `a4-topk32` (before the fix it
silently plotted only `a6_default`). Use `--results results/frontier` for
the frontier figures (what `figures/` holds); the new `--min-tokens` flag
(e.g. `--min-tokens 1000000`) keeps sub-frontier rows - the 500K-token
`a4-topk32`, and any smoke/debug runs, whose 1-FVU can be negative and
wreck the y-axis - off an everything-view plot.

## Missing / newly recorded metrics files

- `results/mvp/metrics.json` - RECORDED (GPU host, 2026-05-20):
  1-FVU 0.7254 / loss recovered 0.9539 / L0 14.1 / 79 dead - within 0.0005
  of `a6_default`, exactly as predicted (`results/README.md`, prediction
  check). Copy `results/mvp/{config.json, metrics.json}` off the GPU host and
  commit them (the run's `train_log.jsonl` / `train.log` /
  `training_history.png` / `checkpoint.pt` stay local under the one-level
  `.gitignore` patterns).
- Checkpoints (`checkpoint.pt`, `sae_gpt2_layer8.pt`) - gitignored (`*.pt`)
  by design and excluded from the archived copy for size, but PRESENT on the
  machines: `python -m eval --compare` reproduction remains available, and
  was exercised on the MVP checkpoint the day it was written (numbers
  reproduced exactly).
- `data/train.bin` / `data/holdout.bin` - gitignored (`data/`), regenerated
  from scratch on the GPU host in 190 s; the fresh sidecars matched the ones
  recorded inside every earlier run's `metrics.json` (doc ranges and seq
  counts identical) - deterministic shard regeneration is verified, not
  just documented.
- The per-run `train_log.jsonl` / `train.log` / `training_history.png` ARE
  on disk (restored from the GPU host) but intentionally gitignored - they are
  what made the resample-churn and AuxK measurements above possible. If
  those measurements should stay reproducible from the repo, lift the
  ignore for `train_log.jsonl` only (~51 MB total across 23 runs) and
  commit; never commit `train.log` (~41 MB of tqdm carriage returns).
- No `sweeps/frontier_fill.json` results (the 6 fill-in runs have not run).
