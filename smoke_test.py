"""CPU smoke test of the whole training path: build two tiny document-disjoint
shards from in-memory text, train a small SAE through `main.main` for 2000 tokens
(real GPT-2, TransformerLens forward, derived schedule)"""

import tempfile
from pathlib import Path

import eval as eval_mod
import main
from data import TokenShard, tokenize_corpus

SEQ_LEN = 64
DOCS = [
    f"Smoke test document {i}. The quick brown fox jumps over the lazy dog "
    f"while the sparse autoencoder learns a dictionary of {i * 7} features. "
    * (2 + i % 3)
    for i in range(48)
]

tmp = Path(tempfile.mkdtemp())
tokenize_corpus(
    DOCS,
    out_path=str(tmp / "train.bin"),
    n_tokens=96 * SEQ_LEN,
    seq_len=SEQ_LEN,
    holdout_docs=8,
    holdout_path=str(tmp / "holdout.bin"),
)
train, holdout = TokenShard(tmp / "train.bin"), TokenShard(tmp / "holdout.bin")
print(
    f"shards: train {train.n_seqs} seqs x {train.seq_len} (docs "
    f"{train.meta['doc_range']}), holdout {holdout.n_seqs} seqs (docs "
    f"{holdout.meta['doc_range']})"
)

results = tmp / "results"
checkpoint = results / "smoke" / "checkpoint.pt"
# Small everything; warmup / resampling use the derived defaults.
main.main([
    "--train-shard", str(train.path), "--holdout-shard", str(holdout.path),
    "--run-name", "smoke", "--results-dir", str(results),
    "--checkpoint", str(checkpoint), "--no-analysis", "--forward", "tl",
    "--device", "cpu", "--expansion", "1", "--n-tokens", "2000",
    "--batch-tokens", "128", "--buffer-tokens", "1024", "--batch-seqs", "8",
    "--calibration-tokens", "512", "--eval-tokens", "500",
    "--eval-batch-seqs", "8", "--log-interval", "2",
])

metrics_json = results / "smoke" / "metrics.json"
print("\n--- python -m eval --compare (must reproduce metrics.json) ---")
# --n-tokens / --batch-seqs come from the checkpoint; exits 1 on mismatch.
eval_mod.main([
    "--checkpoint", str(checkpoint), "--holdout", str(holdout.path),
    "--compare", str(metrics_json), "--device", "cpu",
])
print("\nsmoke test passed: main.py trained, python -m eval reproduced metrics.json")
