"""CPU smoke test: build a tiny shard from in-memory text, run GPT-2 through the
ActivationLoader, train a small SAE for 2000 tokens."""

import tempfile
from pathlib import Path

from transformer_lens import HookedTransformer

from data import ActivationLoader, TokenShard, resid_post_hook, tokenize_corpus
from sparse_autoencoder import SAEConfig, SparseAutoencoder
from training import train_sae

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
    n_tokens=64 * SEQ_LEN,
    seq_len=SEQ_LEN,
)
shard = TokenShard(tmp / "train.bin")
print(f"shard: {shard.n_seqs} seqs x {shard.seq_len}")

model = HookedTransformer.from_pretrained("gpt2")
loader = ActivationLoader(
    model, shard, resid_post_hook(8),
    batch_seqs=8, batch_tokens=256, buffer_tokens=2048, device="cpu",
    log_every=0,
)
cfg = SAEConfig(d_model=768, n_features=256, l1_coefficient=8e-4, warmup_steps=10)
sae = SparseAutoencoder(cfg)
history = train_sae(
    sae, loader, n_training_tokens=2000, log_interval=1, device="cpu",
    calibration_tokens=1024,
)

if history["loss"]:
    print("smoke test passed, final loss:", history["loss"][-1])
    print(f"loader: {loader.throughput_tok_s():,.0f} tok/s on CPU")
else:
    print("Training ran but produced no log entries - increase n_training_tokens or lower log_interval")
