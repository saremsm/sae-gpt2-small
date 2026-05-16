"""benchmark the training pipeline: GPT-2 forward + activation buffer + SAE step.
That is ~5% under the 100K gate, which was calibrated at 4x (the frontier sweep:
102.6K ReLU 4x, 91K TopK 4x); the two SAE matmuls are twice as large at 8x."""

from __future__ import annotations

import argparse
import sys
import time

import torch
from torch.optim import AdamW

from data import ActivationLoader, TokenShard, load_forward_model, resid_post_hook
from sparse_autoencoder import SAEConfig, SparseAutoencoder
from training import BATCH_SIZE, train_step

WARMUP_STEPS = 20
UTIL_SAMPLE_SECONDS = 0.5


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _gpu_utilization(device: str) -> int | None:
    if torch.device(device).type != "cuda":
        return None
    try:
        return int(torch.cuda.utilization(device))
    except Exception:  # pynvml missing or unsupported
        return None


def make_loader(
    model,
    shard: TokenShard,
    args: argparse.Namespace,
    device: str,
    profile: bool,
) -> ActivationLoader:
    return ActivationLoader(
        model=model,
        shard=shard,
        hook_name=resid_post_hook(args.layer),
        batch_seqs=args.batch_seqs,
        batch_tokens=args.batch_tokens,
        buffer_tokens=args.buffer_tokens,
        device=device,
        seed=args.seed,
        autocast=not args.no_autocast,
        log_every=0,
        profile=profile,
    )


class Window:
    """One timed benchmark window: rows yielded."""

    def __init__(self, yielded: int, forwarded_yieldable: float, seconds: float):
        self.yielded = yielded
        self.forwarded_yieldable = forwarded_yieldable
        self.seconds = seconds

    @property
    def raw_tok_s(self) -> float:
        return self.yielded / self.seconds if self.seconds > 0 else 0.0

    @property
    def forward_tok_s(self) -> float:
        """Yieldable rows forwarded per second."""
        return self.forwarded_yieldable / self.seconds if self.seconds > 0 else 0.0


def _yieldable_fraction(loader: ActivationLoader) -> float:
    """Share of forwarded tokens the loader yields (position 0 dropped)."""
    seq_len = loader.shard.seq_len
    return (seq_len - 1) / seq_len if loader.exclude_bos else 1.0


def bench_loader(loader: ActivationLoader, n_tokens: int, device: str) -> Window:
    """The loader alone over n_tokens yielded rows, timed from AFTER the first yield
    (the initial buffer fill is outside the window)."""
    it = iter(loader)
    next(it)
    _sync(device)
    fwd0 = loader.tokens_forwarded
    t0 = time.perf_counter()
    n = 0
    for batch in it:
        n += batch.shape[0]
        if n >= n_tokens:
            break
    _sync(device)
    seconds = time.perf_counter() - t0
    fwd = (loader.tokens_forwarded - fwd0) * _yieldable_fraction(loader)
    return Window(n, fwd, seconds)


def bench_loader_and_sae(
    loader: ActivationLoader,
    sae: SparseAutoencoder,
    n_tokens: int,
    device: str,
) -> tuple[Window, float | None]:
    """Loader + SAE train step over n_tokens rows (after WARMUP_STEPS untimed
    steps): the timed Window plus mean sampled GPU utilization. main() turns the
    window into the steady-state number using the loader's forward rate from pass"""
    optimizer = AdamW(
        sae.parameters(), lr=sae.config.lr, betas=sae.config.adam_betas,
        weight_decay=0.0,
    )
    sae.train()
    it = iter(loader)

    first = next(it)
    sae.set_input_scale_from_activations(first)
    train_step(sae, optimizer, first)
    for _ in range(WARMUP_STEPS - 1):
        train_step(sae, optimizer, next(it))
    _sync(device)

    utils: list[int] = []
    next_sample = time.perf_counter() + UTIL_SAMPLE_SECONDS
    fwd0 = loader.tokens_forwarded
    t0 = time.perf_counter()
    n = 0
    for batch in it:
        train_step(sae, optimizer, batch)
        n += batch.shape[0]
        now = time.perf_counter()
        if now >= next_sample:
            u = _gpu_utilization(device)
            if u is not None:
                utils.append(u)
            next_sample = now + UTIL_SAMPLE_SECONDS
        if n >= n_tokens:
            break
    _sync(device)
    seconds = time.perf_counter() - t0
    fwd = (loader.tokens_forwarded - fwd0) * _yieldable_fraction(loader)
    mean_util = sum(utils) / len(utils) if utils else None
    return Window(n, fwd, seconds), mean_util


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", default="data/train.bin")
    parser.add_argument("--n-tokens", type=int, default=5_000_000)
    parser.add_argument("--min-tok-s", type=float, default=0.0)
    parser.add_argument("--batch-seqs", type=int, default=256)
    parser.add_argument("--batch-tokens", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer-tokens", type=int, default=2_000_000)
    parser.add_argument("--n-features", type=int, default=6144,
                        help="dictionary size (default 6144 = 8 x 768, the "
                        "default expansion)")
    parser.add_argument("--activation", choices=["relu", "topk"], default="relu",
                        help="encoder variant of the benched SAE step")
    parser.add_argument("--k", type=int, default=32,
                        help="latents per token under --activation topk")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-autocast", action="store_true",
        help="run the GPT-2 forward in fp32 instead of bf16 autocast",
    )
    parser.add_argument(
        "--forward", choices=["tl", "hf"], default="hf",
        help="GPT-2 implementation the loader runs: TransformerLens "
        "HookedTransformer (tl) or HuggingFace GPT2Model with SDPA (hf); "
        "same residual either way (data.HFResidualModel)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: cpu (no CUDA; numbers are not the target)")

    shard = TokenShard(args.shard)
    print(
        f"Shard: {args.shard} ({shard.n_seqs:,} seqs x {shard.seq_len}); "
        f"batch_seqs={args.batch_seqs}, batch_tokens={args.batch_tokens}, "
        f"buffer_tokens={args.buffer_tokens:,}, "
        f"autocast={'off' if args.no_autocast else 'bf16'}, "
        f"forward={args.forward}, sae={args.n_features} features "
        f"{args.activation}" + (f" k={args.k}" if args.activation == "topk" else "")
    )

    model = load_forward_model(args.forward, device)

    # Pass 1: loader alone, with the read / forward split measured.
    loader = make_loader(model, shard, args, device, profile=True)
    w1 = bench_loader(loader, args.n_tokens, device)
    loader_tok_s = w1.forward_tok_s
    busy = loader.read_seconds + loader.forward_seconds
    read_pct = 100.0 * loader.read_seconds / busy if busy > 0 else 0.0
    print(
        f"\nLoader alone        : {loader_tok_s:,.0f} tok/s steady state "
        f"(yieldable rows forwarded per second; raw window {w1.raw_tok_s:,.0f} "
        f"tok/s over {w1.yielded:,} yielded / {w1.forwarded_yieldable:,.0f} "
        f"forwarded in {w1.seconds:.1f}s - the difference is the buffer's "
        f"borrowed top half; {loader.n_chunks} forwards of {args.batch_seqs} "
        f"seqs, {loader.n_refills} refills)"
    )
    print(
        f"  shard read + H2D  : {read_pct:.1f}% of loader busy time "
        f"({loader.read_seconds:.2f}s vs {loader.forward_seconds:.2f}s "
        f"GPT-2 forward) - pinned memory only pays if this is large"
    )
    if loader.n_chunks:
        # Effective matmul throughput of the forward: 2 FLOP per weight per token over
        # the blocks actually run (12 d_model^2 weights per block: 4 attention + 8
        ms = 1000.0 * loader.forward_seconds / loader.n_chunks
        tokens_per_fwd = loader.tokens_forwarded / loader.n_chunks
        block_params = 12 * model.cfg.d_model ** 2 * (args.layer + 1)
        tflops = 2 * block_params * tokens_per_fwd / (ms / 1000.0) / 1e12
        print(
            f"  per forward       : {ms:.1f} ms for {tokens_per_fwd:,.0f} tokens "
            f"through blocks 0-{args.layer} (~{tflops:.1f} TFLOP/s effective)"
        )

    # Pass 2: fresh loader (no profiling syncs) + fresh SAE, real step.
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    loader = make_loader(model, shard, args, device, profile=False)
    is_topk = args.activation == "topk"
    cfg = SAEConfig(
        d_model=model.cfg.d_model, n_features=args.n_features,
        l1_coefficient=0.0 if is_topk else SAEConfig.l1_coefficient,
        activation=args.activation, k=args.k if is_topk else None,
    )
    sae = SparseAutoencoder(cfg).to(device)
    w2, util = bench_loader_and_sae(loader, sae, args.n_tokens, device)
    # Steady state: the window's time minus what its forwards cost at the
    if loader_tok_s > 0 and w2.yielded > 0:
        step_seconds = max(0.0, w2.seconds - w2.forwarded_yieldable / loader_tok_s)
        s_per_tok = step_seconds / w2.yielded
        tok_s = 1.0 / (1.0 / loader_tok_s + s_per_tok)
    else:
        s_per_tok = 0.0
        tok_s = w2.raw_tok_s
    print(
        f"Loader + SAE step   : {tok_s:,.0f} tok/s steady state "
        f"(raw window {w2.raw_tok_s:,.0f} tok/s over {w2.yielded:,} rows in "
        f"{w2.seconds:.1f}s; batch {args.batch_tokens}, {args.n_features} "
        f"features, {args.activation})"
    )
    print(
        f"  SAE step          : {1e6 * s_per_tok:.2f} us/token = "
        f"{1e3 * s_per_tok * args.batch_tokens:.1f} ms per "
        f"{args.batch_tokens}-token step (window time minus its forwards "
        f"at the loader's rate)"
    )
    if util is None:
        print(
            "  GPU utilization   : unavailable (CPU run, or "
            "`pip install nvidia-ml-py` for torch.cuda.utilization)"
        )
    else:
        print(f"  GPU utilization   : {util:.0f}% (mean of samples during the loop)")
    if device == "cuda":
        print(
            f"  Peak GPU memory   : "
            f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        )

    if tok_s < args.min_tok_s:
        print(
            f"\nFAIL: {tok_s:,.0f} tok/s < --min-tok-s {args.min_tok_s:,.0f}"
        )
        sys.exit(1)
    print(f"\nOK: {tok_s:,.0f} tok/s >= {args.min_tok_s:,.0f}")


if __name__ == "__main__":
    main()
