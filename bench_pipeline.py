"""benchmark the training pipeline: GPT-2 forward + activation buffer + SAE step.
Exits non-zero if pass-2 tok/s < --min-tok-s."""

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


def bench_loader(loader: ActivationLoader, n_tokens: int, device: str) -> float:
    """tok/s of the loader alone over n_tokens yielded rows."""
    _sync(device)
    t0 = time.perf_counter()
    n = 0
    for batch in loader:
        n += batch.shape[0]
        if n >= n_tokens:
            break
    _sync(device)
    return n / (time.perf_counter() - t0)


def bench_loader_and_sae(
    loader: ActivationLoader,
    sae: SparseAutoencoder,
    n_tokens: int,
    device: str,
) -> tuple[float, float | None]:
    """tok/s of loader + SAE train step over n_tokens rows (after WARMUP_STEPS
    untimed steps), plus mean sampled GPU utilization."""
    optimizer = AdamW(sae.parameters(), lr=sae.config.lr, weight_decay=0.0)
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
    tok_s = n / (time.perf_counter() - t0)
    mean_util = sum(utils) / len(utils) if utils else None
    return tok_s, mean_util


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", default="data/train.bin")
    parser.add_argument("--n-tokens", type=int, default=5_000_000)
    parser.add_argument("--min-tok-s", type=float, default=0.0)
    parser.add_argument("--batch-seqs", type=int, default=256)
    parser.add_argument("--batch-tokens", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer-tokens", type=int, default=1_000_000)
    parser.add_argument("--n-features", type=int, default=3072)
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
        f"forward={args.forward}"
    )

    model = load_forward_model(args.forward, device)

    # Pass 1: loader alone, with the read / forward split measured.
    loader = make_loader(model, shard, args, device, profile=True)
    loader_tok_s = bench_loader(loader, args.n_tokens, device)
    busy = loader.read_seconds + loader.forward_seconds
    read_pct = 100.0 * loader.read_seconds / busy if busy > 0 else 0.0
    print(
        f"\nLoader alone        : {loader_tok_s:,.0f} tok/s "
        f"({loader.n_chunks} forwards of {args.batch_seqs} seqs, "
        f"{loader.n_refills} refills)"
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
    cfg = SAEConfig(d_model=model.cfg.d_model, n_features=args.n_features)
    sae = SparseAutoencoder(cfg).to(device)
    tok_s, util = bench_loader_and_sae(loader, sae, args.n_tokens, device)
    print(
        f"Loader + SAE step   : {tok_s:,.0f} tok/s "
        f"(batch {args.batch_tokens}, {args.n_features} features)"
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
