from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from transformer_lens import HookedTransformer
    from datasets import load_dataset
except ImportError as exc:
    raise ImportError(
        "Missing deps. Run: pip install transformer_lens datasets"
    ) from exc

from sparse_autoencoder import SparseAutoencoder, SAEConfig
from data import ActivationLoader, TokenShard, load_forward_model, resid_post_hook
from training import BATCH_SIZE, train_sae
from eval import (
    evaluate,
    make_run_record,
    print_metrics,
    utc_now_iso,
    write_json,
)
from analysis import (
    build_activation_cache,
    build_feature_cache,
    feature_activation_stats,
    find_interesting_features,
    analyze_feature,
)


TARGET_LAYER = 8
# Schedule knobs are denominated in TOKENS so a different --batch-tokens keeps
# the same warmup / resampling / logging cadence per token.
WARMUP_TOKENS = 51_200
RESAMPLE_TOKENS = 128_000
LOG_TOKENS = 25_600

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a layer-8 SAE from a pre-tokenized shard "
        "(see `python -m data tokenize`)."
    )
    parser.add_argument(
        "--train-shard", default="data/train.bin",
        help="packed uint16 shard to train on (data.py)",
    )
    parser.add_argument(
        "--holdout-shard", default="data/holdout.bin",
        help="document-disjoint shard for the post-training evaluation "
        "(eval.evaluate: FVU, L0, CE clean/recon/zero, loss recovered); "
        "skipped with a warning if the file is missing",
    )
    parser.add_argument(
        "--n-tokens", type=int, default=500_000,
        help="training tokens (position-0 rows excluded)",
    )
    parser.add_argument(
        "--batch-seqs", type=int, default=256,
        help="sequences per GPT-2 forward in the loader",
    )
    parser.add_argument(
        "--batch-tokens", type=int, default=BATCH_SIZE,
        help="activations per SAE optimizer step. 512 is the recipe every "
        "numbers row uses (977 steps for 500K tokens; 90K tok/s on the A10). "
        "2048 clears the A10 throughput gate (107K tok/s) but at 500K tokens "
        "that is 244 steps at the same lr and the SAE is badly undertrained "
        "(FVU 0.82 vs 0.63) - use it for long runs (the MVP run-scale token budgets), "
        "not for the 500K headline. Warmup / resampling / logging cadence is "
        "token-denominated so it is the same per token at either batch.",
    )
    parser.add_argument(
        "--buffer-tokens", type=int, default=1_000_000,
        help="on-device shuffle buffer size in activations "
        "(x d_model x 4 bytes of GPU memory)",
    )
    parser.add_argument(
        "--eval-tokens", type=int, default=2_000_000,
        help="held-out tokens for the post-training evaluation (position-0 "
        "rows not counted); three fp32 GPT-2 forwards per token",
    )
    parser.add_argument(
        "--eval-batch-seqs", type=int, default=64,
        help="rows per GPT-2 forward in the evaluation (fp32 logits are "
        "b x seq_len x 50257: 64 rows ~ 1.7 GB)",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="name of the results/<run>/ directory that receives "
        "metrics.json (default: UTC timestamp)",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="parent directory of results/<run>/",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-autocast", action="store_true",
        help="run the GPT-2 forward in fp32 instead of bf16 autocast (CUDA)",
    )
    parser.add_argument(
        "--forward", choices=["tl", "hf"], default="hf",
        help="GPT-2 implementation the activation loader runs (analysis "
        "always uses TransformerLens): tl = HookedTransformer, hf = "
        "HuggingFace GPT2Model with SDPA, same residual (data.HFResidualModel)",
    )
    return parser.parse_args()


def plot_training_history(history: dict, save_path: str = "training_history.png") -> None:
    if not history.get("step"):
        print("No training history to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("SAE Training History", fontsize=14)

    steps = history["step"]

    axes[0, 0].plot(steps, history["loss"])
    axes[0, 0].set_title("Total loss")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_yscale("log")

    axes[0, 1].plot(steps, history["reconstruction_loss"])
    axes[0, 1].set_title("Reconstruction loss (MSE, scaled space)")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("MSE")

    axes[0, 2].plot(steps, history["l0"])
    axes[0, 2].set_title("L0 (avg active features per token)")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylabel("Active features")
    axes[0, 2].axhline(y=20, color="r", linestyle="--", alpha=0.5, label="target L0=20")
    axes[0, 2].legend()

    axes[1, 0].plot(steps, history["dead_features"])
    axes[1, 0].set_title("Dead features over training")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Dead feature count")

    if history.get("act_norm"):
        axes[1, 1].plot(steps, history["act_norm"])
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel(r"$\|x\|_2$ mean (raw)")
        axes[1, 1].set_title(
            "Raw input norm (SAE multiplies by input_scale internally)",
            fontsize=9,
        )

    axes[1, 2].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training history saved to '{save_path}'")


def main() -> None:
    args = parse_args()
    started_at = utc_now_iso()

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Fail on a missing shard before spending time on the model load.
    if not os.path.exists(args.train_shard):
        raise SystemExit(
            f"train shard {args.train_shard!r} not found; build it with "
            f"`python -m data tokenize ...` (see README, pipeline)."
        )
    holdout_shard = args.holdout_shard
    if not os.path.exists(holdout_shard):
        print(
            f"WARNING: holdout shard {holdout_shard!r} not found; the "
            f"post-training held-out evaluation is skipped (metrics is null "
            f"in metrics.json)."
        )
        holdout_shard = None

    print("\nLoading GPT-2-small via TransformerLens...")
    model = HookedTransformer.from_pretrained("gpt2").to(device)
    model.eval()
    print(
        f"Model: {model.cfg.n_layers} layers, "
        f"{model.cfg.n_heads} heads, "
        f"d_model={model.cfg.d_model}"
    )

    hook_name = resid_post_hook(TARGET_LAYER)
    # The model the loader forwards through. TL for analysis regardless.
    if args.forward == "hf":
        print("Loading GPT-2-small via HuggingFace (SDPA) for the loader...")
        forward_model = load_forward_model("hf", device)
    else:
        forward_model = model

    # warmup 51.2K tokens (100 steps at 512) vs ~976 steps at 512: the previous
    # 1000-step warmup kept the whole run inside warmup (train_sae also warn-
    # clamps).
    warmup_steps = max(10, WARMUP_TOKENS // args.batch_tokens)
    resample_interval = max(1, RESAMPLE_TOKENS // args.batch_tokens)
    log_interval = max(1, LOG_TOKENS // args.batch_tokens)
    config = SAEConfig(
        d_model=model.cfg.d_model,
        n_features=3072,
        l1_coefficient=5e-3,
        lr=2e-4,
        warmup_steps=warmup_steps,
    )
    print(
        f"Schedule: batch_tokens={args.batch_tokens}, warmup {warmup_steps} steps "
        f"({WARMUP_TOKENS:,} tok), resample every {resample_interval} steps "
        f"({RESAMPLE_TOKENS:,} tok), log every {log_interval} steps"
    )

    sae = SparseAutoencoder(config)

    train_shard = TokenShard(args.train_shard)
    print(
        f"Train shard: {args.train_shard} ({train_shard.n_seqs:,} seqs x "
        f"{train_shard.seq_len}, docs {train_shard.meta['doc_range']})"
    )
    if args.buffer_tokens > args.n_tokens:
        print(
            f"NOTE: --buffer-tokens ({args.buffer_tokens:,}) > --n-tokens "
            f"({args.n_tokens:,}): the loader fills the whole buffer before "
            f"the first step and forwards ~buffer/2 tokens more than training "
            f"consumes, so wall time is dominated by the fill. For short runs "
            f"lower --buffer-tokens (throughput numbers come from "
            f"bench_pipeline.py, not from here)."
        )
    loader = ActivationLoader(
        model=forward_model,
        shard=train_shard,
        hook_name=hook_name,
        batch_seqs=args.batch_seqs,
        batch_tokens=args.batch_tokens,
        buffer_tokens=args.buffer_tokens,
        device=device,
        seed=SEED,
        autocast=not args.no_autocast,
    )

    t_train = time.perf_counter()
    history = train_sae(
        sae=sae,
        loader=loader,
        n_training_tokens=args.n_tokens,
        resample_interval=resample_interval,
        log_interval=log_interval,
        device=device,
        seed=SEED,
    )
    train_wall = time.perf_counter() - t_train
    print(
        f"Loader ({args.forward}): {loader.tokens_yielded:,} tokens yielded at "
        f"{loader.throughput_tok_s():,.0f} tok/s end-to-end (GPT-2 forward + "
        f"SAE step, {loader.n_refills} buffer refills)"
    )

    checkpoint_path = "sae_gpt2_layer8.pt"
    # plain-dict config: loads under torch.load's safe weights_only=True.
    torch.save(
        {
            "sae_state_dict": sae.state_dict(),
            "config": asdict(config),
            "layer": TARGET_LAYER,
            "training_history": history,
            "train_shard": train_shard.meta,
        },
        checkpoint_path,
    )
    print(
        f"\nModel saved to '{checkpoint_path}' "
        f"(input_scale={sae.input_scale.item():.5f})"
    )

    plot_training_history(history)

    metrics = None
    holdout_meta = None
    if holdout_shard is not None:
        print(
            f"\nEvaluating on the held-out shard ({args.eval_tokens:,} tokens, "
            f"position 0 excluded; 3 fp32 forwards per batch)..."
        )
        holdout = TokenShard(holdout_shard)
        holdout_meta = holdout.meta
        # train_meta: refuses to run if the two shards' document ranges overlap.
        metrics = evaluate(
            sae, model, holdout, hook_name,
            n_tokens=args.eval_tokens, batch_seqs=args.eval_batch_seqs,
            device=device, exclude_bos=True, train_meta=train_shard.meta,
        )
        print_metrics(metrics, sae.input_scale.item())

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    record = make_run_record(
        run=run_name,
        config={
            "sae": asdict(config),
            "layer": TARGET_LAYER,
            "hook_name": hook_name,
            "device": device,
            "args": vars(args),
            "schedule": {
                "warmup_steps": warmup_steps,
                "resample_interval": resample_interval,
                "log_interval": log_interval,
            },
        },
        metrics=metrics,
        started_at=started_at,
        train_shard=train_shard.meta,
        holdout_shard=holdout_meta,
        checkpoint=checkpoint_path,
        training={
            "tokens": loader.tokens_yielded,
            "steps": history["step"][-1] if history["step"] else 0,
            "final_loss": history["loss"][-1] if history["loss"] else None,
            "final_l0": history["l0"][-1] if history["l0"] else None,
            "final_dead_features": (
                history["dead_features"][-1] if history["dead_features"] else None
            ),
            "input_scale": sae.input_scale.item(),
            "loader_tok_s": loader.throughput_tok_s(),
            "train_wall_seconds": train_wall,
        },
    )
    metrics_path = write_json(
        os.path.join(args.results_dir, run_name, "metrics.json"), record
    )
    print(f"metrics written to {metrics_path}")
    if metrics is None:
        print("(no held-out shard: metrics is null in metrics.json)")

    print("\nLoading analysis texts...")
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    analysis_texts = [
        item.get("text", "")
        for item in list(dataset)[:200]
        if len(item.get("text", "")) > 100
    ][:100]
    print(f"Loaded {len(analysis_texts)} texts for analysis.")

    print("Building activation cache (one model forward pass per text)...")
    act_cache = build_activation_cache(model, analysis_texts, TARGET_LAYER, device=device)

    print("Building feature cache (one SAE encode pass per text)...")
    feat_cache = build_feature_cache(sae, act_cache, device=device)

    interesting = find_interesting_features(
        feat_cache,
        n_features=config.n_features,
    )
    # find_interesting_features returns data only; the printing lives here.
    stats = feature_activation_stats(feat_cache)
    print(
        f"\nTop {len(interesting)} candidate features "
        f"(activation rate 0.1%-20%, mean activation when active > 0.5):"
    )
    for feature_idx in interesting:
        print(
            f"  feature {feature_idx:5d}: "
            f"rate={stats.rate[feature_idx]:.4f}, "
            f"mean act when active={stats.mean_when_active[feature_idx]:.3f}"
        )

    n_to_show = min(5, len(interesting))
    print(f"\nAnalyzing {n_to_show} interesting features:")
    for feature_idx in interesting[:n_to_show]:
        analyze_feature(sae, model, feature_idx, feat_cache)

    print("\nReflection: see README.md.")


if __name__ == "__main__":
    main()
