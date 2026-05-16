"""profile one SAE training step - training.train_step, the same function train_sae
and bench_pipeline.py run: forward + backward + decoder-gradient projection +
grad clipping + optimizer.step - under torch.profiler; top ops by self-CUDA-time."""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.optim import AdamW

from sparse_autoencoder import SAEConfig, SparseAutoencoder
from training import BATCH_SIZE, train_step


WARMUP_STEPS = 5
PROFILE_STEPS = 20
D_MODEL = 768
DEFAULT_EXPANSION = 8.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        help="Optional path to write a Chrome trace JSON for chrome://tracing",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=15,
        help="Number of rows to print in the summary table",
    )
    parser.add_argument(
        "--batch-tokens", type=int, default=BATCH_SIZE,
        help=f"activations per step (default {BATCH_SIZE}, the default recipe)",
    )
    parser.add_argument(
        "--expansion", type=float, default=DEFAULT_EXPANSION,
        help="n_features = expansion x 768 (default 8 -> 6144)",
    )
    parser.add_argument(
        "--activation", choices=["relu", "topk"], default="relu",
        help="encoder variant of the profiled step",
    )
    parser.add_argument(
        "--k", type=int, default=32,
        help="latents per token under --activation topk",
    )
    parser.add_argument(
        "--aux-k", type=int, default=0,
        help="AuxK width under topk (0 = off; the sweep used 2k)",
    )
    parser.add_argument(
        "--aux-coeff", type=float, default=0.0,
        help="AuxK weight (0.03125 = 1/32 in the paper)",
    )
    parser.add_argument(
        "--no-tf32", action="store_true",
        help="profile the step with TF32 matmuls off (fp32 sgemm); default "
        "on, as in training (the loader enables TF32 process-wide)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Profiler script requires CUDA. Run on a GPU instance.")

    device = "cuda"
    # Same matmul precision as the training step: ActivationLoader sets both
    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
    torch.backends.cudnn.allow_tf32 = not args.no_tf32
    n_features = int(round(args.expansion * D_MODEL))
    is_topk = args.activation == "topk"
    print(f"Profiling on: {torch.cuda.get_device_name(0)}")
    print(
        f"Config: batch={args.batch_tokens}, d_model={D_MODEL}, "
        f"n_features={n_features} ({args.expansion:g}x expansion), "
        f"{args.activation}" + (f" k={args.k}" if is_topk else "")
        + (f", aux_k={args.aux_k}" if is_topk and args.aux_k else "")
        + f", tf32={'off' if args.no_tf32 else 'on'}"
    )

    cfg = SAEConfig(
        d_model=D_MODEL,
        n_features=n_features,
        l1_coefficient=0.0 if is_topk else SAEConfig.l1_coefficient,
        activation=args.activation,
        k=args.k if is_topk else None,
        aux_k=args.aux_k if is_topk else 0,
        aux_coeff=args.aux_coeff if is_topk else 0.0,
    )
    sae = SparseAutoencoder(cfg).to(device)
    sae.train()

    optimizer = AdamW(
        sae.parameters(), lr=cfg.lr, betas=cfg.adam_betas, weight_decay=0.0,
    )

    # raw synthetic activations at roughly layer-8 scale (norm ~166)
    x = torch.randn(args.batch_tokens, D_MODEL, device=device) * 6.0
    sae.set_input_scale_from_activations(x)

    # warmup: cudnn autotune, allocator caching, lazy init
    print(f"\nWarmup: {WARMUP_STEPS} steps...")
    for _ in range(WARMUP_STEPS):
        train_step(sae, optimizer, x)
    torch.cuda.synchronize()

    print(f"Profiling: {PROFILE_STEPS} steps...\n")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(PROFILE_STEPS):
            with record_function("sae_step"):
                train_step(sae, optimizer, x)

    torch.cuda.synchronize()

    print("Top ops by self-CUDA-time:")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=args.row_limit,
        )
    )

    if args.trace is not None:
        prof.export_chrome_trace(args.trace)
        print(f"\nChrome trace written to {args.trace}")


if __name__ == "__main__":
    main()
