"""profile one SAE training step: forward + backward + project_decoder_grad +
optimizer.step under torch.profiler; top ops by self-CUDA-time."""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.optim import AdamW

from sparse_autoencoder import SAEConfig, SparseAutoencoder


WARMUP_STEPS = 5
PROFILE_STEPS = 20
BATCH_SIZE = 512
D_MODEL = 768
N_FEATURES = 3072


def main() -> None:
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Profiler script requires CUDA. Run on a GPU instance.")

    device = "cuda"
    print(f"Profiling on: {torch.cuda.get_device_name(0)}")
    print(
        f"Config: batch={BATCH_SIZE}, d_model={D_MODEL}, "
        f"n_features={N_FEATURES} (4x expansion)"
    )

    cfg = SAEConfig(
        d_model=D_MODEL,
        n_features=N_FEATURES,
        l1_coefficient=5e-3,
    )
    sae = SparseAutoencoder(cfg).to(device)
    sae.train()

    optimizer = AdamW(sae.parameters(), lr=2e-4, weight_decay=0.0)

    # synthetic activations pre-scaled to sqrt(d_model)
    x_raw = torch.randn(BATCH_SIZE, D_MODEL, device=device) * 6.0
    x = (
        x_raw
        / x_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        * (D_MODEL ** 0.5)
    )

    # warmup: cudnn autotune, allocator caching, lazy init
    print(f"\nWarmup: {WARMUP_STEPS} steps...")
    for _ in range(WARMUP_STEPS):
        out = sae(x)
        optimizer.zero_grad()
        out.loss.backward()
        sae.project_decoder_grad()
        optimizer.step()
    torch.cuda.synchronize()

    print(f"Profiling: {PROFILE_STEPS} steps...\n")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(PROFILE_STEPS):
            with record_function("sae_step"):
                with record_function("forward"):
                    out = sae(x)
                with record_function("backward"):
                    optimizer.zero_grad()
                    out.loss.backward()
                with record_function("decoder_grad_projection"):
                    sae.project_decoder_grad()
                with record_function("optimizer_step"):
                    optimizer.step()

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
