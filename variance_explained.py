"""compute FVU for a trained SAE checkpoint on the held-out shard. standard FVU:
residual SS over SS about the per-dimension mean."""

from __future__ import annotations

import argparse

import torch

from data import ActivationLoader, TokenShard, load_forward_model, resid_post_hook
from sparse_autoencoder import SAEConfig, SparseAutoencoder


def evaluate_fvu(
    sae: SparseAutoencoder,
    loader: ActivationLoader,
    n_tokens: int,
) -> dict:
    """FVU / 1-FVU (raw space), MSE (scaled space) and L0 of `sae` over the first
    n_tokens rows the loader yields (rounded to whole batches). Returns a dict
    with keys n_tokens, fvu, variance_explained, mse_scaled, l0."""
    sae.eval()
    d = sae.config.d_model
    sum_x = torch.zeros(d, dtype=torch.float64, device=loader.device)
    sum_x2 = torch.zeros(d, dtype=torch.float64, device=loader.device)
    resid_ss = torch.zeros((), dtype=torch.float64, device=loader.device)
    mse_sum = 0.0
    l0_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            out = sae(batch)  # scales internally; recon_raw is raw space
            x = batch.double()
            sum_x += x.sum(dim=0)
            sum_x2 += (x * x).sum(dim=0)
            resid_ss += (x - out.recon_raw.double()).pow(2).sum()
            mse_sum += out.reconstruction_loss.item() * batch.shape[0]
            l0_sum += out.l0.item() * batch.shape[0]
            n += batch.shape[0]
            if n >= n_tokens:
                break
    if n == 0:
        raise ValueError("loader yielded no tokens")
    total_ss = (sum_x2 - sum_x * sum_x / n).sum()
    fvu = (resid_ss / total_ss).item()
    return {
        "n_tokens": n,
        "fvu": fvu,
        "variance_explained": 1.0 - fvu,
        "mse_scaled": mse_sum / n,
        "l0": l0_sum / n,
    }


def print_fvu(result: dict, input_scale: float) -> None:
    ve = result["variance_explained"]
    print(f"Tokens evaluated     : {result['n_tokens']:,}")
    print(f"input_scale          : {input_scale:.5f}")
    print(f"Reconstruction MSE   : {result['mse_scaled']:.4f}  (scaled space)")
    print(f"L0 (avg active feats): {result['l0']:.1f}")
    print(f"FVU                  : {result['fvu']:.3f}  (raw space; scale-invariant)")
    print(f"Variance explained   : {ve:.3f}  ({ve*100:.1f}%)  [1 - FVU]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="sae_gpt2_layer8.pt")
    parser.add_argument("--holdout-shard", default="data/holdout.bin")
    parser.add_argument("--n-tokens", type=int, default=100_000)
    parser.add_argument("--batch-seqs", type=int, default=64)
    parser.add_argument("--buffer-tokens", type=int, default=131_072)
    parser.add_argument("--forward", choices=["tl", "hf"], default="hf")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_forward_model(args.forward, device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = SAEConfig(**ckpt["config"])
    sae = SparseAutoencoder(cfg).to(device)
    sae.load_state_dict(ckpt["sae_state_dict"])  # includes input_scale
    sae.eval()

    loader = ActivationLoader(
        model,
        TokenShard(args.holdout_shard),
        hook_name=resid_post_hook(ckpt["layer"]),
        batch_seqs=args.batch_seqs,
        batch_tokens=4096,
        buffer_tokens=args.buffer_tokens,
        device=device,
        epochs=1,
        log_every=0,
    )
    result = evaluate_fvu(sae, loader, n_tokens=args.n_tokens)
    print(f"Eval shard           : {args.holdout_shard} (position 0 excluded)")
    print_fvu(result, sae.input_scale.item())


if __name__ == "__main__":
    main()
