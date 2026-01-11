from __future__ import annotations

import random

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
from training import InlineActivationSource, train_sae
from analysis import (
    build_activation_cache,
    build_feature_cache,
    find_interesting_features,
    analyze_feature,
)


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
    axes[0, 1].set_title("Reconstruction loss (MSE, normalized space)")
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
            "Raw input norm (SAE rescales to $\\sqrt{d}$ internally)",
            fontsize=9,
        )

    axes[1, 2].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training history saved to '{save_path}'")


def main() -> None:
    SEED = 42
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

    print("\nLoading GPT-2-small via TransformerLens...")
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()
    print(
        f"Model: {model.cfg.n_layers} layers, "
        f"{model.cfg.n_heads} heads, "
        f"d_model={model.cfg.d_model}"
    )

    TARGET_LAYER = 8

    # warmup_steps=100 vs ~976 expected steps: the previous 1000 kept the whole
    config = SAEConfig(
        d_model=model.cfg.d_model,
        n_features=3072,
        l1_coefficient=5e-3,
        lr=2e-4,
        warmup_steps=100,
    )

    sae = SparseAutoencoder(config)

    activation_source = InlineActivationSource(
        model=model,
        layer=TARGET_LAYER,
        batch_size=32,
        context_length=128,
        device=device,
    )

    history = train_sae(
        sae=sae,
        activation_source=activation_source,
        n_training_tokens=500_000,
        resample_interval=250,
        log_interval=50,
        device=device,
        seed=SEED,
    )

    checkpoint_path = "sae_gpt2_layer8.pt"
    torch.save(
        {
            "sae_state_dict": sae.state_dict(),
            "config": config,
            "layer": TARGET_LAYER,
            "training_history": history,
        },
        checkpoint_path,
    )
    print(f"\nModel saved to '{checkpoint_path}'")

    plot_training_history(history)

    print("\nLoading analysis texts...")
    dataset = load_dataset("NeelNanda/pile-10k", split="train", trust_remote_code=True)
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

    interesting = find_interesting_features(feat_cache, sae)

    n_to_show = min(5, len(interesting))
    print(f"\nAnalyzing {n_to_show} interesting features:")
    for feature_idx in interesting[:n_to_show]:
        analyze_feature(sae, model, feature_idx, feat_cache)

    print("\nReflection: see README.md.")


if __name__ == "__main__":
    main()
