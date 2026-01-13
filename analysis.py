"""analysis.py - feature analysis for a trained SAE. three stages:
build_activation_cache (model forwards, residual stream) -> build_feature_cache"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, TYPE_CHECKING, TypedDict

import torch
from torch import Tensor

if TYPE_CHECKING:
    from sparse_autoencoder import SparseAutoencoder
    from transformer_lens import HookedTransformer

# structured types
class MaxActivatingExample(TypedDict):
    """one top-activating example for a feature"""
    activation: float
    peak_token: str
    context: list[str]
    context_activations: list[float]
    peak_position_in_context: int
    full_text: str


class TokenProjection(NamedTuple):
    """logit-lens projection of one decoder row through the unembedding"""
    positive_tokens: list[tuple[str, float]]
    negative_tokens: list[tuple[str, float]]

# activation cache

@dataclass
class ActivationCache:
    """precomputed residual-stream activations for a corpus (RAW scale)"""
    activations: list[Tensor]
    token_strings: list[list[str]]
    texts: list[str]


def build_activation_cache(
    model: "HookedTransformer",
    texts: list[str],
    layer: int,
    device: str = "cpu",
) -> ActivationCache:
    hook_name = f"blocks.{layer}.hook_resid_post"
    model.eval()
    model.to(device)

    all_activations: list[Tensor] = []
    all_token_strings: list[list[str]] = []

    for text in texts:
        tokens = model.to_tokens(text, prepend_bos=True)
        tok_strs = model.to_str_tokens(text)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens.to(device),
                names_filter=hook_name,
            )

        acts = cache[hook_name][0].cpu()
        acts = acts[1:]
        tok_strs = tok_strs[1:]
        all_activations.append(acts)
        all_token_strings.append(tok_strs)

    return ActivationCache(
        activations=all_activations,
        token_strings=all_token_strings,
        texts=texts,
    )

# feature cache

@dataclass
class FeatureCache:
    """precomputed SAE feature activations for every token in the corpus"""
    feature_acts: Tensor
    text_offsets: Tensor
    token_strings: list[list[str]]
    texts: list[str]


def build_feature_cache(
    sae: "SparseAutoencoder",
    activation_cache: ActivationCache,
    device: str = "cpu",
    encode_batch_size: int = 2048,
) -> FeatureCache:
    sae = sae.to(device)
    sae.eval()

    all_feature_acts: list[Tensor] = []
    offsets: list[int] = [0]

    for acts in activation_cache.activations:
        seq_len = acts.shape[0]
        text_features: list[Tensor] = []

        for start in range(0, seq_len, encode_batch_size):
            chunk = acts[start : start + encode_batch_size].to(device)
            with torch.no_grad():
                # encode() applies the SAE's input normalization: raw residuals.
                text_features.append(sae.encode(chunk).cpu())

        text_feature_tensor = torch.cat(text_features, dim=0)
        all_feature_acts.append(text_feature_tensor)
        offsets.append(offsets[-1] + seq_len)

    return FeatureCache(
        feature_acts=torch.cat(all_feature_acts, dim=0),
        text_offsets=torch.tensor(offsets, dtype=torch.long),
        token_strings=activation_cache.token_strings,
        texts=activation_cache.texts,
    )

# queries

def find_max_activating_examples(
    feature_cache: FeatureCache,
    feature_idx: int,
    top_k: int = 10,
    context_window: int = 5,
) -> list[MaxActivatingExample]:
    all_acts: Tensor = feature_cache.feature_acts[:, feature_idx]

    total_tokens = all_acts.shape[0]
    k = min(top_k, total_tokens)

    top_values, top_flat_indices = all_acts.topk(k)

    examples: list[MaxActivatingExample] = []
    offsets = feature_cache.text_offsets

    for rank in range(k):
        activation_value = top_values[rank].item()
        if activation_value <= 0:
            break

        flat_pos = int(top_flat_indices[rank].item())

        # right=True: peaks at offsets[k] go to text k, not k-1
        text_idx = int(
            torch.searchsorted(
                offsets.contiguous(),
                torch.tensor(flat_pos),
                right=True,
            ).item()
        ) - 1

        tok_start = int(offsets[text_idx].item())
        pos_in_text = flat_pos - tok_start

        tok_strs = feature_cache.token_strings[text_idx]
        text = feature_cache.texts[text_idx]

        tok_end = int(offsets[text_idx + 1].item())
        text_acts = all_acts[tok_start:tok_end]

        ctx_start = max(0, pos_in_text - context_window)
        ctx_end = min(len(tok_strs), pos_in_text + context_window + 1)

        examples.append(
            MaxActivatingExample(
                activation=activation_value,
                peak_token=tok_strs[pos_in_text],
                context=tok_strs[ctx_start:ctx_end],
                context_activations=text_acts[ctx_start:ctx_end].tolist(),
                peak_position_in_context=pos_in_text - ctx_start,
                full_text=text[:100],
            )
        )

    examples.sort(key=lambda e: e["activation"], reverse=True)
    return examples


def feature_token_projection(
    sae: "SparseAutoencoder",
    model: "HookedTransformer",
    feature_idx: int,
    top_k: int = 20,
) -> TokenProjection:
    """Feature direction through the unembedding: W_dec[i] @ W_U, no ln_final."""
    feature_direction = sae.W_dec[feature_idx]

    with torch.no_grad():
        logits = feature_direction @ model.W_U

    top_vals, top_idxs = logits.topk(top_k)
    bot_vals, bot_idxs = logits.topk(top_k, largest=False)

    positive_tokens = [
        (model.to_string(int(idx.item())), float(val.item()))
        for idx, val in zip(top_idxs, top_vals)
    ]
    negative_tokens = [
        (model.to_string(int(idx.item())), float(val.item()))
        for idx, val in zip(bot_idxs, bot_vals)
    ]

    return TokenProjection(
        positive_tokens=positive_tokens,
        negative_tokens=negative_tokens,
    )


def analyze_feature(
    sae: "SparseAutoencoder",
    model: "HookedTransformer",
    feature_idx: int,
    feature_cache: FeatureCache,
) -> None:
    """Presentational: prints a human-readable feature report."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"Feature {feature_idx} analysis")
    print(sep)

    print("\nTop promoted tokens (logit-lens projection of decoder row):")
    projection = feature_token_projection(sae, model, feature_idx)
    pos_display = ", ".join(
        f"'{t}' ({v:.2f})" for t, v in projection.positive_tokens[:10]
    )
    neg_display = ", ".join(
        f"'{t}' ({v:.2f})" for t, v in projection.negative_tokens[:5]
    )
    print(f"  Positive : {pos_display}")
    print(f"  Negative : {neg_display}")

    print("\nTop activating contexts:")
    examples = find_max_activating_examples(feature_cache, feature_idx, top_k=5)

    for i, ex in enumerate(examples):
        context_parts = []
        for j, (tok, act) in enumerate(
            zip(ex["context"], ex["context_activations"])
        ):
            if j == ex["peak_position_in_context"]:
                context_parts.append(f"[{tok}|{act:.2f}]")
            elif act > 0.1:
                context_parts.append(f"{tok}({act:.1f})")
            else:
                context_parts.append(tok)

        print(f"\n  Example {i + 1}: activation = {ex['activation']:.3f}")
        print(f"  Peak token : '{ex['peak_token']}'")
        print(f"  Context    : {''.join(context_parts)}")

    print()


def find_interesting_features(
    feature_cache: FeatureCache,
    sae: "SparseAutoencoder",
    n_features_to_return: int = 50,
    chunk_size: int = 2048,
) -> list[int]:
    """Rank features by mean activation when active, within a 0.1%-20% activation-
    rate band. Takes n_features, not the SAE - the model served only a shape
    check."""
    fa = feature_cache.feature_acts

    n_features = sae.config.n_features
    if fa.shape[1] != n_features:
        raise ValueError(
            f"FeatureCache has {fa.shape[1]} features but SAE has {n_features}"
        )

    total_tokens, _ = fa.shape

    feature_counts = torch.zeros(n_features, dtype=torch.float32)
    feature_sum_acts = torch.zeros(n_features, dtype=torch.float32)

    for start in range(0, total_tokens, chunk_size):
        chunk = fa[start : start + chunk_size]
        feature_counts += (chunk > 0).float().sum(dim=0)
        feature_sum_acts += chunk.sum(dim=0)

    rate = feature_counts / max(total_tokens, 1)
    mean_when_active = feature_sum_acts / feature_counts.clamp(min=1)

    mask = (rate > 0.001) & (rate < 0.2) & (mean_when_active > 0.5)
    candidate_indices = mask.nonzero(as_tuple=True)[0].tolist()

    ranked = sorted(
        candidate_indices,
        key=lambda i: mean_when_active[i].item(),
        reverse=True,
    )

    print(
        f"Found {len(ranked)} candidate features in the "
        f"0.1%-20% activation-rate band:"
    )
    for i in ranked[:n_features_to_return]:
        print(
            f"  feature {i:5d}: rate={rate[i]:.4f}, "
            f"mean act when active={mean_when_active[i]:.3f}"
        )

    return ranked[:n_features_to_return]
