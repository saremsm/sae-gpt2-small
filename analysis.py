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

        text_idx = int(
            torch.searchsorted(
                offsets.contiguous(),
                torch.tensor(flat_pos),
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
