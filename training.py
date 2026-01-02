from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import torch

try:
    from transformer_lens import HookedTransformer
    from datasets import load_dataset
except ImportError as exc:
    raise ImportError(
        "Run: pip install transformer_lens datasets"
    ) from exc


@runtime_checkable
class ActivationSource(Protocol):
    """yields (n_tokens, d_model) float32 tensors of residual activations."""

    def __iter__(self) -> Iterator[torch.Tensor]:
        ...


class InlineActivationSource:
    """stream residual activations from `model` at `layer`."""

    def __init__(
        self,
        model: HookedTransformer,
        layer: int,
        dataset_name: str = "NeelNanda/pile-10k",
        batch_size: int = 32,
        context_length: int = 128,
        device: str = "cpu",
        max_batches: int | None = None,
    ) -> None:
        self.model = model
        self.layer = layer
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.max_batches = max_batches
        self._hook_name = f"blocks.{layer}.hook_resid_post"

        tokenizer = model.tokenizer
        bos = tokenizer.bos_token_id
        if bos is None:
            bos = tokenizer.eos_token_id
        if bos is None:
            raise ValueError(
                "tokenizer has neither bos_token_id nor eos_token_id"
            )
        self._bos_id = bos

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def _tokenize_batch(
        self, texts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoding = self.model.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.context_length - 1,
        )
        input_ids = encoding["input_ids"]
        attn_mask = encoding["attention_mask"]

        n = input_ids.shape[0]
        bos_col = torch.full((n, 1), self._bos_id, dtype=input_ids.dtype)
        bos_mask = torch.ones((n, 1), dtype=attn_mask.dtype)
        tokens = torch.cat([bos_col, input_ids], dim=1)
        attn_mask = torch.cat([bos_mask, attn_mask], dim=1)
        return tokens, attn_mask

    def __iter__(self) -> Iterator[torch.Tensor]:
        dataset = load_dataset(
            self.dataset_name, split="train", trust_remote_code=True
        )
        self.model.eval()

        batch_count = 0
        texts: list[str] = []

        for item in dataset:
            text = item.get("text", item.get("content", ""))
            if not text or len(text) < 50:
                continue
            texts.append(text)

            if len(texts) < self.batch_size:
                continue

            tokens, attn_mask = self._tokenize_batch(texts)

            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    tokens.to(self.device),
                    names_filter=self._hook_name,
                )

            acts = cache[self._hook_name].cpu()
            yield acts[attn_mask.bool()]

            texts = []
            batch_count += 1
            if self.max_batches is not None and batch_count >= self.max_batches:
                return
