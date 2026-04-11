"""data.py - pre-tokenized token shards and a GPU-resident activation loader. Mid-
row 50256s are ordinary tokens; position 0 is the attention-sink outlier the
README describes, and the loader drops it BY POSITION, never by id."""

from __future__ import annotations

import argparse
import json
import re
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Iterator, TYPE_CHECKING

import numpy as np
import torch

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "Run: pip install datasets transformers"
    ) from exc

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


TOKENIZER_NAME = "gpt2"
# GPT-2: BOS == EOS == pad == 50256.
BOS_ID = 50256
EOS_ID = 50256
DEFAULT_DATASET = "monology/pile-uncopyrighted"
DEFAULT_SPLIT = "train"
DEFAULT_SEQ_LEN = 128
# `IterableDataset.shuffle(seed, buffer_size)` window, in documents.
SHUFFLE_BUFFER_DOCS = 10_000
# All GPT-2 ids are < 65536, so two bytes per token.
SHARD_DTYPE = np.uint16
LOG_EVERY_TOKENS = 10_000_000
# Documents per tokenizer call.
TOKENIZE_BATCH_DOCS = 256

_HOOK_LAYER_RE = re.compile(r"^blocks\.(\d+)\.")


def resid_post_hook(layer: int) -> str:
    """The hook name convention used throughout the repo."""
    return f"blocks.{layer}.hook_resid_post"


def layer_from_hook_name(hook_name: str) -> int:
    """`blocks.<layer>.<...>` -> layer."""
    m = _HOOK_LAYER_RE.match(hook_name)
    if m is None:
        raise ValueError(
            f"hook_name must look like 'blocks.<layer>.<hook>', got "
            f"{hook_name!r}"
        )
    return int(m.group(1))


def shard_meta_path(bin_path: str | Path) -> Path:
    """`data/train.bin` -> `data/train.json` (the sidecar)."""
    return Path(bin_path).with_suffix(".json")


# Tokenization / packing


class _PackedShardWriter:
    """Packs a stream of documents into rows of [BOS] + (seq_len - 1) stream tokens
    and appends them to a uint16 file."""

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        max_tokens: int | None,
    ) -> None:
        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {seq_len}")
        self.path = Path(path)
        self.seq_len = seq_len
        self.max_seqs = None if max_tokens is None else max_tokens // seq_len
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "wb")
        self._pending = np.zeros(0, dtype=SHARD_DTYPE)
        self.n_seqs = 0
        self.n_docs = 0
        self._next_log = LOG_EVERY_TOKENS
        self._t0 = time.perf_counter()

    @property
    def n_tokens(self) -> int:
        return self.n_seqs * self.seq_len

    @property
    def full(self) -> bool:
        return self.max_seqs is not None and self.n_seqs >= self.max_seqs

    def write_docs(self, docs_ids: list[list[int]]) -> None:
        """Append documents (lists of token ids; empty ones are skipped) to the
        stream and flush every complete row."""
        parts = [self._pending]
        for ids in docs_ids:
            if not ids:
                continue
            arr = np.asarray(ids, dtype=np.int64)
            if arr.max() > np.iinfo(SHARD_DTYPE).max:
                raise ValueError(
                    f"token id {int(arr.max())} does not fit {SHARD_DTYPE}"
                )
            parts.append(arr.astype(SHARD_DTYPE))
            parts.append(np.array([EOS_ID], dtype=SHARD_DTYPE))
            self.n_docs += 1
        stream = np.concatenate(parts)

        body = self.seq_len - 1
        n_new = len(stream) // body
        if self.max_seqs is not None:
            n_new = min(n_new, self.max_seqs - self.n_seqs)
        if n_new > 0:
            rows = np.empty((n_new, self.seq_len), dtype=SHARD_DTYPE)
            rows[:, 0] = BOS_ID
            rows[:, 1:] = stream[: n_new * body].reshape(n_new, body)
            self._file.write(rows.tobytes())
            self.n_seqs += n_new
        self._pending = stream[n_new * body :]

        if self.n_tokens >= self._next_log:
            elapsed = time.perf_counter() - self._t0
            print(
                f"[tokenize] {self.path.name}: {self.n_tokens:,} tokens "
                f"({self.n_seqs:,} seqs, {self.n_docs:,} docs) in "
                f"{elapsed:.0f}s, {self.n_tokens / max(elapsed, 1e-9):,.0f} "
                f"tok/s"
            )
            self._next_log += LOG_EVERY_TOKENS

    def close(self) -> None:
        self._file.close()


def _iter_texts(dataset: Iterable, text_field: str) -> Iterator[str]:
    """Yield the text of every item; items may be dicts (HF rows) or plain strings."""
    for item in dataset:
        if isinstance(item, str):
            yield item
        else:
            text = item.get(text_field, "")
            yield text if isinstance(text, str) else ""


def _consume_docs(
    texts: Iterator[str],
    writer: _PackedShardWriter,
    tokenizer,
    max_docs: int | None,
) -> int:
    """Pull documents from `texts` into `writer` until max_docs have been taken
    (None: no limit), the writer is full, or the stream ends."""
    taken = 0
    batch: list[str] = []

    def flush() -> None:
        if batch:
            writer.write_docs(tokenizer(batch)["input_ids"])
            batch.clear()

    for text in texts:
        batch.append(text)
        taken += 1
        if len(batch) >= TOKENIZE_BATCH_DOCS:
            flush()
            if writer.full:
                break
        if max_docs is not None and taken >= max_docs:
            break
    flush()
    return taken


def tokenize_corpus(
    dataset: str | Iterable,
    split: str = DEFAULT_SPLIT,
    out_path: str = "data/train.bin",
    n_tokens: int = 220_000_000,
    seq_len: int = DEFAULT_SEQ_LEN,
    seed: int = 0,
    text_field: str = "text",
    holdout_docs: int = 0,
    holdout_path: str | None = None,
    streaming: bool = True,
) -> dict:
    """Tokenize a text corpus once into packed uint16 shards."""
    if n_tokens < seq_len:
        raise ValueError(f"n_tokens ({n_tokens}) < seq_len ({seq_len})")
    if holdout_docs < 0:
        raise ValueError(f"holdout_docs must be >= 0, got {holdout_docs}")
    if holdout_docs > 0 and holdout_path is None:
        raise ValueError("holdout_docs > 0 requires holdout_path")

    # The HF tokenizer alone (not TransformerLens): no model load.
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.eos_token_id != EOS_ID:
        raise ValueError(
            f"expected GPT-2 tokenizer with eos id {EOS_ID}, got "
            f"{tokenizer.eos_token_id}"
        )
    # Documents longer than the model context are fine here.
    tokenizer.model_max_length = 10**9

    if isinstance(dataset, str):
        dataset_name = dataset
        ds = load_dataset(dataset, split=split, streaming=streaming)
        if streaming:
            ds = ds.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER_DOCS)
        else:
            ds = ds.shuffle(seed=seed)
    else:
        dataset_name = "<iterable>"
        ds = dataset
    texts = _iter_texts(ds, text_field)

    def meta_for(writer: _PackedShardWriter, doc_range: tuple[int, int]) -> dict:
        return {
            "n_tokens": writer.n_tokens,
            "n_seqs": writer.n_seqs,
            "seq_len": seq_len,
            "seed": seed,
            "dataset": dataset_name,
            "split": split,
            "text_field": text_field,
            "doc_range": list(doc_range),
            "n_docs": writer.n_docs,
            "tokenizer": TOKENIZER_NAME,
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
            "dtype": np.dtype(SHARD_DTYPE).name,
            "packing": "bos at position 0 of every row; eos between documents",
        }

    def write_meta(path: str | Path, meta: dict) -> None:
        with open(shard_meta_path(path), "w") as f:
            json.dump(meta, f, indent=2)

    t0 = time.perf_counter()
    holdout_meta: dict | None = None
    n_taken = 0

    if holdout_docs > 0:
        assert holdout_path is not None
        writer = _PackedShardWriter(holdout_path, seq_len, max_tokens=None)
        try:
            n_taken = _consume_docs(texts, writer, tokenizer, max_docs=holdout_docs)
        finally:
            writer.close()
        if n_taken < holdout_docs:
            print(
                f"WARNING: stream ended after {n_taken:,} documents; "
                f"wanted {holdout_docs:,} held-out documents."
            )
        holdout_meta = meta_for(writer, (0, n_taken))
        write_meta(holdout_path, holdout_meta)
        print(
            f"[tokenize] holdout: {writer.n_tokens:,} tokens "
            f"({writer.n_seqs:,} seqs, {writer.n_docs:,} docs) -> "
            f"{holdout_path}"
        )

    writer = _PackedShardWriter(out_path, seq_len, max_tokens=n_tokens)
    try:
        n_train_taken = _consume_docs(texts, writer, tokenizer, max_docs=None)
    finally:
        writer.close()
    if not writer.full:
        print(
            f"WARNING: stream exhausted at {writer.n_tokens:,} tokens; "
            f"wanted {n_tokens:,}. Shard written with what there is."
        )
    train_meta = meta_for(writer, (n_taken, n_taken + n_train_taken))
    write_meta(out_path, train_meta)
    print(
        f"[tokenize] train: {writer.n_tokens:,} tokens "
        f"({writer.n_seqs:,} seqs, {writer.n_docs:,} docs) -> {out_path} "
        f"in {time.perf_counter() - t0:.0f}s"
    )
    return {"train": train_meta, "holdout": holdout_meta}


# Reading shards


class TokenShard:
    """A packed token shard on disk: uint16 memmap of shape (n_seqs, seq_len)"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        meta_path = shard_meta_path(self.path)
        if not self.path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"shard needs both {self.path} and {meta_path}; build them "
                f"with `python -m data tokenize ...`"
            )
        with open(meta_path) as f:
            self.meta: dict = json.load(f)
        self.seq_len = int(self.meta["seq_len"])
        flat = np.memmap(self.path, dtype=SHARD_DTYPE, mode="r")
        n_seqs = flat.shape[0] // self.seq_len
        if n_seqs * self.seq_len != flat.shape[0]:
            raise ValueError(
                f"{self.path}: {flat.shape[0]} tokens is not a multiple of "
                f"seq_len={self.seq_len}"
            )
        if n_seqs != int(self.meta["n_seqs"]):
            raise ValueError(
                f"{self.path}: sidecar says {self.meta['n_seqs']} seqs, "
                f"file holds {n_seqs}"
            )
        self._tokens = flat.reshape(n_seqs, self.seq_len)
        self.n_seqs = n_seqs

    @property
    def n_tokens(self) -> int:
        return self.n_seqs * self.seq_len

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, seq_indices) -> torch.Tensor:
        """Rows as int64 (LongTensor). numpy indexing semantics: an int gives."""
        if isinstance(seq_indices, torch.Tensor):
            seq_indices = seq_indices.cpu().numpy()
        rows = np.ascontiguousarray(self._tokens[seq_indices], dtype=np.int64)
        return torch.from_numpy(rows)

    def iter_batches(
        self,
        batch_seqs: int,
        shuffle: bool = True,
        seed: int = 0,
        epochs: int | None = 1,
    ) -> Iterator[torch.Tensor]:
        """Yield (<= batch_seqs, seq_len) LongTensors."""
        if batch_seqs < 1:
            raise ValueError(f"batch_seqs must be >= 1, got {batch_seqs}")
        epoch = 0
        while epochs is None or epoch < epochs:
            if shuffle:
                order = np.random.default_rng(seed + epoch).permutation(self.n_seqs)
            else:
                order = np.arange(self.n_seqs)
            for start in range(0, self.n_seqs, batch_seqs):
                idx = order[start : start + batch_seqs]
                if shuffle:
                    idx = np.sort(idx)
                yield self[idx]
            epoch += 1

# Forward backends


class HFResidualModel:
    """GPT-2 through HuggingFace `GPT2Model` (SDPA attention) instead of
    TransformerLens, exposing exactly what ActivationLoader needs:"""

    def __init__(self, hf_model, center: bool = True) -> None:
        # Accept GPT2LMHeadModel or GPT2Model.
        self.hf = getattr(hf_model, "transformer", hf_model)
        self.center = center
        cfg = self.hf.config
        self.cfg = SimpleNamespace(
            d_model=int(cfg.n_embd), n_layers=int(cfg.n_layer),
            n_ctx=int(cfg.n_positions),
        )
        self.hf.eval()

    @classmethod
    def from_pretrained(cls, name: str = "gpt2", device: str = "cpu",
                        center: bool = True) -> "HFResidualModel":
        from transformers import GPT2Model

        hf = GPT2Model.from_pretrained(name, attn_implementation="sdpa")
        return cls(hf.to(device), center=center)

    def eval(self) -> "HFResidualModel":
        self.hf.eval()
        return self

    def to(self, device) -> "HFResidualModel":
        self.hf.to(device)
        return self

    def parameters(self):
        return self.hf.parameters()

    def resid_post(self, tokens: torch.Tensor, layer: int) -> torch.Tensor:
        """(b, T) ids -> (b, T, d_model) residual after block `layer`, centered to
        match TransformerLens when self.center."""
        if not 0 <= layer < self.cfg.n_layers:
            raise ValueError(f"layer {layer} outside 0..{self.cfg.n_layers - 1}")
        t = self.hf
        pos = torch.arange(tokens.shape[1], device=tokens.device)[None]
        h = t.wte(tokens) + t.wpe(pos)
        for block in t.h[: layer + 1]:
            out = block(h)
            h = out[0] if isinstance(out, tuple) else out
        if self.center:
            h = h - h.mean(dim=-1, keepdim=True)
        return h


def load_forward_model(backend: str, device: str, name: str = "gpt2"):
    """The model the ActivationLoader runs: 'tl' -> HookedTransformer (repo default;
    also what analysis needs), 'hf' -> HFResidualModel."""
    if backend == "tl":
        from transformer_lens import HookedTransformer

        return HookedTransformer.from_pretrained(name).to(device).eval()
    if backend == "hf":
        return HFResidualModel.from_pretrained(name, device=device)
    raise ValueError(f"backend must be 'tl' or 'hf', got {backend!r}")

# Activations


class ActivationLoader:
    """GPT-2 forward over shard rows + on-device shuffle buffer. Iterating the same
    loader twice continues from where it was; it does not restart."""

    def __init__(
        self,
        model: "HookedTransformer | HFResidualModel",
        shard: TokenShard,
        hook_name: str,
        batch_seqs: int,
        batch_tokens: int,
        buffer_tokens: int,
        device: str,
        dtype: torch.dtype = torch.float32,
        exclude_bos: bool = True,
        seed: int = 0,
        epochs: int | None = None,
        autocast: bool = True,
        log_every: int = 200,
        profile: bool = False,
    ) -> None:
        if batch_tokens < 1:
            raise ValueError(f"batch_tokens must be >= 1, got {batch_tokens}")
        if batch_tokens > buffer_tokens // 2:
            raise ValueError(
                f"batch_tokens ({batch_tokens}) must be <= buffer_tokens // 2 "
                f"({buffer_tokens // 2}): the buffer refills when it drops "
                f"below half, so a batch must fit in the other half."
            )
        self.model = model
        self.shard = shard
        self.hook_name = hook_name
        self.layer = layer_from_hook_name(hook_name)
        self.batch_seqs = batch_seqs
        self.batch_tokens = batch_tokens
        self.buffer_tokens = buffer_tokens
        self.device = device
        self.dtype = dtype
        self.exclude_bos = exclude_bos
        self.seed = seed
        self.epochs = epochs
        self.autocast = autocast
        self.log_every = log_every
        self.profile = profile

        self.d_model = int(model.cfg.d_model)
        self._device_type = torch.device(device).type
        self.model.eval()
        model_device = next(model.parameters()).device
        if model_device.type != self._device_type:
            print(
                f"WARNING: ActivationLoader device is {device!r} but the "
                f"model is on {model_device}; activations will be copied "
                f"across every forward. Move the model to {device!r}."
            )
        if self._device_type == "cuda":
            # Any matmul autocast leaves in fp32 (or the fp32 path when autocast=False)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._buffer = torch.empty(
            buffer_tokens, self.d_model, device=device, dtype=dtype
        )
        # Slot bookkeeping: `_free` are slots holding no live activation.
        self._free = torch.arange(buffer_tokens, device=device)
        self._perm = torch.empty(0, dtype=torch.long, device=device)
        self._ptr = 0
        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(seed)

        self._token_iter = shard.iter_batches(
            batch_seqs, shuffle=True, seed=seed, epochs=epochs
        )
        self._exhausted = False
        # Activations from a forward that did not fully fit the buffer.
        self._pending: torch.Tensor | None = None

        # Counters / timing.
        self.tokens_yielded = 0
        self.tokens_forwarded = 0
        self.batches_yielded = 0
        self.n_chunks = 0
        self.n_refills = 0
        self.read_seconds = 0.0
        self.forward_seconds = 0.0
        self._t_start: float | None = None

    # timing
    def wall_time(self) -> float:
        if self._t_start is None:
            return 0.0
        return time.perf_counter() - self._t_start

    def throughput_tok_s(self) -> float:
        wall = self.wall_time()
        return self.tokens_yielded / wall if wall > 0 else 0.0

    # forward
    def _next_chunk(self) -> torch.Tensor | None:
        """Activations of the next shard batch as `dtype` on `device`: (b * (seq_len
        - 1), d) rows, or (b * seq_len, d) with exclude_bos=False."""
        try:
            t0 = time.perf_counter()
            tokens = next(self._token_iter).to(self.device)
            if self.profile:
                self.read_seconds += time.perf_counter() - t0
        except StopIteration:
            self._exhausted = True
            return None

        t0 = time.perf_counter()
        use_autocast = self.autocast and self._device_type == "cuda"
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_autocast
            else nullcontext()
        )
        with torch.no_grad(), ctx:
            if isinstance(self.model, HFResidualModel):
                acts = self.model.resid_post(tokens, self.layer)
            else:
                _, cache = self.model.run_with_cache(
                    tokens,
                    names_filter=self.hook_name,
                    stop_at_layer=self.layer + 1,
                )
                acts = cache[self.hook_name]
        if self.exclude_bos:
            acts = acts[:, 1:]
        # .to(device): TransformerLens moves inputs to the model's own device.
        acts = acts.reshape(-1, self.d_model).to(device=self.device, dtype=self.dtype)
        if self.profile:
            if self._device_type == "cuda":
                torch.cuda.synchronize(self.device)
            self.forward_seconds += time.perf_counter() - t0
        self.tokens_forwarded += tokens.numel()
        self.n_chunks += 1
        return acts

    def _refill(self) -> None:
        """Overwrite every free / already-yielded slot with fresh activations (as
        far as the shard allows) and draw a new permutation over the live slots."""
        dead = torch.cat([self._free, self._perm[: self._ptr]])
        n_dead = dead.shape[0]
        filled = 0
        while filled < n_dead:
            chunk = self._pending
            self._pending = None
            if chunk is None:
                chunk = self._next_chunk()
                if chunk is None:
                    break
            take = min(chunk.shape[0], n_dead - filled)
            self._buffer[dead[filled : filled + take]] = chunk[:take]
            if take < chunk.shape[0]:
                self._pending = chunk[take:]
            filled += take

        live = torch.cat([self._perm[self._ptr :], dead[:filled]])
        self._free = dead[filled:]
        n_live = live.shape[0]
        order = torch.randperm(n_live, generator=self._gen, device=self.device)
        self._perm = live[order]
        self._ptr = 0
        self.n_refills += 1

    @property
    def n_unread(self) -> int:
        """Live rows in the buffer not yet yielded."""
        return self._perm.shape[0] - self._ptr

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        if self._t_start is None:
            self._t_start = time.perf_counter()
        if not self._exhausted and (
            self.n_unread < self.batch_tokens
            or self.n_unread < self.buffer_tokens // 2
        ):
            self._refill()
        if self.n_unread < self.batch_tokens:
            # Only reachable once the shard iterator is exhausted.
            raise StopIteration
        idx = self._perm[self._ptr : self._ptr + self.batch_tokens]
        self._ptr += self.batch_tokens
        batch = self._buffer[idx]  # advanced indexing: an owned copy

        self.tokens_yielded += batch.shape[0]
        self.batches_yielded += 1
        if self.log_every and self.batches_yielded % self.log_every == 0:
            print(
                f"[ActivationLoader] {self.tokens_yielded:,} tok yielded, "
                f"{self.throughput_tok_s():,.0f} tok/s, "
                f"{self.n_refills} refills"
            )
        return batch


# CLI


def _cmd_tokenize(args: argparse.Namespace) -> None:
    result = tokenize_corpus(
        dataset=args.dataset,
        split=args.split,
        out_path=args.out,
        n_tokens=args.n_tokens,
        seq_len=args.seq_len,
        seed=args.seed,
        text_field=args.text_field,
        holdout_docs=args.holdout_docs,
        holdout_path=args.holdout_out,
        streaming=not args.no_streaming,
    )
    print(json.dumps(result, indent=2))


def _cmd_info(args: argparse.Namespace) -> None:
    shard = TokenShard(args.shard)
    print(json.dumps(shard.meta, indent=2))
    first = shard[0].tolist()
    print(f"n_seqs={shard.n_seqs:,} n_tokens={shard.n_tokens:,}")
    print(f"row 0 (first 16 ids): {first[:16]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m data", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tokenize", help="tokenize an HF dataset into shards")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--n-tokens", type=int, required=True)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--text-field", default="text")
    p.add_argument("--holdout-docs", type=int, default=0)
    p.add_argument("--out", default="data/train.bin")
    p.add_argument("--holdout-out", default=None)
    p.add_argument(
        "--no-streaming", action="store_true",
        help="download the full dataset instead of streaming it",
    )
    p.set_defaults(func=_cmd_tokenize)

    p = sub.add_parser("info", help="print a shard's sidecar")
    p.add_argument("shard")
    p.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
