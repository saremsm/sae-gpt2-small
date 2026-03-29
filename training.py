from __future__ import annotations

from collections import deque
from typing import Iterator, Protocol, TypedDict, runtime_checkable

import torch

from sparse_autoencoder import SparseAutoencoder
from torch.optim import AdamW
from tqdm import tqdm

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
        self._epoch = 0

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
        # no trust_remote_code: datasets>=3 ignores it and prints a warning on every.
        dataset = load_dataset(self.dataset_name, split="train")
        if self._epoch > 0:
            # Re-shuffle on repeat passes: epoch 2 must not replay epoch 1 byte-
            dataset = dataset.shuffle(seed=self._epoch)
        self._epoch += 1
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


BATCH_SIZE = 512
BUFFER_TARGET_TOKENS = 65_536
RENORM_INTERVAL = 100
# batches feeding the resampling pool. 8 x 512 x 768 fp32 ~ 12 MB.
RESAMPLE_POOL_BATCHES = 8


class TrainingHistory(TypedDict):
    step: list[int]
    loss: list[float]
    reconstruction_loss: list[float]
    sparsity_loss: list[float]
    l0: list[float]
    dead_features: list[int]
    act_norm: list[float]


def _make_lr_lambda(warmup_steps: int):
    def lr_lambda(step: int) -> float:
        return min(1.0, (step + 1) / max(1, warmup_steps))

    return lr_lambda


def train_sae(
    sae: SparseAutoencoder,
    activation_source: ActivationSource,
    n_training_tokens: int = 5_000_000,
    resample_interval: int = 5_000,
    log_interval: int = 100,
    device: str = "cpu",
    seed: int = 42,
    calibration_tokens: int = 100_000,
) -> TrainingHistory:
    """Train `sae` on raw residuals from `activation_source`."""
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    sae = sae.to(device)
    sae.train()

    optimizer = AdamW(sae.parameters(), lr=sae.config.lr, weight_decay=0.0)

    # Guard a silent failure: the original warmup_steps=1000 vs ~976-step run
    expected_steps = max(1, n_training_tokens // BATCH_SIZE)
    warmup_steps = sae.config.warmup_steps
    if warmup_steps >= expected_steps:
        clamped = max(10, expected_steps // 20)
        print(
            f"WARNING: warmup_steps ({warmup_steps}) >= expected steps "
            f"({expected_steps}); clamping warmup to {clamped} so the run "
            f"actually reaches lr={sae.config.lr}."
        )
        warmup_steps = clamped

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_make_lr_lambda(warmup_steps)
    )

    history: TrainingHistory = {
        "step": [],
        "loss": [],
        "reconstruction_loss": [],
        "sparsity_loss": [],
        "l0": [],
        "dead_features": [],
        "act_norm": [],
    }

    src_iter = iter(activation_source)
    buffer = torch.empty(0, sae.config.d_model)
    buffer_offset = 0

    def refill_buffer(target_tokens: int = BUFFER_TARGET_TOKENS) -> None:
        nonlocal buffer, buffer_offset, src_iter

        chunks: list[torch.Tensor] = []
        if buffer_offset < buffer.shape[0]:
            chunks.append(buffer[buffer_offset:])
        buffer_offset = 0

        total = sum(c.shape[0] for c in chunks)
        restarts = 0
        while total < target_tokens:
            try:
                chunk = next(src_iter)
            except StopIteration:
                if restarts >= 1:
                    break
                src_iter = iter(activation_source)
                restarts += 1
                continue
            chunks.append(chunk)
            total += chunk.shape[0]

        if not chunks or total == 0:
            buffer = torch.empty(0, sae.config.d_model)
            return

        flat = torch.cat(chunks, dim=0)
        perm = torch.randperm(flat.shape[0])
        buffer = flat[perm].contiguous()

    # Calibrate the SAE's dataset-wide input scale (sqrt(d) / mean ||x||)
    n_calib = 0
    if calibration_tokens > 0:
        print(
            f"Calibrating input_scale on >= {calibration_tokens:,} raw "
            f"tokens..."
        )
        refill_buffer(max(BUFFER_TARGET_TOKENS, calibration_tokens))
        n_calib = buffer.shape[0]
        if n_calib == 0:
            print(
                "WARNING: activation source yielded no tokens; input_scale "
                f"left at {sae.input_scale.item():.4g}."
            )
        else:
            if n_calib < calibration_tokens:
                print(
                    f"WARNING: only {n_calib:,} tokens available for "
                    f"input_scale calibration (wanted {calibration_tokens:,}); "
                    f"the source is exhausted, calibrating on what there is."
                )
            sae.set_input_scale_from_activations(buffer)

    print("=" * 60)
    print("Training SAE")
    print(
        f"  Features      : {sae.config.n_features} "
        f"(d_model={sae.config.d_model})"
    )
    print(
        f"  LR schedule   : {sae.config.lr} with "
        f"{warmup_steps}-step warmup"
    )
    print(f"  Target tokens : {n_training_tokens:,}")
    calib_note = (
        f"calibrated on {n_calib:,} tokens" if n_calib
        else "not calibrated in this call"
    )
    print(
        f"  Input scaling : x * input_scale inside the SAE, "
        f"input_scale={sae.input_scale.item():.4g} "
        f"({calib_note}; normalize_input={sae.config.normalize_input})"
    )
    print("  Loss / MSE    : reported in scaled space")
    print()

    # Rolling resampling pool: raw activations + their per-token errors from
    resample_pool: deque[tuple[torch.Tensor, torch.Tensor]] = deque(
        maxlen=RESAMPLE_POOL_BATCHES
    )

    step = 0
    tokens_trained = 0
    pbar = tqdm(total=n_training_tokens, unit="tok")

    while tokens_trained < n_training_tokens:
        if buffer.shape[0] - buffer_offset < BATCH_SIZE:
            refill_buffer()
            if buffer.shape[0] - buffer_offset < BATCH_SIZE:
                break

        # Raw residuals: the SAE applies input_scale internally.
        batch = buffer[buffer_offset : buffer_offset + BATCH_SIZE].to(device)
        buffer_offset += BATCH_SIZE

        output = sae(batch)

        optimizer.zero_grad()
        output.loss.backward()
        if sae.config.normalize_decoder:
            sae.project_decoder_grad()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if sae.config.normalize_decoder and step % RENORM_INTERVAL == 0:
            sae.normalize_decoder()

        # clone(): CPU-path .cpu() is a no-op VIEW into the 64K-token buffer.
        resample_pool.append(
            (
                batch.detach().cpu().clone(),
                output.per_token_recon_error.cpu().clone(),
            )
        )

        step += 1
        tokens_trained += batch.shape[0]
        pbar.update(batch.shape[0])

        if step % resample_interval == 0:
            dead_indices = sae.get_dead_features(threshold=0)
            if len(dead_indices) > 0:
                pool_acts = torch.cat(
                    [acts for acts, _ in resample_pool]
                ).to(device)
                pool_errors = torch.cat(
                    [errs for _, errs in resample_pool]
                ).to(device)
                sae.resample_dead_features(
                    dead_feature_indices=dead_indices,
                    activations=pool_acts,
                    errors=pool_errors,
                    optimizer=optimizer,
                )
            else:
                # Fresh counting window regardless: counts were zeroed only when a resample
                # fired.
                sae.feature_activation_counts.zero_()

        if step % log_interval == 0:
            n_dead = len(sae.get_dead_features(threshold=0))
            current_lr = scheduler.get_last_lr()[0]
            act_norm = batch.norm(dim=-1).mean().item()
            history["step"].append(step)
            history["loss"].append(output.loss.item())
            history["reconstruction_loss"].append(
                output.reconstruction_loss.item()
            )
            history["sparsity_loss"].append(output.sparsity_loss.item())
            history["l0"].append(output.l0.item())
            history["dead_features"].append(n_dead)
            history["act_norm"].append(act_norm)
            pbar.set_postfix(
                {
                    "loss": f"{output.loss.item():.3f}",
                    "L0": f"{output.l0.item():.1f}",
                    "dead": n_dead,
                    "act_norm": f"{act_norm:.1f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

    pbar.close()
    if history["loss"]:
        print("\nTraining complete.")
        print(f"  Final loss        : {history['loss'][-1]:.4f} (scaled space)")
        print(
            f"  Final recon MSE   : {history['reconstruction_loss'][-1]:.4f} "
            f"(scaled space)"
        )
        print(f"  Final L0          : {history['l0'][-1]:.1f}")
        print(
            f"  Dead features     : {history['dead_features'][-1]} / "
            f"{sae.config.n_features}"
        )
    return history
