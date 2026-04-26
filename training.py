from __future__ import annotations

from collections import deque
from typing import Iterator, Protocol, TypedDict, runtime_checkable

import torch

from sparse_autoencoder import SAEOutput, SparseAutoencoder
from torch.optim import AdamW
from tqdm import tqdm


@runtime_checkable
class ActivationSource(Protocol):
    """Iterable of (batch_tokens, d_model) float32 tensors of RAW residual
    activations, already shuffled and already on the training device."""

    batch_tokens: int

    def __iter__(self) -> Iterator[torch.Tensor]:
        ...


# Default SAE batch (activations per optimizer step)
BATCH_SIZE = 512
RENORM_INTERVAL = 100
# batches feeding the resampling pool. 8 x 512 x 768 fp32 ~ 12 MB, kept on the
# training device (a per-step .cpu() would sync every step).
RESAMPLE_POOL_BATCHES = 8


class TrainingHistory(TypedDict):
    step: list[int]
    loss: list[float]
    reconstruction_loss: list[float]
    sparsity_loss: list[float]
    # AuxK term (unweighted MSE); all zeros unless aux_k > 0.
    aux_loss: list[float]
    l0: list[float]
    dead_features: list[int]
    act_norm: list[float]


def _make_lr_lambda(warmup_steps: int):
    def lr_lambda(step: int) -> float:
        return min(1.0, (step + 1) / max(1, warmup_steps))

    return lr_lambda


def train_step(
    sae: SparseAutoencoder,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
) -> SAEOutput:
    """One optimizer step on a batch of raw residuals: forward, backward, decoder-
    gradient projection, grad clipping, optimizer.step()."""
    output = sae(batch)
    optimizer.zero_grad(set_to_none=True)
    output.loss.backward()
    if sae.config.normalize_decoder:
        sae.project_decoder_grad()
    torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
    optimizer.step()
    return output


def train_sae(
    sae: SparseAutoencoder,
    loader: ActivationSource,
    n_training_tokens: int = 5_000_000,
    resample_interval: int = 5_000,
    log_interval: int = 100,
    device: str = "cpu",
    seed: int = 42,
    calibration_tokens: int = 100_000,
) -> TrainingHistory:
    """Train `sae` on raw residual batches from `loader`."""
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    sae = sae.to(device)
    sae.train()

    optimizer = AdamW(sae.parameters(), lr=sae.config.lr, weight_decay=0.0)

    # Guard a silent failure: the original warmup_steps=1000 vs ~976-step run
    batch_tokens = loader.batch_tokens
    expected_steps = max(1, n_training_tokens // batch_tokens)
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
        "aux_loss": [],
        "l0": [],
        "dead_features": [],
        "act_norm": [],
    }

    src_iter = iter(loader)

    # Calibrate the SAE's dataset-wide input scale (sqrt(d) / mean ||x||)
    stash: deque[torch.Tensor] = deque()
    n_calib = 0
    if calibration_tokens > 0:
        print(
            f"Calibrating input_scale on >= {calibration_tokens:,} raw "
            f"tokens..."
        )
        while n_calib < calibration_tokens:
            try:
                batch = next(src_iter)
            except StopIteration:
                break
            stash.append(batch)
            n_calib += batch.shape[0]
        if n_calib == 0:
            print(
                "WARNING: loader yielded no tokens; input_scale left at "
                f"{sae.input_scale.item():.4g}."
            )
        else:
            if n_calib < calibration_tokens:
                print(
                    f"WARNING: only {n_calib:,} tokens available for "
                    f"input_scale calibration (wanted {calibration_tokens:,}); "
                    f"the loader is exhausted, calibrating on what there is."
                )
            sae.set_input_scale_from_activations(torch.cat(list(stash)))

    def batches() -> Iterator[torch.Tensor]:
        # popleft so calibration batches are freed as they are consumed rather than
        while stash:
            yield stash.popleft()
        yield from src_iter

    print("=" * 60)
    print("Training SAE")
    print(
        f"  Features      : {sae.config.n_features} "
        f"(d_model={sae.config.d_model})"
    )
    if sae.config.activation == "topk":
        aux_note = (
            f"AuxK aux_k={sae.config.aux_k}, aux_coeff={sae.config.aux_coeff}"
            if sae.config.aux_k > 0 else "AuxK off"
        )
        print(f"  Activation    : topk, k={sae.config.k} ({aux_note})")
    else:
        print(
            f"  Activation    : relu, l1_coefficient="
            f"{sae.config.l1_coefficient}"
        )
    print(
        f"  LR schedule   : {sae.config.lr} with "
        f"{warmup_steps}-step warmup"
    )
    print(f"  Target tokens : {n_training_tokens:,} ({batch_tokens} per step)")
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

    batch_iter = batches()
    while tokens_trained < n_training_tokens:
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        # Raw residuals: the SAE applies input_scale internally.
        batch = batch.to(device)

        output = train_step(sae, optimizer, batch)
        scheduler.step()

        if sae.config.normalize_decoder and step % RENORM_INTERVAL == 0:
            sae.normalize_decoder()

        # Loader batches are owned tensors and per_token_recon_error is already.
        resample_pool.append((batch.detach(), output.per_token_recon_error))

        step += 1
        tokens_trained += batch.shape[0]
        pbar.update(batch.shape[0])

        if step % resample_interval == 0:
            dead_indices = sae.get_dead_features(threshold=0)
            if len(dead_indices) > 0:
                pool_acts = torch.cat([acts for acts, _ in resample_pool])
                pool_errors = torch.cat([errs for _, errs in resample_pool])
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
            history["aux_loss"].append(output.aux_loss.item())
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
    if tokens_trained < n_training_tokens:
        print(
            f"WARNING: loader ended after {tokens_trained:,} tokens "
            f"(wanted {n_training_tokens:,})."
        )
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
