"""eval.py - held-out evaluation of a trained SAE: the metrics the field compares
on, all from one pass over a document-disjoint token shard."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, TYPE_CHECKING

import torch

from data import TokenShard, resid_post_hook
from sparse_autoencoder import SAEConfig, SparseAutoencoder

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


# Keys every `evaluate` result carries (metrics.json schema test).
METRIC_KEYS: tuple[str, ...] = (
    "fvu",
    "variance_explained",
    "mse_raw",
    "mse_scaled",
    "l0",
    "dead_frac_eval",
    "n_dead_eval",
    "n_tokens",
    "n_seqs",
    "ce_clean",
    "ce_recon",
    "ce_zero",
    "loss_recovered",
    "exclude_bos",
    "identity",
    "hook_name",
)

# Top-level keys of results/<run>/metrics.json.
RUN_RECORD_KEYS: tuple[str, ...] = (
    "run",
    "config",
    "metrics",
    "started_at",
    "finished_at",
    "git_sha",
    "train_shard",
    "holdout_shard",
    "checkpoint",
    "training",
)


# Shard disjointness


def check_holdout_disjoint(holdout_meta: dict, train_meta: dict) -> None:
    """Raise ValueError unless the two sidecars certify document-disjoint shards.
    Different dataset: different corpora, disjoint by construction."""
    for name, meta in (("holdout", holdout_meta), ("train", train_meta)):
        if "doc_range" not in meta:
            raise ValueError(f"{name} shard metadata has no 'doc_range'")
    if holdout_meta.get("dataset") != train_meta.get("dataset"):
        return
    for key in ("split", "seed"):
        if holdout_meta.get(key) != train_meta.get(key):
            raise ValueError(
                f"held-out and training shards come from the same dataset "
                f"{train_meta.get('dataset')!r} but differ in {key!r} "
                f"({holdout_meta.get(key)!r} vs {train_meta.get(key)!r}); "
                f"their document ranges index different shuffles, so "
                f"disjointness cannot be certified. Refusing to evaluate."
            )
    h0, h1 = (int(v) for v in holdout_meta["doc_range"])
    t0, t1 = (int(v) for v in train_meta["doc_range"])
    if h1 > h0 and t1 > t0 and max(h0, t0) < min(h1, t1):
        raise ValueError(
            f"held-out shard doc_range [{h0}, {h1}) overlaps the training "
            f"shard doc_range [{t0}, {t1}) - the eval set is not held out. "
            f"Refusing to evaluate."
        )


# Streaming per-dimension moments (Chan / Welford), float64


class _StreamingMoments:
    """Per-dimension running mean and sum of squared deviations (M2) merged batch by
    batch (Chan et al.)"""

    def __init__(self, d: int, device) -> None:
        self.n = 0
        self.mean = torch.zeros(d, dtype=torch.float64, device=device)
        self.m2 = torch.zeros(d, dtype=torch.float64, device=device)

    def update(self, x: torch.Tensor) -> None:
        x = x.double()
        n_b = x.shape[0]
        if n_b == 0:
            return
        mean_b = x.mean(dim=0)
        m2_b = (x - mean_b).pow(2).sum(dim=0)
        n_new = self.n + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * (n_b / n_new)
        self.m2 = self.m2 + m2_b + delta.pow(2) * (self.n * n_b / n_new)
        self.n = n_new

    def total_ss(self) -> float:
        return self.m2.sum().item()


# Evaluation


def _positions_slice(exclude_bos: bool) -> slice:
    """Positions the SAE is applied to (and CE is spliced at): 1.. when position 0."""
    return slice(1, None) if exclude_bos else slice(0, None)


def evaluate(
    sae: SparseAutoencoder,
    model: "HookedTransformer",
    holdout: TokenShard,
    hook_name: str,
    n_tokens: int,
    batch_seqs: int,
    device,
    exclude_bos: bool = True,
    identity: bool = False,
    train_meta: dict | None = None,
    log_every: int = 20,
) -> dict:
    """Held-out metrics of `sae` at `hook_name` of `model` over the first
    ceil(n_tokens / tokens-per-batch) batches of `holdout` (rows in shard order,
    no shuffling, so repeated evaluations see the same tokens)."""
    if n_tokens < 1:
        raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")
    if batch_seqs < 1:
        raise ValueError(f"batch_seqs must be >= 1, got {batch_seqs}")
    if train_meta is not None:
        check_holdout_disjoint(holdout.meta, train_meta)

    sae = sae.to(device)
    sae.eval()
    model.eval()
    d = sae.config.d_model
    if int(model.cfg.d_model) != d:
        raise ValueError(
            f"model d_model {model.cfg.d_model} != sae d_model {d}"
        )
    pos = _positions_slice(exclude_bos)
    per_row = holdout.seq_len - (1 if exclude_bos else 0)
    if n_tokens > holdout.n_seqs * per_row:
        print(
            f"WARNING: --n-tokens {n_tokens:,} exceeds the held-out shard "
            f"({holdout.n_seqs:,} seqs); evaluating on the whole shard."
        )

    moments = _StreamingMoments(d, device)
    resid_ss = torch.zeros((), dtype=torch.float64, device=device)
    l0_sum = 0.0
    active_any = torch.zeros(
        sae.config.n_features, dtype=torch.bool, device=device
    )
    ce_sums = {"clean": 0.0, "recon": 0.0, "zero": 0.0}
    n_ce_positions = 0
    n_eval_tokens = 0
    n_seqs = 0

    def recon_hook(act: torch.Tensor, hook) -> torch.Tensor:
        """Overwrite the evaluated positions with the reconstruction (or the raw
        activation in identity mode) and accumulate SAE stats from the live."""
        nonlocal l0_sum, n_eval_tokens
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            x = act[:, pos].reshape(-1, d).float()
            h = sae.encode(x)
            x_hat = x if identity else sae.postprocess(sae.decode(h))
            moments.update(x)
            resid_ss.add_((x.double() - x_hat.double()).pow(2).sum())
            l0_sum += (h > 0).float().sum(dim=-1).sum().item()
            active_any.logical_or_((h > 0).any(dim=0))
            n_eval_tokens += x.shape[0]
        out = act.clone()
        out[:, pos] = x_hat.reshape(act.shape[0], -1, d).to(act.dtype)
        return out

    def zero_hook(act: torch.Tensor, hook) -> torch.Tensor:
        out = act.clone()
        out[:, pos] = 0.0
        return out

    def run_loss(
        tokens: torch.Tensor, hooks: list[tuple[str, Callable]]
    ) -> tuple[float, int]:
        # per-token loss over the model's own positions (0..T-2 predict 1..T-1)
        loss = model.run_with_hooks(
            tokens, return_type="loss", loss_per_token=True, fwd_hooks=hooks
        )
        return loss.double().sum().item(), loss.numel()

    t0 = time.perf_counter()
    n_batches = 0
    with torch.no_grad():
        for tokens in holdout.iter_batches(batch_seqs, shuffle=False, epochs=1):
            tokens = tokens.to(device)
            s, n_pos = run_loss(tokens, [(hook_name, recon_hook)])
            ce_sums["recon"] += s
            s, _ = run_loss(tokens, [(hook_name, zero_hook)])
            ce_sums["zero"] += s
            s, _ = run_loss(tokens, [])
            ce_sums["clean"] += s
            n_ce_positions += n_pos
            n_seqs += tokens.shape[0]
            n_batches += 1
            if log_every and n_batches % log_every == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"[eval] {n_eval_tokens:,} tokens in {elapsed:.0f}s "
                    f"({n_eval_tokens / max(elapsed, 1e-9):,.0f} tok/s incl. "
                    f"3 forwards)"
                )
            if n_eval_tokens >= n_tokens:
                break
    if n_eval_tokens == 0:
        raise ValueError("held-out shard yielded no tokens")

    total_ss = moments.total_ss()
    fvu = resid_ss.item() / total_ss if total_ss > 0 else float("nan")
    mse_raw = resid_ss.item() / (n_eval_tokens * d)
    scale = sae.input_scale.item() if sae.config.normalize_input else 1.0
    n_dead = int((~active_any).sum().item())
    ce = {k: v / n_ce_positions for k, v in ce_sums.items()}
    denom = ce["zero"] - ce["clean"]
    if denom > 0:
        loss_recovered = (ce["zero"] - ce["recon"]) / denom
    else:
        print(
            f"WARNING: ce_zero ({ce['zero']:.4f}) <= ce_clean "
            f"({ce['clean']:.4f}); loss_recovered is undefined."
        )
        loss_recovered = float("nan")

    return {
        "fvu": fvu,
        "variance_explained": 1.0 - fvu,
        "mse_raw": mse_raw,
        "mse_scaled": mse_raw * scale * scale,
        "l0": l0_sum / n_eval_tokens,
        "dead_frac_eval": n_dead / sae.config.n_features,
        "n_dead_eval": n_dead,
        "n_tokens": n_eval_tokens,
        "n_seqs": n_seqs,
        "ce_clean": ce["clean"],
        "ce_recon": ce["recon"],
        "ce_zero": ce["zero"],
        "loss_recovered": loss_recovered,
        "exclude_bos": exclude_bos,
        "identity": identity,
        "hook_name": hook_name,
    }


def print_metrics(metrics: dict, input_scale: float | None = None) -> None:
    m = metrics
    print(f"Eval tokens          : {m['n_tokens']:,} ({m['n_seqs']:,} seqs; "
          f"position 0 {'excluded' if m['exclude_bos'] else 'INCLUDED'})")
    if input_scale is not None:
        print(f"input_scale          : {input_scale:.5f}")
    if m["identity"]:
        print("MODE                 : identity splice (sanity check)")
    print(f"FVU                  : {m['fvu']:.4f}  (raw space)")
    print(f"Variance explained   : {m['variance_explained']:.4f}  [1 - FVU]")
    print(f"MSE                  : {m['mse_raw']:.4f} raw / "
          f"{m['mse_scaled']:.4f} scaled space")
    print(f"L0 (avg active feats): {m['l0']:.1f}")
    print(f"Dead on eval set     : {m['n_dead_eval']} "
          f"({m['dead_frac_eval'] * 100:.1f}%)")
    print(f"CE clean / recon / zero: {m['ce_clean']:.4f} / "
          f"{m['ce_recon']:.4f} / {m['ce_zero']:.4f} nats")
    print(f"Loss recovered       : {m['loss_recovered']:.4f}")


# Run records (results/<run>/metrics.json)


def git_sha() -> str | None:
    """HEAD commit of the repo this file lives in, or None (no git, not a repo, git
    missing)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def make_run_record(
    run: str,
    config: dict,
    metrics: dict | None,
    started_at: str,
    finished_at: str | None = None,
    train_shard: dict | None = None,
    holdout_shard: dict | None = None,
    checkpoint: str | None = None,
    training: dict | None = None,
) -> dict:
    """Assemble the metrics.json content: config (SAEConfig + CLI args)"""
    if metrics is not None:
        missing = [k for k in METRIC_KEYS if k not in metrics]
        if missing:
            raise ValueError(f"metrics missing keys: {missing}")
    return {
        "run": run,
        "config": dict(config),
        "metrics": None if metrics is None else dict(metrics),
        "started_at": started_at,
        "finished_at": finished_at or utc_now_iso(),
        "git_sha": git_sha(),
        "train_shard": train_shard,
        "holdout_shard": holdout_shard,
        "checkpoint": checkpoint,
        "training": training,
    }


def write_json(path: str | Path, obj: dict) -> Path:
    """Write `obj` as indented JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


# CLI


def load_checkpoint(path: str, device: str) -> tuple[SparseAutoencoder, dict]:
    """(sae, checkpoint dict) from a main.py checkpoint: plain-dict config =>
    weights_only=True; the state_dict carries input_scale."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = SAEConfig(**ckpt["config"])
    sae = SparseAutoencoder(cfg).to(device)
    sae.load_state_dict(ckpt["sae_state_dict"])
    sae.eval()
    return sae, ckpt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m eval", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="sae_gpt2_layer8.pt")
    parser.add_argument("--holdout", default="data/holdout.bin",
                        help="held-out shard (never the training shard)")
    parser.add_argument(
        "--train-shard", default=None,
        help="training shard whose sidecar to check for document overlap; "
        "default: the sidecar stored in the checkpoint, if any",
    )
    parser.add_argument("--n-tokens", type=int, default=2_000_000)
    parser.add_argument("--batch-seqs", type=int, default=64,
                        help="rows per GPT-2 forward (logits are "
                        "b x seq_len x 50257 fp32: 64 rows ~ 1.7 GB)")
    parser.add_argument("--identity", action="store_true",
                        help="splice the raw activation back instead of the "
                        "reconstruction: loss_recovered must be 1, fvu 0")
    parser.add_argument("--include-bos", action="store_true",
                        help="also splice / evaluate position 0 (diagnostic)")
    parser.add_argument("--json", default=None,
                        help="also write the metrics dict to this path")
    parser.add_argument("--device", default=None,
                        help="cuda | cpu (default: cuda if available)")
    args = parser.parse_args(argv)

    from transformer_lens import HookedTransformer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    sae, ckpt = load_checkpoint(args.checkpoint, device)
    hook_name = resid_post_hook(int(ckpt["layer"]))
    print(f"Checkpoint {args.checkpoint}: {sae.config.n_features} features, "
          f"{hook_name}, input_scale={sae.input_scale.item():.5f}")

    holdout = TokenShard(args.holdout)
    train_meta: dict | None = None
    if args.train_shard is not None:
        train_meta = TokenShard(args.train_shard).meta
    elif isinstance(ckpt.get("train_shard"), dict):
        train_meta = ckpt["train_shard"]
    if train_meta is None:
        print("WARNING: no training-shard sidecar available; document "
              "disjointness of the held-out shard was NOT checked.")

    model = HookedTransformer.from_pretrained("gpt2", device=device).eval()
    metrics = evaluate(
        sae, model, holdout, hook_name, n_tokens=args.n_tokens,
        batch_seqs=args.batch_seqs, device=device,
        exclude_bos=not args.include_bos, identity=args.identity,
        train_meta=train_meta,
    )
    print(f"Eval shard           : {args.holdout}")
    print_metrics(metrics, sae.input_scale.item())
    if args.identity:
        ok = (
            abs(metrics["loss_recovered"] - 1.0) <= 1e-3
            and abs(metrics["fvu"]) <= 1e-6
        )
        print(f"identity check       : {'OK' if ok else 'FAILED'}")
        if not ok:
            raise SystemExit(1)
    if args.json:
        write_json(args.json, metrics)
        print(f"metrics written to {args.json}")


if __name__ == "__main__":
    main()
