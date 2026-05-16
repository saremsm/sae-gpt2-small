"""eval.py - held-out evaluation of a trained SAE: the metrics the field compares
on, all from one pass over a document-disjoint token shard."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterator, TYPE_CHECKING

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
    "torch_version",
    "gpu_name",
    "train_shard",
    "holdout_shard",
    "checkpoint",
    "training",
)

# Keys of a main.py checkpoint (make_checkpoint / load_checkpoint).
CHECKPOINT_KEYS: tuple[str, ...] = (
    "sae_state_dict",
    "config",
    "layer",
    "hook_name",
    "input_scale",
    "activation",
    "k",
    "git_sha",
    "torch_version",
    "gpu_name",
    "device",
    "created_at",
    "run_config",
    "training_history",
    "train_shard",
)

# Default tolerance of the reproducibility check (`python -m eval --compare`)
COMPARE_TOL = 1e-4


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


@contextmanager
def fp32_matmul() -> Iterator[None]:
    """Disable TF32 matmuls (torch.backends.cuda.matmul / cudnn allow_tf32) for the
    block and restore the previous flags after. data.ActivationLoader enables."""
    if not torch.cuda.is_available():
        yield
        return
    prev = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev[0]
        torch.backends.cudnn.allow_tf32 = prev[1]


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
    with torch.no_grad(), fp32_matmul():
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


def torch_version() -> str:
    return str(torch.__version__)


def gpu_name(device: str | torch.device | None) -> str | None:
    """torch.cuda.get_device_name of `device` when it is a CUDA device and CUDA is
    available; None otherwise (cpu / mps / no device)."""
    if device is None:
        return None
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(dev)
    except (RuntimeError, AssertionError):
        return None


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
    device: str | None = None,
) -> dict:
    """Assemble the metrics.json content: config (SAEConfig + CLI args), every
    `evaluate` metric (None when the evaluation was skipped, e.g. no held-out
    shard), ISO-8601 UTC timestamps, git SHA (None when unavailable)"""
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
        "torch_version": torch_version(),
        "gpu_name": gpu_name(device),
        "train_shard": train_shard,
        "holdout_shard": holdout_shard,
        "checkpoint": checkpoint,
        "training": training,
    }


def make_checkpoint(
    sae: SparseAutoencoder,
    layer: int,
    hook_name: str,
    run_config: dict | None = None,
    training_history: dict | None = None,
    train_shard: dict | None = None,
    device: str | None = None,
) -> dict:
    """The dict main.py torch.save()s (CHECKPOINT_KEYS): the SAE state_dict (which
    carries the input_scale buffer), the config as a plain dict (asdict, so
    torch.load(weights_only=True) works), the layer and hook name."""
    config = sae.config
    return {
        "sae_state_dict": sae.state_dict(),
        "config": asdict(config),
        "layer": int(layer),
        "hook_name": hook_name,
        "input_scale": float(sae.input_scale.item()),
        "activation": config.activation,
        "k": config.k,
        "git_sha": git_sha(),
        "torch_version": torch_version(),
        "gpu_name": gpu_name(device),
        "device": None if device is None else str(device),
        "created_at": utc_now_iso(),
        "run_config": None if run_config is None else dict(run_config),
        "training_history": training_history,
        "train_shard": train_shard,
    }


def compare_metrics(
    reference: dict, actual: dict, tol: float = COMPARE_TOL
) -> list[str]:
    """Mismatches between two `evaluate` dicts, one string per METRIC_KEY that
    differs: floats beyond `tol` (absolute; NaN equals NaN), ints / bools /
    strings not equal, keys missing from either side."""
    problems: list[str] = []
    for key in METRIC_KEYS:
        if key not in reference or key not in actual:
            side = "reference" if key not in reference else "actual"
            problems.append(f"{key}: missing from {side}")
            continue
        r, a = reference[key], actual[key]
        if isinstance(r, bool) or isinstance(a, bool) or isinstance(r, str):
            if r != a:
                problems.append(f"{key}: {r!r} != {a!r}")
        elif isinstance(r, int) and isinstance(a, int):
            if r != a:
                problems.append(f"{key}: {r} != {a}")
        else:
            r_f, a_f = float(r), float(a)
            if math.isnan(r_f) and math.isnan(a_f):
                continue
            if not abs(r_f - a_f) <= tol:
                problems.append(
                    f"{key}: recorded {r_f:.6g}, got {a_f:.6g} "
                    f"(|diff| {abs(r_f - a_f):.3g} > {tol:g})"
                )
    return problems


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


def _checkpoint_arg(ckpt: dict, name: str):
    """A main.py CLI value the checkpoint recorded (run_config.args), or None when
    the checkpoint predates run_config."""
    run_config = ckpt.get("run_config")
    if not isinstance(run_config, dict):
        return None
    args = run_config.get("args")
    if not isinstance(args, dict):
        return None
    return args.get(name)


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
    parser.add_argument("--n-tokens", type=int, default=None,
                        help="held-out tokens to evaluate (position 0 not "
                        "counted); default: the eval_tokens the checkpoint "
                        "recorded, else 2,000,000")
    parser.add_argument("--batch-seqs", type=int, default=None,
                        help="rows per GPT-2 forward (logits are b x seq_len "
                        "x 50257 fp32: 64 rows ~ 1.7 GB); default: the "
                        "eval_batch_seqs the checkpoint recorded, else 64. "
                        "Part of the reproducibility contract - the eval "
                        "set is cut in whole batches")
    parser.add_argument("--identity", action="store_true",
                        help="splice the raw activation back instead of the "
                        "reconstruction: loss_recovered must be 1, fvu 0")
    parser.add_argument("--include-bos", action="store_true",
                        help="also splice / evaluate position 0 (diagnostic)")
    parser.add_argument("--compare", default=None, metavar="METRICS_JSON",
                        help="a results/<run>/metrics.json (or a bare metrics "
                        "dict written by --json): check that this evaluation "
                        "reproduces its numbers within --tol and exit 1 if not")
    parser.add_argument("--tol", type=float, default=COMPARE_TOL,
                        help=f"absolute tolerance for --compare (default "
                        f"{COMPARE_TOL:g})")
    parser.add_argument("--json", default=None,
                        help="also write the metrics dict to this path")
    parser.add_argument("--device", default=None,
                        help="cuda | cpu (default: cuda if available)")
    args = parser.parse_args(argv)

    from transformer_lens import HookedTransformer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    sae, ckpt = load_checkpoint(args.checkpoint, device)
    hook_name = ckpt.get("hook_name") or resid_post_hook(int(ckpt["layer"]))
    print(f"Checkpoint {args.checkpoint}: {sae.config.n_features} features, "
          f"{sae.config.activation}"
          + (f" k={sae.config.k}" if sae.config.k is not None else "")
          + f", {hook_name}, input_scale={sae.input_scale.item():.5f}")
    print(f"  trained with : git {ckpt.get('git_sha') or '?'}, torch "
          f"{ckpt.get('torch_version') or '?'}, gpu "
          f"{ckpt.get('gpu_name') or 'none/unknown'}"
          + (f", {ckpt['created_at']}" if ckpt.get("created_at") else ""))
    print(f"  evaluating on: git {git_sha() or '?'}, torch {torch_version()}, "
          f"gpu {gpu_name(device) or 'none'}")

    n_tokens = args.n_tokens
    if n_tokens is None:
        n_tokens = _checkpoint_arg(ckpt, "eval_tokens") or 2_000_000
    batch_seqs = args.batch_seqs
    if batch_seqs is None:
        batch_seqs = _checkpoint_arg(ckpt, "eval_batch_seqs") or 64
    print(f"Eval set             : first {n_tokens:,} tokens in batches of "
          f"{batch_seqs} rows")

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
        sae, model, holdout, hook_name, n_tokens=n_tokens,
        batch_seqs=batch_seqs, device=device,
        exclude_bos=not args.include_bos, identity=args.identity,
        train_meta=train_meta,
    )
    print(f"Eval shard           : {args.holdout}")
    print_metrics(metrics, sae.input_scale.item())
    ok = True
    if args.identity:
        ok = (
            abs(metrics["loss_recovered"] - 1.0) <= 1e-3
            and abs(metrics["fvu"]) <= 1e-6
        )
        print(f"identity check       : {'OK' if ok else 'FAILED'}")
    if args.json:
        write_json(args.json, metrics)
        print(f"metrics written to {args.json}")
    if args.compare is not None:
        with open(args.compare) as f:
            record = json.load(f)
        # a run record (metrics.json) or a bare metrics dict (--json).
        reference = record.get("metrics") if "run" in record else record
        if not isinstance(reference, dict):
            raise SystemExit(
                f"{args.compare}: no metrics to compare against (the run "
                f"was evaluated without a held-out shard?)"
            )
        problems = compare_metrics(reference, metrics, tol=args.tol)
        if problems:
            print(f"reproduces {args.compare}: MISMATCH (tol {args.tol:g})")
            for line in problems:
                print(f"  {line}")
            ok = False
        else:
            print(f"reproduces {args.compare}: OK (every metric within "
                  f"{args.tol:g})")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
