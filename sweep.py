"""sweep.py - run a list of main.py configurations as subprocesses and keep an index
of what happened. python sweep.py sweeps/frontier.json --results results/ python
sweep.py sweeps/smoke.json --results /tmp/r \ --shard /tmp/tiny/train.bin."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval import utc_now_iso, write_json


NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
INDEX_FILE = "index.jsonl"
ENTRY_FILE = "sweep_entry.json"
LOG_FILE = "train.log"
LOG_TAIL_LINES = 30


# Sweep files


def load_sweep(path: str | Path) -> list[dict]:
    """Entries of a sweep file: a JSON list, or an object with a "runs" list (other
    top-level keys are comments)."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        if "runs" not in raw or not isinstance(raw["runs"], list):
            raise ValueError(
                f"{path}: a sweep object needs a 'runs' list "
                f"(top-level keys: {sorted(raw)})"
            )
        raw = raw["runs"]
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of run entries")
    entries: list[dict] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry {i} is not an object")
        name = entry.get("name")
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise ValueError(
                f"{path}: entry {i} needs a `name` matching {NAME_RE.pattern}, "
                f"got {name!r}"
            )
        if name in seen:
            raise ValueError(f"{path}: duplicate run name {name!r}")
        seen.add(name)
        entries.append({k: v for k, v in entry.items() if not k.startswith("_")})
    return entries


def parse_set(items: list[str]) -> dict:
    """--set key=value pairs -> dict; values parsed as JSON when they are (so 2000
    -> int, 4e-4 -> float, true -> True), otherwise strings."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def resolve_entry(
    entry: dict,
    results_dir: str | Path,
    shard: str | None = None,
    holdout: str | None = None,
    overrides: dict | None = None,
) -> dict:
    """The --config-json content for one entry: the entry itself, plus the layout
    this script owns (run_name, results_dir, checkpoint under results/<name>/)"""
    cfg = dict(entry)
    name = cfg.pop("name")
    run_dir = Path(results_dir) / name
    cfg.setdefault("no_analysis", True)
    if shard is not None:
        cfg["train_shard"] = shard
    if holdout is not None:
        cfg["holdout_shard"] = holdout
    if overrides:
        cfg.update(overrides)
    # Layout keys are not overridable: sweep.py owns where runs land.
    cfg["run_name"] = name
    cfg["results_dir"] = str(results_dir)
    cfg["checkpoint"] = str(run_dir / "checkpoint.pt")
    return cfg


def command_for(cfg_path: str | Path, trainer: str, python: str) -> list[str]:
    """`python -u trainer --config-json cfg`: -u so the subprocess's stdout."""
    return [python, "-u", trainer, "--config-json", str(cfg_path)]


# Running


def _tail(path: Path, n: int = LOG_TAIL_LINES) -> str:
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def _headline(metrics_path: Path) -> dict:
    """l0 / fvu / loss_recovered from a metrics.json (empty when the evaluation was
    skipped or the file is unreadable)."""
    try:
        with open(metrics_path) as f:
            record = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    m = record.get("metrics") or {}
    return {k: m.get(k) for k in ("l0", "fvu", "loss_recovered") if k in m}


def run_entry(
    entry: dict,
    results_dir: str | Path,
    trainer: str = "main.py",
    python: str | None = None,
    force: bool = False,
    shard: str | None = None,
    holdout: str | None = None,
    overrides: dict | None = None,
    env: dict | None = None,
    dry_run: bool = False,
) -> dict:
    """Run one entry as a subprocess and return its index line (status ok | failed |
    skipped | dry-run). Never raises for a failed run; the caller decides what to
    do with the status."""
    name = entry["name"]
    run_dir = Path(results_dir) / name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return {
            "name": name, "status": "skipped",
            "reason": f"{metrics_path} exists (use --force)",
            "metrics_path": str(metrics_path), "at": utc_now_iso(),
        }
    cfg = resolve_entry(entry, results_dir, shard, holdout, overrides)
    cfg_path = run_dir / ENTRY_FILE
    cmd = command_for(cfg_path, trainer, python or sys.executable)
    if dry_run:
        return {"name": name, "status": "dry-run", "command": cmd, "config": cfg}

    write_json(cfg_path, cfg)
    log_path = run_dir / LOG_FILE
    started = utc_now_iso()
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        try:
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                env=None if env is None else {**os.environ, **env},
                check=False,
            )
            returncode = proc.returncode
        except OSError as exc:  # interpreter / script not found
            log.write(f"sweep.py: could not start {cmd}: {exc}\n")
            returncode = -1
    wall_min = (time.perf_counter() - t0) / 60.0
    line = {
        "name": name,
        "returncode": returncode,
        "wall_min": wall_min,
        "started_at": started,
        "finished_at": utc_now_iso(),
        "command": cmd,
        "log": str(log_path),
        "metrics_path": str(metrics_path),
    }
    if returncode == 0 and metrics_path.exists():
        line["status"] = "ok"
        line.update(_headline(metrics_path))
    else:
        line["status"] = "failed"
        line["error"] = (
            f"exit {returncode}" if returncode != 0
            else "exited 0 but wrote no metrics.json"
        )
        line["log_tail"] = _tail(log_path)
    return line


def run_sweep(
    entries: list[dict],
    results_dir: str | Path,
    trainer: str = "main.py",
    python: str | None = None,
    force: bool = False,
    parallel: int = 1,
    gpus: list[str] | None = None,
    shard: str | None = None,
    holdout: str | None = None,
    overrides: dict | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Run every entry (sequentially, or `parallel` at a time), append each result
    line to results/index.jsonl as it finishes, print a one-line summary per run,
    and return the lines in entry order."""
    if parallel < 1:
        raise ValueError(f"parallel must be >= 1, got {parallel}")
    results_dir = Path(results_dir)
    index_path = results_dir / INDEX_FILE
    if not dry_run:
        results_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    gpu_pool: queue.Queue | None = None
    if gpus:
        gpu_pool = queue.Queue()
        for g in gpus:
            gpu_pool.put(g)

    def one(entry: dict) -> dict:
        env = None
        gpu = None
        if gpu_pool is not None:
            gpu = gpu_pool.get()
            env = {"CUDA_VISIBLE_DEVICES": gpu}
        try:
            line = run_entry(
                entry, results_dir, trainer=trainer, python=python, force=force,
                shard=shard, holdout=holdout, overrides=overrides, env=env,
                dry_run=dry_run,
            )
        finally:
            if gpu_pool is not None:
                gpu_pool.put(gpu)
        if gpu is not None:
            line["gpu"] = gpu
        with lock:
            if dry_run:
                print(f"[sweep] {line['name']}: {' '.join(line['command'])}", flush=True)
            else:
                with open(index_path, "a") as f:
                    f.write(json.dumps(line) + "\n")
                _print_line(line)
        return line

    if parallel == 1:
        return [one(e) for e in entries]
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        return list(pool.map(one, entries))


def _print_line(line: dict) -> None:
    """One console line per finished run (flushed: the sweep is usually started
    under nohup with stdout redirected to a file)."""
    status = line["status"]
    if status == "ok":
        extra = ", ".join(
            f"{k}={line[k]:.4g}" for k in ("l0", "fvu", "loss_recovered")
            if isinstance(line.get(k), (int, float))
        )
        print(f"[sweep] {line['name']}: ok in {line['wall_min']:.1f} min"
              + (f" ({extra})" if extra else ""), flush=True)
    elif status == "skipped":
        print(f"[sweep] {line['name']}: skipped, {line['reason']}", flush=True)
    else:
        print(f"[sweep] {line['name']}: FAILED ({line['error']}), see "
              f"{line['log']}", flush=True)
        tail = line.get("log_tail", "").rstrip()
        if tail:
            print("        " + tail.replace("\n", "\n        "), flush=True)


# CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python sweep.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sweep", help="JSON sweep file (see module doc)")
    parser.add_argument("--results", default="results",
                        help="parent directory of results/<name>/ (default "
                        "results)")
    parser.add_argument("--shard", default=None,
                        help="train shard for every run (main.py --train-shard)")
    parser.add_argument("--holdout", default=None,
                        help="held-out shard for every run (main.py "
                        "--holdout-shard)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a main.py flag for every run "
                        "(repeatable; value parsed as JSON when possible)")
    parser.add_argument("--force", action="store_true",
                        help="re-run entries whose metrics.json exists")
    parser.add_argument("--parallel", type=int, default=1,
                        help="subprocesses at a time; one GPU per process "
                        "assumed - keep 1 on a single-GPU box")
    parser.add_argument("--gpus", default=None,
                        help="comma-separated device ids to round-robin "
                        "worker slots over (CUDA_VISIBLE_DEVICES), e.g. 0,1")
    parser.add_argument("--trainer", default="main.py",
                        help="script run per entry (default main.py; the "
                        "tests point it at a stub)")
    parser.add_argument("--python", default=None,
                        help="interpreter for the subprocesses (default: "
                        "this one)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands, write nothing")
    args = parser.parse_args(argv)

    entries = load_sweep(args.sweep)
    overrides = parse_set(args.set)
    gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else None
    if gpus and args.parallel > len(gpus):
        print(f"NOTE: --parallel {args.parallel} > {len(gpus)} GPUs; slots "
              f"share devices.")
    print(f"[sweep] {len(entries)} entries from {args.sweep} -> {args.results}"
          + (f" ({args.parallel} at a time)" if args.parallel > 1 else ""),
          flush=True)
    lines = run_sweep(
        entries, args.results, trainer=args.trainer, python=args.python,
        force=args.force, parallel=args.parallel, gpus=gpus, shard=args.shard,
        holdout=args.holdout, overrides=overrides, dry_run=args.dry_run,
    )
    n = {s: sum(1 for l in lines if l["status"] == s)
         for s in ("ok", "failed", "skipped", "dry-run")}
    print(f"[sweep] done: {n['ok']} ok, {n['failed']} failed, "
          f"{n['skipped']} skipped" + (f", {n['dry-run']} dry-run" if n["dry-run"] else "")
          + (f"; index at {Path(args.results) / INDEX_FILE}" if not args.dry_run else ""),
          flush=True)
    return 1 if n["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
