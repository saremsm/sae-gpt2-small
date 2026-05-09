"""plot.py - the frontier figures and tables the README publishes."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TABLE_COLUMNS: tuple[str, ...] = (
    "name", "activation", "k/λ", "features", "n_tokens", "L0", "1-FVU",
    "loss_recovered", "ce_clean", "ce_recon", "ce_zero", "dead_frac_eval",
    "tok/s", "wall_min",
)
ACTIVATION_MARKERS = {"relu": "o", "topk": "s"}
DEFAULT_MARKER = "^"


@dataclass
class RunRow:
    """One run's table row, all read from its metrics.json."""
    name: str
    activation: str
    k: int | None
    l1_coefficient: float
    n_features: int
    expansion: float
    n_tokens: int
    l0: float
    variance_explained: float
    loss_recovered: float
    ce_clean: float
    ce_recon: float
    ce_zero: float
    dead_frac_eval: float
    tok_s: float | None
    wall_min: float | None
    path: str

    @property
    def sparsity_param(self) -> str:
        if self.activation == "topk":
            return f"k={self.k}"
        return f"λ={self.l1_coefficient:.3g}"


def _num(value, default=float("nan")) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def load_run(path: str | Path) -> RunRow | None:
    """RunRow from one metrics.json, or None when its `metrics` is null (evaluation
    skipped)."""
    path = Path(path)
    with open(path) as f:
        record = json.load(f)
    metrics = record.get("metrics")
    if not metrics:
        return None
    config = record.get("config") or {}
    sae = config.get("sae") or {}
    args = config.get("args") or {}
    training = record.get("training") or {}
    d_model = int(sae.get("d_model") or 768)
    n_features = int(sae.get("n_features") or 0)
    n_tokens = training.get("tokens") or args.get("n_tokens") or 0
    wall = training.get("train_wall_seconds")
    return RunRow(
        name=str(record.get("run") or path.parent.name),
        activation=str(sae.get("activation") or "relu"),
        k=sae.get("k"),
        l1_coefficient=_num(sae.get("l1_coefficient"), 0.0),
        n_features=n_features,
        expansion=n_features / d_model if d_model else float("nan"),
        n_tokens=int(n_tokens),
        l0=_num(metrics.get("l0")),
        variance_explained=_num(metrics.get("variance_explained")),
        loss_recovered=_num(metrics.get("loss_recovered")),
        ce_clean=_num(metrics.get("ce_clean")),
        ce_recon=_num(metrics.get("ce_recon")),
        ce_zero=_num(metrics.get("ce_zero")),
        dead_frac_eval=_num(metrics.get("dead_frac_eval")),
        tok_s=training.get("loader_tok_s"),
        wall_min=None if wall is None else float(wall) / 60.0,
        path=str(path),
    )


def load_runs(results_dir: str | Path) -> tuple[list[RunRow], list[str]]:
    """(rows sorted by expansion, activation, L0; names of runs whose metrics were
    null) from every results/*/metrics.json."""
    rows: list[RunRow] = []
    unevaluated: list[str] = []
    for path in sorted(Path(results_dir).glob("*/metrics.json")):
        row = load_run(path)
        if row is None:
            unevaluated.append(path.parent.name)
        else:
            rows.append(row)
    rows.sort(key=lambda r: (r.expansion, r.activation, r.l0))
    return rows, unevaluated


# Figures


def _colors(rows: list[RunRow]) -> dict[float, tuple]:
    widths = sorted({r.expansion for r in rows})
    cmap = plt.get_cmap("tab10")
    return {w: cmap(i % 10) for i, w in enumerate(widths)}


def frontier_figure(
    rows: list[RunRow],
    ykey: str,
    ylabel: str,
    title: str,
    out_path: str | Path,
) -> Path:
    """Scatter of L0 vs `ykey` (a RunRow attribute), colored by expansion, marker by
    activation, each point annotated with its name; two legends (widths,
    activations)."""
    colors = _colors(rows)
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        y = getattr(r, ykey)
        if not (math.isfinite(r.l0) and math.isfinite(y)):
            continue
        ax.scatter(
            r.l0, y, s=60, color=colors[r.expansion],
            marker=ACTIVATION_MARKERS.get(r.activation, DEFAULT_MARKER),
            edgecolors="black", linewidths=0.5, zorder=3,
        )
        ax.annotate(
            r.name, (r.l0, y), textcoords="offset points", xytext=(4, 4),
            fontsize=6,
        )
    ax.set_xlabel("L0 (active features per token, held out)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    width_handles = [
        Line2D([], [], marker="o", linestyle="", color=c,
               label=f"{w:g}x ({int(round(w * 768))} features)")
        for w, c in colors.items()
    ]
    acts = sorted({r.activation for r in rows})
    act_handles = [
        Line2D([], [], marker=ACTIVATION_MARKERS.get(a, DEFAULT_MARKER),
               linestyle="", color="gray", label=a)
        for a in acts
    ]
    if width_handles:
        first = ax.legend(handles=width_handles, title="expansion",
                          loc="lower right", fontsize=8)
        ax.add_artist(first)
    if act_handles:
        ax.legend(handles=act_handles, title="activation", loc="lower center",
                  fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# Tables


def _fmt(value, spec: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return format(value, spec)


def row_cells(r: RunRow) -> list[str]:
    return [
        r.name, r.activation, r.sparsity_param, str(r.n_features),
        f"{r.n_tokens:,}", _fmt(r.l0, ".1f"), _fmt(r.variance_explained, ".3f"),
        _fmt(r.loss_recovered, ".3f"), _fmt(r.ce_clean, ".3f"),
        _fmt(r.ce_recon, ".3f"), _fmt(r.ce_zero, ".3f"),
        _fmt(r.dead_frac_eval, ".3f"), _fmt(r.tok_s, ",.0f"),
        _fmt(r.wall_min, ".1f"),
    ]


def markdown_table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return "\n".join(lines) + "\n"


def best_per_width(
    rows: list[RunRow], max_l0: float
) -> dict[float, tuple[RunRow | None, RunRow | None]]:
    """{expansion: (best 1-FVU run, best loss_recovered run)} among runs with L0 <=
    max_l0 (None when no run of that width qualifies)."""
    out: dict[float, tuple[RunRow | None, RunRow | None]] = {}
    for w in sorted({r.expansion for r in rows}):
        eligible = [
            r for r in rows
            if r.expansion == w and math.isfinite(r.l0) and r.l0 <= max_l0
        ]
        by_ve = [r for r in eligible if math.isfinite(r.variance_explained)]
        by_lr = [r for r in eligible if math.isfinite(r.loss_recovered)]
        out[w] = (
            max(by_ve, key=lambda r: r.variance_explained) if by_ve else None,
            max(by_lr, key=lambda r: r.loss_recovered) if by_lr else None,
        )
    return out


def write_tables(
    rows: list[RunRow],
    unevaluated: list[str],
    out_path: str | Path,
    max_l0: float,
) -> Path:
    parts = ["# Sweep results\n", "\n## All runs\n\n"]
    parts.append(markdown_table(list(TABLE_COLUMNS), [row_cells(r) for r in rows]))
    parts.append(f"\n## Best per width (L0 <= {max_l0:g})\n\n")
    header = ["expansion", "features", "best 1-FVU run", "1-FVU", "L0",
              "best loss_recovered run", "loss_recovered", "L0"]
    body = []
    for w, (ve, lr) in best_per_width(rows, max_l0).items():
        body.append([
            f"{w:g}x", str(int(round(w * 768))),
            ve.name if ve else "-", _fmt(ve.variance_explained if ve else None, ".3f"),
            _fmt(ve.l0 if ve else None, ".1f"),
            lr.name if lr else "-", _fmt(lr.loss_recovered if lr else None, ".3f"),
            _fmt(lr.l0 if lr else None, ".1f"),
        ])
    parts.append(markdown_table(header, body))
    if unevaluated:
        parts.append("\nNot evaluated (metrics null, no held-out shard): "
                     + ", ".join(unevaluated) + "\n")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path


# CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python plot.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", default="results",
                        help="directory holding <run>/metrics.json files")
    parser.add_argument("--out", default="figures",
                        help="output directory for frontier.png, ce.png, tables.md")
    parser.add_argument("--max-l0", type=float, default=40.0,
                        help="L0 cap for the best-per-width table (default 40)")
    args = parser.parse_args(argv)

    rows, unevaluated = load_runs(args.results)
    if not rows and not unevaluated:
        raise SystemExit(f"no <run>/metrics.json under {args.results}")
    out = Path(args.out)
    written = [
        frontier_figure(rows, "variance_explained", "1 - FVU (raw space, held out)",
                        "Reconstruction frontier", out / "frontier.png"),
        frontier_figure(rows, "loss_recovered", "loss recovered",
                        "CE frontier", out / "ce.png"),
        write_tables(rows, unevaluated, out / "tables.md", args.max_l0),
    ]
    print(f"{len(rows)} evaluated runs" + (
        f", {len(unevaluated)} unevaluated" if unevaluated else "") + ":")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
