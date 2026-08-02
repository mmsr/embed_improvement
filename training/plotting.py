"""Publication-ready plots from durable run histories and the evaluation registry."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def _run_label(run_id: str, rows: list[dict]) -> str:
    if rows:
        return f"{rows[0].get('loss_fn', 'unknown')} (seed {rows[0].get('seed', '?')})"
    return run_id


def _save_figure(fig, stem: Path) -> list[str]:
    paths = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    return paths


def discover_run_ids(work_dir: str | Path) -> list[str]:
    """Find runs that contain the new durable epoch history."""
    runs_dir = Path(work_dir) / "runs"
    if not runs_dir.exists():
        return []
    return sorted(
        path.parent.parent.name
        for path in runs_dir.glob("*/training/epoch_metrics.jsonl")
    )


def plot_run_training(
    work_dir: str | Path,
    run_id: str,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Plot raw train/validation loss, LR, and ordering metrics for one run."""
    import matplotlib.pyplot as plt

    run_root = Path(work_dir) / "runs" / run_id
    epochs = _read_jsonl(run_root / "training" / "epoch_metrics.jsonl")
    steps = _read_jsonl(run_root / "training" / "step_metrics.jsonl")
    if not epochs:
        raise FileNotFoundError(f"No epoch history found for run {run_id}")

    output = Path(output_dir) if output_dir else run_root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    label = _run_label(run_id, epochs)

    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes_grid.flat

    epoch_numbers = [row["epoch"] for row in epochs]
    axes[0].plot(epoch_numbers, [row.get("train_loss") for row in epochs], "o-", label="Train")
    axes[0].plot(epoch_numbers, [row.get("val_loss") for row in epochs], "s-", label="Validation")
    axes[0].set(title="Epoch loss", xlabel="Epoch", ylabel="Raw objective loss")
    axes[0].legend()

    if steps:
        axes[1].plot(
            [row["global_step"] for row in steps],
            [row["train_loss"] for row in steps],
            linewidth=1,
            alpha=0.75,
            label="Training loss",
        )
        lr_axis = axes[1].twinx()
        lr_axis.plot(
            [row["global_step"] for row in steps],
            [row["learning_rate"] for row in steps],
            color="tab:orange",
            linewidth=1,
            alpha=0.7,
            label="Learning rate",
        )
        lr_axis.set_ylabel("Learning rate", color="tab:orange")
    axes[1].set(title="Optimization trace", xlabel="Global step", ylabel="Raw objective loss")

    axes[2].plot(
        epoch_numbers,
        [row.get("val_triplet_accuracy") for row in epochs],
        "o-",
        label="Triplet accuracy",
    )
    axes[2].set(
        title="Common validation metric",
        xlabel="Epoch",
        ylabel="Triplet accuracy",
        ylim=(-0.05, 1.05),
    )
    gap_axis = axes[2].twinx()
    gap_axis.plot(
        epoch_numbers,
        [row.get("val_cosine_gap") for row in epochs],
        "s--",
        color="tab:orange",
        label="Cosine gap",
    )
    gap_axis.set_ylabel("Cosine gap", color="tab:orange")

    ordering_keys = (
        ("ordering_mean", "Spearman"),
        ("pairwise_ordering_mean", "Pairwise ordering"),
        ("inversion_rate_mean", "Inversion rate"),
    )
    plotted = False
    for key, name in ordering_keys:
        values = [row.get(key) for row in epochs]
        if any(value is not None for value in values):
            axes[3].plot(epoch_numbers, values, "o-", label=name)
            plotted = True
    axes[3].set(title="Ordering probes", xlabel="Epoch", ylabel="Score", ylim=(-0.05, 1.05))
    if plotted:
        axes[3].legend()

    fig.suptitle(f"Training history — {label}\n{run_id}", fontsize=11)
    fig.tight_layout()
    paths = _save_figure(fig, output / f"training_{_safe_name(run_id)}")
    plt.close(fig)
    return paths


def plot_training_comparison(
    work_dir: str | Path,
    run_ids: Iterable[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Compare loss improvement and ordering across runs without mixing raw loss scales.

    Different objectives have different numerical loss scales. The combined loss panel
    therefore divides each run's validation loss by its own first-epoch value. Raw loss
    remains available in each run's individual figure and history file.
    """
    import matplotlib.pyplot as plt

    selected = list(run_ids) if run_ids is not None else discover_run_ids(work_dir)
    histories = {
        run_id: _read_jsonl(
            Path(work_dir) / "runs" / run_id / "training" / "epoch_metrics.jsonl"
        )
        for run_id in selected
    }
    histories = {run_id: rows for run_id, rows in histories.items() if rows}
    if not histories:
        raise FileNotFoundError("No durable epoch histories were found")

    output = Path(output_dir) if output_dir else Path(work_dir) / "runs" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for run_id, rows in histories.items():
        label = _run_label(run_id, rows)
        x = [row["epoch"] for row in rows]
        val = [row.get("val_loss") for row in rows]
        baseline = val[0] if val and val[0] not in (None, 0) else 1.0
        axes[0].plot(x, [value / baseline for value in val], "o-", label=label)

        ordering = [row.get("ordering_mean") for row in rows]
        if any(value is not None for value in ordering):
            axes[1].plot(x, ordering, "o-", label=label)

        triplet_accuracy = [row.get("val_triplet_accuracy") for row in rows]
        if any(value is not None for value in triplet_accuracy):
            axes[2].plot(x, triplet_accuracy, "o-", label=label)

    axes[0].set(title="Normalized validation loss", xlabel="Epoch", ylabel="Fraction of epoch-1 loss")
    axes[1].set(title="Mean ordering Spearman", xlabel="Epoch", ylabel="Spearman", ylim=(-0.05, 1.05))
    axes[2].set(title="Validation triplet accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(-0.05, 1.05))
    for axis in axes:
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
    fig.suptitle("Loss-family training comparison", fontsize=12)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    paths = _save_figure(fig, output / "training_comparison")
    plt.close(fig)
    return paths


def plot_evaluation_comparison(
    work_dir: str | Path,
    run_ids: Iterable[str] | None = None,
    eval_split: str = "val",
    output_dir: str | Path | None = None,
) -> list[str]:
    """Create paper-ready final validation metric bars from registry.jsonl."""
    import matplotlib.pyplot as plt

    registry_path = Path(work_dir) / "runs" / "registry.jsonl"
    rows = _read_jsonl(registry_path)
    selected = set(run_ids) if run_ids is not None else None
    rows = [
        row for row in rows
        if (selected is None or row.get("run_id") in selected)
        and str(row.get("eval_file", "")).startswith(f"{eval_split}:")
    ]
    if not rows:
        raise FileNotFoundError(f"No {eval_split!r} evaluation rows found in {registry_path}")

    # Preserve only the newest registry record if a run appears more than once.
    by_run = {row["run_id"]: row for row in rows}
    rows = list(by_run.values())
    triplet_suffix = "/triplet_accuracy"
    prefixes = sorted({
        key[:-len(triplet_suffix)]
        for row in rows
        for key in row
        if key.endswith(triplet_suffix)
    })
    if not prefixes:
        raise ValueError("Registry rows do not contain a triplet_accuracy metric")
    if len(prefixes) > 1:
        raise ValueError(
            f"Multiple evaluation metric prefixes found: {prefixes}. "
            "Compare runs evaluated with the same dataset label."
        )
    prefix = prefixes[0]
    metrics = [
        (f"{prefix}/triplet_accuracy", "Triplet accuracy"),
        (f"{prefix}/cosine_gap", "Cosine gap"),
        (f"{prefix}/pairwise_mrr", "Pairwise MRR"),
        ("ordering_mean_random", "Random-suite Spearman"),
        ("pairwise_ordering_mean_random", "Random pairwise ordering"),
        ("inversion_rate_mean_random", "Random inversion rate"),
    ]
    metrics = [(key, title) for key, title in metrics if any(key in row for row in rows)]
    if not metrics:
        raise ValueError("Registry rows do not contain the expected evaluation metrics")

    labels = [f"{row.get('loss_fn', 'unknown')}\nseed {row.get('seed', '?')}" for row in rows]
    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)
    for axis, (key, title) in zip(axes.flat, metrics):
        values = [row.get(key, float("nan")) for row in rows]
        axis.bar(range(len(rows)), values)
        axis.set_title(title)
        axis.set_xticks(range(len(rows)), labels, rotation=30, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    for axis in axes.flat[len(metrics):]:
        axis.set_visible(False)

    output = Path(output_dir) if output_dir else Path(work_dir) / "runs" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / f"evaluation_comparison_{eval_split}.csv"
    fieldnames = ["run_id", "loss_fn", "seed"] + [key for key, _ in metrics]
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    fig.suptitle(f"Final {eval_split} evaluation comparison", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    paths = _save_figure(fig, output / f"evaluation_comparison_{eval_split}")
    plt.close(fig)
    return paths + [str(table_path)]


def plot_checkpoint_comparison(
    work_dir: str | Path,
    run_ids: Iterable[str] | None = None,
    eval_split: str = "val",
    output_dir: str | Path | None = None,
) -> list[str]:
    """Plot common validation metrics for every saved epoch across runs/seeds."""
    import matplotlib.pyplot as plt

    registry_path = Path(work_dir) / "runs" / "checkpoint_registry.jsonl"
    rows = _read_jsonl(registry_path)
    selected = set(run_ids) if run_ids is not None else None
    rows = [
        row for row in rows
        if (selected is None or row.get("run_id") in selected)
        and row.get("eval_split") == eval_split
    ]
    if not rows:
        raise FileNotFoundError(f"No {eval_split!r} checkpoint rows found in {registry_path}")

    suffix = "/triplet_accuracy"
    prefixes = sorted({
        key[:-len(suffix)] for row in rows for key in row if key.endswith(suffix)
    })
    if len(prefixes) != 1:
        raise ValueError(f"Expected one checkpoint metric prefix, found {prefixes}")
    prefix = prefixes[0]
    panels = [
        (f"{prefix}/triplet_accuracy", "Triplet accuracy", (0.0, 1.02)),
        (f"{prefix}/recall@1", "Recall@1", (0.0, 1.02)),
        ("ordering_mean_random", "Random-suite Spearman", (-0.05, 1.05)),
        ("pairwise_ordering_mean_random", "Random pairwise ordering", (0.0, 1.02)),
    ]

    by_run: dict[str, list[dict]] = {}
    for row in rows:
        by_run.setdefault(row["run_id"], []).append(row)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for key, title, limits in panels:
        axis = axes.flat[panels.index((key, title, limits))]
        for run_id, run_rows in sorted(by_run.items()):
            run_rows.sort(key=lambda row: row["epoch"])
            first = run_rows[0]
            label = f"{first.get('loss_fn', 'unknown')} · seed {first.get('seed', '?')}"
            axis.plot(
                [row["epoch"] for row in run_rows],
                [row.get(key) for row in run_rows],
                marker="o",
                label=label,
            )
        axis.set(title=title, xlabel="Epoch", ylabel=title, ylim=limits)
        axis.grid(alpha=0.2)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), fontsize=8)
    fig.suptitle(f"Per-epoch {eval_split} checkpoint comparison", fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))

    output = Path(output_dir) if output_dir else Path(work_dir) / "runs" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    paths = _save_figure(fig, output / f"checkpoint_comparison_{eval_split}")
    plt.close(fig)

    table_path = output / f"checkpoint_comparison_{eval_split}.csv"
    fieldnames = [
        "run_id", "loss_fn", "seed", "epoch", "checkpoint_path",
        *[key for key, _, _ in panels],
        "ordering_mean", "inversion_rate_mean_random", "sim_spread_mean_random",
    ]
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["run_id"], row["epoch"])))
    return paths + [str(table_path)]


def generate_experiment_graphs(
    work_dir: str | Path,
    run_ids: Iterable[str] | None = None,
    eval_split: str = "val",
) -> dict[str, list[str]]:
    """Generate individual, combined-training, and final-evaluation artifacts."""
    selected = list(run_ids) if run_ids is not None else discover_run_ids(work_dir)
    artifacts: dict[str, list[str]] = {}
    for run_id in selected:
        artifacts[f"run/{run_id}"] = plot_run_training(work_dir, run_id)
    artifacts["training_comparison"] = plot_training_comparison(work_dir, selected)
    artifacts["evaluation_comparison"] = plot_evaluation_comparison(
        work_dir, selected, eval_split=eval_split
    )
    checkpoint_registry = Path(work_dir) / "runs" / "checkpoint_registry.jsonl"
    if checkpoint_registry.exists():
        artifacts["checkpoint_comparison"] = plot_checkpoint_comparison(
            work_dir, selected, eval_split=eval_split
        )
    return artifacts
