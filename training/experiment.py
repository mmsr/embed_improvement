"""Run naming, checkpoint saving, and results logging for LoRA fine-tuning runs."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from training.config import RunConfig

__all__ = [
    "RunConfig",
    "new_run_id",
    "run_dir",
    "save_run",
    "save_checkpoint",
    "log_result",
    "load_registry",
]


def new_run_id(tag: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    safe_tag = tag.strip().replace(" ", "-")
    return f"{stamp}_{safe_tag}"


def run_dir(work_dir: str, run_id: str) -> Path:
    d = Path(work_dir) / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(work_dir: str, run_id: str, model, tokenizer, config: RunConfig) -> Path:
    d = run_dir(work_dir, run_id)
    adapter_dir = d / "adapter"
    adapter_dir.mkdir(exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (d / "config.json").write_text(json.dumps(asdict(config), indent=2))
    return d


def save_checkpoint(work_dir: str, run_id: str, epoch: int, model, tokenizer) -> Path:
    """Per-epoch adapter snapshot at runs/<run_id>/checkpoints/epoch_<n>/, separate from
    the final adapter (save_run) and the raw accelerator resume state."""
    d = run_dir(work_dir, run_id) / "checkpoints" / f"epoch_{epoch}"
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d)
    tokenizer.save_pretrained(d)
    return d


def log_result(
    work_dir: str,
    run_id: str,
    config: RunConfig,
    metrics: dict[str, Any],
    eval_file: str = "",
) -> None:
    registry_path = Path(work_dir) / "runs" / "registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eval_file": eval_file,
        **asdict(config),
        **metrics,
    }
    with registry_path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def load_registry(work_dir: str):
    import pandas as pd

    registry_path = Path(work_dir) / "runs" / "registry.jsonl"
    if not registry_path.exists():
        return pd.DataFrame()
    return pd.read_json(registry_path, lines=True)
