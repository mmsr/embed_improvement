"""Training loop orchestration for a single RunConfig. Entry point: train_one_run()."""
from __future__ import annotations

import inspect
import json
import os
import random

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from dataclasses import asdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.config import RunConfig
from training.data import TripletDataset, make_triplet_collator
from training.experiment import new_run_id, run_dir, save_checkpoint, save_run
from training.losses import get_loss_fn
from training.model import build_model, save_aux_heads


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model, config: RunConfig):
    """Param groups matching embed_trainer_3.ipynb: LoRA weights and numeric
    head each get their own learning rate (plus metric_proj when enabled)."""
    lora_params = [
        p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad
    ]
    groups = [
        {"params": lora_params, "lr": config.lora_lr, "weight_decay": config.weight_decay},
        {"params": list(model.numeric_head.parameters()), "lr": config.head_lr, "weight_decay": 0.0},
    ]
    if model.metric_proj is not None:
        groups.append(
            {"params": list(model.metric_proj.parameters()), "lr": config.proj_lr,
             "weight_decay": config.weight_decay}
        )
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(optimizer, config: RunConfig, total_steps: int):
    if config.lr_schedule == "none":
        return None
    from transformers import get_cosine_schedule_with_warmup

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )


def _loss_kwargs(loss_fn, config: RunConfig) -> dict:
    """Pass only the hyperparameters this loss function's signature accepts, so
    registry entries can have different knobs (base_margin/alpha/beta vs tau)."""
    params = inspect.signature(loss_fn).parameters
    candidates = {
        "base_margin": config.base_margin,
        "alpha": config.loss_alpha,
        "beta": config.loss_beta,
        "tau": config.loss_tau,
    }
    return {k: v for k, v in candidates.items() if k in params}


def _run_batch(model, batch, loss_fn, loss_kwargs: dict):
    outputs = model(batch)
    return loss_fn(
        anchor_emb=outputs["a_emb"],
        pos_emb=outputs["p_emb"],
        neg_emb=outputs["n_emb"],
        anchor_score=outputs["a_score"],
        pos_score=outputs["p_score"],
        neg_score=outputs["n_score"],
        anchor_value=batch["anchor_number"],
        pos_value=batch["positive_number"],
        neg_value=batch["negative_number"],
        **loss_kwargs,
    )


def _evaluate_loss(model, loader, loss_fn, loss_kwargs: dict) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            loss, components = _run_batch(model, batch, loss_fn, loss_kwargs)
            totals["loss"] = totals.get("loss", 0.0) + loss.item()
            for k, v in components.items():
                totals[k] = totals.get(k, 0.0) + v
    return {k: v / len(loader) for k, v in totals.items()}


def _epoch_ordering(accelerator, model, tokenizer) -> dict:
    """Run the fixed ordering probe suites on the current encoder (cheap: ~50
    short sentences). Never lets a probe failure kill training."""
    if not accelerator.is_main_process:
        return {}
    try:
        from training.evaluate import ordering_scores

        unwrapped = accelerator.unwrap_model(model)
        unwrapped.eval()
        return ordering_scores(unwrapped.encoder, tokenizer)
    except Exception as e:  # noqa: BLE001 — diagnostics only, training continues
        accelerator.print(f"ordering probe failed: {e}")
        return {}


def train_one_run(config: RunConfig, work_dir: str) -> dict:
    set_seed(config.seed)

    run_id = (
        config.resume_run_id
        if (config.resume and config.resume_run_id)
        else new_run_id(config.tag)
    )
    accel_state_dir = os.path.join(work_dir, "runs", run_id, "accelerator_state")
    os.makedirs(accel_state_dir, exist_ok=True)

    accelerator = Accelerator()
    model, tokenizer = build_model(config)
    loss_fn = get_loss_fn(config.loss_fn)
    loss_kwargs = _loss_kwargs(loss_fn, config)

    collate_fn = make_triplet_collator(tokenizer)
    train_dataset = TripletDataset(config.train_file, tokenizer, config.max_length)
    val_dataset = TripletDataset(config.val_file, tokenizer, config.max_length)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    optimizer = build_optimizer(model, config)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    scheduler = build_scheduler(optimizer, config, len(train_loader) * config.epochs)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    start_epoch, global_step = 0, 0
    metadata_path = os.path.join(accel_state_dir, "training_metadata.json")
    if config.resume and os.path.exists(os.path.join(accel_state_dir, "model.safetensors")):
        accelerator.load_state(accel_state_dir)
        if os.path.exists(metadata_path):
            metadata = json.load(open(metadata_path))
            start_epoch = metadata.get("epoch", 0)
            global_step = metadata.get("global_step", 0)
        accelerator.print(f"Resumed run {run_id}: epoch={start_epoch} step={global_step}")
    elif config.resume:
        accelerator.print(f"No checkpoint found for {run_id} at {accel_state_dir}, starting fresh")

    steps_per_epoch = len(train_loader)
    resume_skip_steps = max(0, global_step - start_epoch * steps_per_epoch)

    wandb.init(project=config.wandb_project, name=run_id, config=asdict(config))

    current_global_step = global_step
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_totals: dict[str, float] = {}
        n_batches = 0
        progress_bar = tqdm(
            train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch + 1}"
        )

        for step, batch in enumerate(progress_bar):
            if epoch == start_epoch and step < resume_skip_steps:
                continue
            current_global_step = global_step + (epoch - start_epoch) * steps_per_epoch + step
            n_batches += 1

            loss, components = _run_batch(model, batch, loss_fn, loss_kwargs)
            accelerator.backward(loss)
            if config.grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            epoch_totals["loss"] = epoch_totals.get("loss", 0.0) + loss.item()
            for k, v in components.items():
                epoch_totals[k] = epoch_totals.get(k, 0.0) + v

            if (current_global_step + 1) % 100 == 0 or current_global_step == 0:
                accelerator.print(
                    f"Step {current_global_step + 1} | loss {loss.item():.4f} | {components}"
                )
                wandb.log(
                    {
                        "train_loss_step": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "global_step": current_global_step + 1,
                        "epoch": epoch,
                    }
                )

            if (current_global_step + 1) % config.save_checkpoint_steps == 0:
                accelerator.save_state(accel_state_dir)
                if accelerator.is_main_process:
                    json.dump(
                        {"epoch": epoch, "global_step": current_global_step + 1},
                        open(metadata_path, "w"),
                    )
                accelerator.wait_for_everyone()
                accelerator.print(f"Checkpoint saved at step {current_global_step + 1}")

        # n_batches can be 0 when resuming from a state saved exactly at an
        # epoch boundary (every step of this epoch already ran before the
        # interruption) — average over batches actually processed.
        train_metrics = {k: v / max(n_batches, 1) for k, v in epoch_totals.items()}
        val_metrics = _evaluate_loss(model, val_loader, loss_fn, loss_kwargs)

        epoch_log = {
            "epoch": epoch + 1,
            "val_loss": val_metrics["loss"],
        }
        if "loss" in train_metrics:
            epoch_log["train_loss"] = train_metrics["loss"]

        if config.ordering_eval_each_epoch:
            ordering = _epoch_ordering(accelerator, model, tokenizer)
            if ordering:
                epoch_log["ordering_mean"] = ordering["ordering_mean"]
                epoch_log["ordering_monotonic_decay"] = ordering.get(
                    "ordering@monotonic_decay"
                )
                epoch_log["sim_spread_mean"] = ordering["sim_spread_mean"]
                accelerator.print(f"Epoch {epoch + 1} ordering probes: {ordering}")
            model.train()

        wandb.log(epoch_log)
        accelerator.print(f"Epoch {epoch + 1} | train {train_metrics} | val {val_metrics}")

        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        epoch_dir = save_checkpoint(work_dir, run_id, epoch + 1, unwrapped.encoder, tokenizer)
        if accelerator.is_main_process:
            save_aux_heads(unwrapped, epoch_dir)
        accelerator.print(f"Epoch {epoch + 1} adapter saved to {epoch_dir}")

        if accelerator.is_main_process:
            json.dump(
                {"epoch": epoch, "global_step": current_global_step + 1},
                open(metadata_path, "w"),
            )
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    final_dir = save_run(work_dir, run_id, unwrapped.encoder, tokenizer, config)
    if accelerator.is_main_process:
        save_aux_heads(unwrapped, run_dir(work_dir, run_id))
    wandb.finish()

    accelerator.print(f"Run {run_id} complete. Final adapter saved to {final_dir}")
    return {"run_id": run_id, "run_dir": str(final_dir)}
