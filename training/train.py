"""Training loop orchestration for a single RunConfig. Entry point: train_one_run()."""
from __future__ import annotations

import json
import os

import torch
import wandb
from accelerate import Accelerator
from dataclasses import asdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.config import RunConfig
from training.data import TripletDataset, make_triplet_collator
from training.experiment import new_run_id, save_checkpoint, save_run
from training.losses import get_loss_fn
from training.model import build_model


def build_optimizer(model, config: RunConfig):
    """Three param groups matching embed_trainer_3.ipynb: LoRA weights, numeric
    head, and metric projection each get their own learning rate."""
    lora_params = [
        p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad
    ]
    head_params = list(model.numeric_head.parameters())
    proj_params = list(model.metric_proj.parameters())
    return torch.optim.AdamW(
        [
            {"params": lora_params, "lr": config.lora_lr, "weight_decay": config.weight_decay},
            {"params": head_params, "lr": config.head_lr, "weight_decay": 0.0},
            {"params": proj_params, "lr": config.proj_lr, "weight_decay": config.weight_decay},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def _run_batch(model, batch, loss_fn, config: RunConfig):
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
        base_margin=config.base_margin,
        alpha=config.loss_alpha,
        beta=config.loss_beta,
    )


def _evaluate_loss(model, loader, loss_fn, config: RunConfig) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            loss, components = _run_batch(model, batch, loss_fn, config)
            totals["loss"] = totals.get("loss", 0.0) + loss.item()
            for k, v in components.items():
                totals[k] = totals.get(k, 0.0) + v
    return {k: v / len(loader) for k, v in totals.items()}


def train_one_run(config: RunConfig, work_dir: str) -> dict:
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
        progress_bar = tqdm(
            train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch + 1}"
        )

        for step, batch in enumerate(progress_bar):
            if epoch == start_epoch and step < resume_skip_steps:
                continue
            current_global_step = global_step + (epoch - start_epoch) * steps_per_epoch + step

            loss, components = _run_batch(model, batch, loss_fn, config)
            accelerator.backward(loss)
            optimizer.step()
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

        train_metrics = {k: v / steps_per_epoch for k, v in epoch_totals.items()}
        val_metrics = _evaluate_loss(model, val_loader, loss_fn, config)

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
            }
        )
        accelerator.print(f"Epoch {epoch + 1} | train {train_metrics} | val {val_metrics}")

        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        epoch_dir = save_checkpoint(work_dir, run_id, epoch + 1, unwrapped.encoder, tokenizer)
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
    wandb.finish()

    accelerator.print(f"Run {run_id} complete. Final adapter saved to {final_dir}")
    return {"run_id": run_id, "run_dir": str(final_dir)}
