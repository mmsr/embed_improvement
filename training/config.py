from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunConfig:
    tag: str
    model_name: str = "answerdotai/ModernBERT-base"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: ["Wqkv", "out_proj", "Wi", "Wo"]
    )
    init_from_adapter: str | None = None

    # Loss (see training.losses.LOSS_REGISTRY for available names)
    loss_fn: str = "dynamic_margin_softplus"
    base_margin: float = 0.2
    loss_alpha: float = 0.5
    loss_beta: float = 0.4

    # Data
    train_file: str = ""
    val_file: str = ""
    max_length: int = 128

    # Optimization (separate LR per param group, matching the three-way split
    # in embed_trainer_3.ipynb: LoRA weights, numeric head, metric projection)
    epochs: int = 3
    batch_size: int = 32
    lora_lr: float = 5e-5
    head_lr: float = 1e-3
    proj_lr: float = 5e-4
    weight_decay: float = 0.01

    # Checkpointing / logging
    save_checkpoint_steps: int = 500
    resume: bool = False
    resume_run_id: str | None = None
    wandb_project: str = "modernbert-numeracy-lora"
    notes: str = ""
