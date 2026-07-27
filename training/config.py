from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunConfig:
    tag: str
    model_name: str = "answerdotai/ModernBERT-base"

    # LoRA — defaults follow the empirical literature (see EXPERIMENTS.md):
    # r=16 with alpha=2r (Raschka's best-practice ratio; rank matters little at
    # this data scale), all linear layers targeted (attention-only LoRA
    # significantly underperforms — MLP layers are where capacity lives, per
    # the Thinking Machines LoRA study), dropout 0.1 for a ~76k-example dataset.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # ModernBERT names: attention = Wqkv + Wo, FFN = Wi + Wo. PEFT suffix-matches,
    # so "Wo" covers both attention out-proj and FFN down-proj. (The old notebooks
    # listed "out_proj" too, but no ModernBERT module has that name — it matched
    # nothing and was silently ignored; this default adapts the same modules.)
    target_modules: list[str] = field(
        default_factory=lambda: ["Wqkv", "Wo", "Wi"]
    )
    # rsLoRA (alpha/sqrt(r) scaling): negligible at r=16, the better default if
    # you ablate r >= 64 where standard alpha/r scaling collapses gradients.
    use_rslora: bool = False
    # Fully unfreeze (not LoRA-adapt) these base-model submodules by name suffix,
    # e.g. ["attn_norm", "mlp_norm", "emb_norm"] to also fine-tune ModernBERT's
    # LayerNorms (tried in embed_trainer_infosec_4132026.ipynb, alongside
    # target_modules including "tok_embeddings" -- not validated as better, see
    # EXPERIMENTS.md). None = LoRA-only, matching embed_trainer_3.ipynb.
    modules_to_save: list[str] | None = None
    init_from_adapter: str | None = None

    # Loss (see training.losses.LOSS_REGISTRY for available names, e.g.
    # "dynamic_margin_softplus_infosec" for the rank-margin-0.3 variant,
    # "cosent_log_ratio" / "cosent_plus_head" for the graded ranking losses,
    # "dynamic_margin_softplus_capped" for the margin-capped variant)
    loss_fn: str = "dynamic_margin_softplus"
    base_margin: float = 0.2
    loss_alpha: float = 0.5
    loss_beta: float = 0.4
    loss_tau: float = 0.05   # temperature for the cosent_* losses

    # Metric-space choice: False (default) applies the metric losses directly to
    # the normalized pooled encoder embedding — the space that is saved and
    # evaluated. True reproduces the old notebooks' behavior of optimizing a
    # separate metric_proj head (which was discarded at save time — the main
    # reason loss changes never showed up in eval; see EXPERIMENTS.md).
    use_metric_proj: bool = False

    # Data
    train_file: str = ""
    val_file: str = ""
    max_length: int = 128

    # Optimization (separate LR per param group, matching the three-way split
    # in embed_trainer_3.ipynb: LoRA weights, numeric head, metric projection)
    epochs: int = 3
    # batch_size 32: LoRA tolerates large batches worse than full fine-tuning
    # (Thinking Machines), and 32 triplets already give 64 in-batch pairs for
    # the cosent losses. 64 is a reasonable A100 option if VRAM allows.
    batch_size: int = 32
    # LoRA optimal LR is consistently ~10x the full-fine-tuning LR (Thinking
    # Machines; the old notebooks used 2e-5 full-FT-style, hence 2e-4 here).
    # Safe together with warmup + cosine schedule + grad clipping below.
    lora_lr: float = 2e-4
    head_lr: float = 1e-3
    proj_lr: float = 5e-4
    weight_decay: float = 0.01
    seed: int = 42
    lr_schedule: str = "cosine"   # "cosine" (warmup + cosine decay) or "none" (constant, old behavior)
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0        # max grad norm; 0 disables clipping

    # Checkpointing / logging
    save_checkpoint_steps: int = 500
    resume: bool = False
    resume_run_id: str | None = None
    wandb_project: str = "modernbert-numeracy-lora"
    ordering_eval_each_epoch: bool = True  # run the ordering probe suites after each epoch
    notes: str = ""
