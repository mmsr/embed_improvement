"""ModernBERT + LoRA contrastive model wrapper."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoTokenizer

from training.config import RunConfig

AUX_HEADS_FILE = "aux_heads.pt"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)


class ContrastiveModel(nn.Module):
    """LoRA-adapted encoder + auxiliary numeric head.

    use_metric_proj=False (default): the metric losses see the normalized pooled
    encoder embedding — the same space that gets saved and evaluated.
    use_metric_proj=True: old notebook behavior — losses see a separate linear
    projection that is NOT part of the saved encoder (kept for A/B comparison).
    """

    def __init__(self, encoder, use_metric_proj: bool = False):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.numeric_head = nn.Linear(hidden_size, 1)
        self.metric_proj = nn.Linear(hidden_size, hidden_size) if use_metric_proj else None

    def encode(self, batch_part):
        out = self.encoder(
            input_ids=batch_part["input_ids"],
            attention_mask=batch_part["attention_mask"],
        )
        return mean_pooling(out, batch_part["attention_mask"])

    def forward(self, batch):
        a_emb = self.encode(batch["anchor"])
        p_emb = self.encode(batch["positive"])
        n_emb = self.encode(batch["negative"])

        a_score = self.numeric_head(a_emb).squeeze(-1)
        p_score = self.numeric_head(p_emb).squeeze(-1)
        n_score = self.numeric_head(n_emb).squeeze(-1)

        if self.metric_proj is not None:
            a_emb = self.metric_proj(a_emb)
            p_emb = self.metric_proj(p_emb)
            n_emb = self.metric_proj(n_emb)

        return {
            "a_emb": F.normalize(a_emb, dim=-1),
            "p_emb": F.normalize(p_emb, dim=-1),
            "n_emb": F.normalize(n_emb, dim=-1),
            "a_score": a_score,
            "p_score": p_score,
            "n_score": n_score,
        }


def save_aux_heads(model: ContrastiveModel, run_dir) -> None:
    """Persist the auxiliary heads next to the adapter so a continued run
    doesn't silently restart them from random init."""
    state = {"numeric_head": model.numeric_head.state_dict()}
    if model.metric_proj is not None:
        state["metric_proj"] = model.metric_proj.state_dict()
    torch.save(state, Path(run_dir) / AUX_HEADS_FILE)


def _load_aux_heads(model: ContrastiveModel, adapter_path: str) -> bool:
    for candidate in (Path(adapter_path) / AUX_HEADS_FILE,
                      Path(adapter_path).parent / AUX_HEADS_FILE):
        if candidate.exists():
            state = torch.load(candidate, map_location="cpu")
            model.numeric_head.load_state_dict(state["numeric_head"])
            if model.metric_proj is not None and "metric_proj" in state:
                model.metric_proj.load_state_dict(state["metric_proj"])
            return True
    return False


def build_model(config: RunConfig):
    """Build tokenizer + ContrastiveModel. Fresh LoRA adapter from config.lora_*,
    or continues training an existing adapter if config.init_from_adapter is set
    (restoring saved auxiliary heads when found alongside the adapter)."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    base_model = AutoModel.from_pretrained(config.model_name)

    if config.init_from_adapter:
        base_model = PeftModel.from_pretrained(
            base_model, config.init_from_adapter, is_trainable=True
        )
    else:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            modules_to_save=config.modules_to_save,
            use_rslora=config.use_rslora,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        base_model = get_peft_model(base_model, lora_config)

    model = ContrastiveModel(base_model, use_metric_proj=config.use_metric_proj)
    if config.init_from_adapter:
        _load_aux_heads(model, config.init_from_adapter)
    return model, tokenizer
