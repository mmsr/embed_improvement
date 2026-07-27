"""ModernBERT + LoRA contrastive model wrapper."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoTokenizer

from training.config import RunConfig


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)


class ContrastiveModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.numeric_head = nn.Linear(hidden_size, 1)
        self.metric_proj = nn.Linear(hidden_size, hidden_size)

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

        a_proj = F.normalize(self.metric_proj(a_emb), dim=-1)
        p_proj = F.normalize(self.metric_proj(p_emb), dim=-1)
        n_proj = F.normalize(self.metric_proj(n_emb), dim=-1)

        return {
            "a_emb": a_proj,
            "p_emb": p_proj,
            "n_emb": n_proj,
            "a_score": a_score,
            "p_score": p_score,
            "n_score": n_score,
        }


def build_model(config: RunConfig):
    """Build tokenizer + ContrastiveModel. Fresh LoRA adapter from config.lora_*,
    or continues training an existing adapter if config.init_from_adapter is set."""
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
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        base_model = get_peft_model(base_model, lora_config)

    return ContrastiveModel(base_model), tokenizer
