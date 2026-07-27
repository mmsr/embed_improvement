"""Numeracy contrastive losses. Register new variants in LOSS_REGISTRY and select
one by name via RunConfig.loss_fn -- no other code needs to change to try a new loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dynamic_margin_sigmoid_loss(
    anchor_emb,
    pos_emb,
    neg_emb,
    anchor_score,
    pos_score,
    neg_score,
    anchor_value,
    pos_value,
    neg_value,
    base_margin=0.2,
    alpha=0.5,
    beta=0.6,
    eps=1e-8,
):
    """Sigmoid-scaled dynamic margin triplet loss, tanh-saturating log-distance
    alignment, sign-based hard-margin rank loss."""
    anchor = F.normalize(anchor_emb, dim=-1)
    pos = F.normalize(pos_emb, dim=-1)
    neg = F.normalize(neg_emb, dim=-1)

    pos_cos_dist = 1.0 - F.cosine_similarity(anchor, pos, dim=-1)
    neg_cos_dist = 1.0 - F.cosine_similarity(anchor, neg, dim=-1)

    log_a = torch.log1p(anchor_value)
    log_p = torch.log1p(pos_value)
    log_n = torch.log1p(neg_value)

    log_pos_dist = torch.abs(log_a - log_p)
    log_neg_dist = torch.abs(log_a - log_n)

    log_diff = log_neg_dist - log_pos_dist
    dyn_margin = base_margin * (1.0 + torch.sigmoid(log_diff))
    triplet_loss = F.relu(pos_cos_dist - neg_cos_dist + dyn_margin).mean()

    norm_pos_cos = pos_cos_dist / 2.0
    norm_neg_cos = neg_cos_dist / 2.0
    target_pos = torch.tanh(log_pos_dist)
    target_neg = torch.tanh(log_neg_dist)
    log_distance_loss = F.mse_loss(norm_pos_cos, target_pos) + F.mse_loss(
        norm_neg_cos, target_neg
    )

    head_loss = (
        F.mse_loss(anchor_score, log_a)
        + F.mse_loss(pos_score, log_p)
        + F.mse_loss(neg_score, log_n)
    ) / 3.0

    margin_rank = 0.3
    ap_sign = torch.sign(log_a - log_p)
    an_sign = torch.sign(log_a - log_n)
    rank_loss = (
        F.relu(margin_rank - ap_sign * (anchor_score - pos_score)).mean()
        + F.relu(margin_rank - an_sign * (anchor_score - neg_score)).mean()
    ) / 2.0

    metric_loss = triplet_loss + log_distance_loss
    supervision_loss = beta * head_loss + (1.0 - beta) * rank_loss
    total_loss = alpha * metric_loss + (1.0 - alpha) * supervision_loss

    components = {
        "triplet": triplet_loss.item(),
        "log_dist": log_distance_loss.item(),
        "head": head_loss.item(),
        "rank": rank_loss.item(),
        "metric": metric_loss.item(),
        "supervision": supervision_loss.item(),
        "total": total_loss.item(),
    }
    return total_loss, components


def dynamic_margin_softplus_loss(
    anchor_emb,
    pos_emb,
    neg_emb,
    anchor_score,
    pos_score,
    neg_score,
    anchor_value,
    pos_value,
    neg_value,
    base_margin=0.2,
    alpha=0.5,
    beta=0.6,
    eps=1e-8,
):
    """Clamp-scaled dynamic margin, non-saturating log-distance alignment
    (anchor-pos, anchor-neg, and pos-neg pairs), softplus smooth rank loss."""
    anchor = F.normalize(anchor_emb, dim=-1)
    pos = F.normalize(pos_emb, dim=-1)
    neg = F.normalize(neg_emb, dim=-1)

    pos_cos = 1.0 - F.cosine_similarity(anchor, pos, dim=-1)
    neg_cos = 1.0 - F.cosine_similarity(anchor, neg, dim=-1)
    pn_cos = 1.0 - F.cosine_similarity(pos, neg, dim=-1)

    log_a = torch.log1p(anchor_value + eps)
    log_p = torch.log1p(pos_value + eps)
    log_n = torch.log1p(neg_value + eps)

    log_pos = torch.abs(log_a - log_p)
    log_neg = torch.abs(log_a - log_n)
    log_pn = torch.abs(log_p - log_n)

    log_diff = log_neg - log_pos
    dyn_margin = base_margin * (1.0 + log_diff.clamp(min=0))
    triplet_loss = F.relu(pos_cos - neg_cos + dyn_margin).mean()

    def scale(x):
        return x / (1.0 + x)

    log_distance_loss = (
        F.mse_loss(pos_cos / 2.0, scale(log_pos))
        + F.mse_loss(neg_cos / 2.0, scale(log_neg))
        + F.mse_loss(pn_cos / 2.0, scale(log_pn))
    ) / 3.0

    head_loss = (
        F.mse_loss(anchor_score, log_a)
        + F.mse_loss(pos_score, log_p)
        + F.mse_loss(neg_score, log_n)
    ) / 3.0

    margin_rank = 0.1
    ap = anchor_score - pos_score
    an = anchor_score - neg_score
    ap_sign = torch.sign(log_a - log_p)
    an_sign = torch.sign(log_a - log_n)
    rank_loss = (
        F.softplus(margin_rank - ap_sign * ap).mean()
        + F.softplus(margin_rank - an_sign * an).mean()
    ) / 2.0

    metric_loss = triplet_loss + log_distance_loss
    supervision_loss = beta * head_loss + (1.0 - beta) * rank_loss
    total_loss = alpha * metric_loss + (1.0 - alpha) * supervision_loss

    components = {
        "triplet": triplet_loss.item(),
        "log_dist": log_distance_loss.item(),
        "head": head_loss.item(),
        "rank": rank_loss.item(),
        "metric": metric_loss.item(),
        "supervision": supervision_loss.item(),
        "total": total_loss.item(),
    }
    return total_loss, components


LOSS_REGISTRY = {
    "dynamic_margin_sigmoid": dynamic_margin_sigmoid_loss,
    "dynamic_margin_softplus": dynamic_margin_softplus_loss,
}


def get_loss_fn(name: str):
    try:
        return LOSS_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown loss_fn {name!r}. Available: {list(LOSS_REGISTRY)}"
        ) from None
