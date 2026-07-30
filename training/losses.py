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
    (anchor-pos, anchor-neg, and pos-neg pairs), softplus smooth rank loss.
    This is the exact loss trained in embed_trainer_3.ipynb (rank margin 0.1).
    See dynamic_margin_softplus_infosec_loss for the rank-margin-0.3 variant
    tried in embed_trainer_infosec_4132026.ipynb."""
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


def dynamic_margin_softplus_infosec_loss(
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
    """Same as dynamic_margin_softplus_loss, but with rank margin 0.3 instead of
    0.1 -- the exact variant trained in embed_trainer_infosec_4132026.ipynb, the
    latest loss tried. Not shown better in isolation: infosec's full recipe
    (this loss + tok_embeddings LoRA + norm-layer unfreezing) scored ordering
    rho=0.73 vs the untouched trainer_3-era CLD3-old's 0.93 -- see EXPERIMENTS.md."""
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

    margin_rank = 0.3
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


# Scale for normalizing head-regression targets: log1p(value) tops out around 20
# on this dataset (max value ~2.5e8), so /20 keeps targets in [0, 1] and stops
# head_loss (~60 at init unnormalized) from dominating early training.
HEAD_TARGET_SCALE = 20.0


def dynamic_margin_sigmoid_balanced_loss(
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
    beta=0.4,
    eps=1e-8,
):
    """Scale-safe correction of :func:`dynamic_margin_sigmoid_loss`.

    It retains the original four signals (triplet ordering, distance
    calibration, magnitude decoding, and direction ranking), but changes how
    each signal is allocated:

    * the sigmoid margin is largest for ambiguous near-gap triplets and falls
      toward ``base_margin`` for easy far negatives;
        * metric distance uses the Paper-1 absolute log-ratio geometry rather than
            mixing it with the magnitude head's ``log1p`` target;
        * the calibration curve is ``0.5 * d/(1+d)``, which is monotonic but keeps
      the implied target cosine in [0, 1] instead of pushing far, semantically
      related sentences toward cosine -1;
    * Smooth L1 replaces calibration/head MSE to reduce outlier influence;
    * head targets and its directional margin use the same normalized scale;
    * a temperature-scaled Softplus replaces the discontinuous hard hinge.

    This is a new experimental loss, not a claim that the corrected composite
    will outperform the simpler ``cosent_log_ratio`` baseline.
    """
    anchor = F.normalize(anchor_emb, dim=-1)
    pos = F.normalize(pos_emb, dim=-1)
    neg = F.normalize(neg_emb, dim=-1)

    pos_cos_dist = 1.0 - F.cosine_similarity(anchor, pos, dim=-1)
    neg_cos_dist = 1.0 - F.cosine_similarity(anchor, neg, dim=-1)

    metric_log_a = torch.log(anchor_value + eps)
    metric_log_p = torch.log(pos_value + eps)
    metric_log_n = torch.log(neg_value + eps)
    log_pos_dist = torch.abs(metric_log_a - metric_log_p)
    log_neg_dist = torch.abs(metric_log_a - metric_log_n)

    # Valid triplets have gap >= 0. Near/ambiguous gaps receive up to 1.5*m0;
    # easy far gaps approach m0, so they deactivate instead of owning a batch.
    distance_gap = (log_neg_dist - log_pos_dist).clamp(min=0.0)
    dyn_margin = base_margin * (1.0 + torch.sigmoid(-distance_gap))
    triplet_loss = F.relu(
        pos_cos_dist - neg_cos_dist + dyn_margin
    ).mean()

    # d/(1+d) retains more separation among medium/far values than tanh. The
    # 0.5 semantic guardrail maps target normalized cosine distance to [0,.5),
    # hence target cosine similarity remains in (0,1] rather than approaching -1.
    def calibration_target(distance):
        return 0.5 * distance / (1.0 + distance)

    calibration_loss = (
        F.smooth_l1_loss(pos_cos_dist / 2.0, calibration_target(log_pos_dist))
        + F.smooth_l1_loss(neg_cos_dist / 2.0, calibration_target(log_neg_dist))
    ) / 2.0

    target_a = torch.log1p(anchor_value) / HEAD_TARGET_SCALE
    target_p = torch.log1p(pos_value) / HEAD_TARGET_SCALE
    target_n = torch.log1p(neg_value) / HEAD_TARGET_SCALE
    head_loss = (
        F.smooth_l1_loss(anchor_score, target_a)
        + F.smooth_l1_loss(pos_score, target_p)
        + F.smooth_l1_loss(neg_score, target_n)
    ) / 3.0

    # Direction is auxiliary: use a small margin in normalized target units
    # and a smooth, temperature-scaled penalty whose magnitude stays bounded.
    rank_margin = 0.1 / HEAD_TARGET_SCALE
    rank_temperature = 0.05
    ap_direction = torch.sign(target_a - target_p)
    an_direction = torch.sign(target_a - target_n)

    def smooth_rank(direction, score_gap):
        violation = (rank_margin - direction * score_gap) / rank_temperature
        return (rank_temperature * F.softplus(violation)).mean()

    rank_loss = (
        smooth_rank(ap_direction, anchor_score - pos_score)
        + smooth_rank(an_direction, anchor_score - neg_score)
    ) / 2.0

    metric_loss = triplet_loss + calibration_loss
    supervision_loss = beta * head_loss + (1.0 - beta) * rank_loss
    total_loss = alpha * metric_loss + (1.0 - alpha) * supervision_loss

    components = {
        "triplet": triplet_loss.item(),
        "calibration": calibration_loss.item(),
        "head": head_loss.item(),
        "rank": rank_loss.item(),
        "margin_mean": dyn_margin.mean().item(),
        "metric": metric_loss.item(),
        "supervision": supervision_loss.item(),
        "total": total_loss.item(),
    }
    return total_loss, components


def _graded_pair_scores(anchor, pos, neg, anchor_value, pos_value, neg_value, eps):
    """Flatten a triplet batch into 2B (anchor, candidate) pairs with cosine
    similarity and a graded, unit-free target = -|log ratio| (higher = closer)."""
    sims = torch.cat([
        F.cosine_similarity(anchor, pos, dim=-1),
        F.cosine_similarity(anchor, neg, dim=-1),
    ])
    log_a = torch.log(anchor_value + eps)
    targets = torch.cat([
        -torch.abs(torch.log(pos_value + eps) - log_a),
        -torch.abs(torch.log(neg_value + eps) - log_a),
    ])
    return sims, targets


def cosent_log_ratio_loss(
    anchor_emb,
    pos_emb,
    neg_emb,
    anchor_score,
    pos_score,
    neg_score,
    anchor_value,
    pos_value,
    neg_value,
    tau=0.05,
    eps=1e-8,
):
    """CoSENT-style graded ranking loss over in-batch (anchor, candidate) pairs.

    Every triplet contributes two pairs — (anchor, positive) and (anchor,
    negative) — with target similarity -|log(candidate/anchor)| (unit-free, so
    pairs are comparable across anchors). For any two pairs where target_k >
    target_l, the loss penalizes cos_l > cos_k:

        loss = log(1 + sum_{t_k > t_l} exp((cos_l - cos_k) / tau))

    Unlike the MSE log-distance alignment in the dynamic_margin_* losses, this
    penalizes only ordering violations (ranking, not regression-to-cosine) and
    the temperature counteracts similarity compression. It directly optimizes
    the Spearman-vs-log-distance metric the eval suites measure. Scores/head
    are unused — pure metric-space loss."""
    sims, targets = _graded_pair_scores(
        anchor_emb, pos_emb, neg_emb, anchor_value, pos_value, neg_value, eps
    )

    diff = (sims.unsqueeze(0) - sims.unsqueeze(1)) / tau   # diff[k, l] = (cos_l - cos_k)/tau
    mask = targets.unsqueeze(1) > targets.unsqueeze(0)     # target_k > target_l
    violations = diff[mask]
    zero = torch.zeros(1, device=sims.device, dtype=sims.dtype)
    total_loss = torch.logsumexp(torch.cat([zero, violations]), dim=0)

    components = {
        "cosent": total_loss.item(),
        "n_pairs": int(mask.sum().item()),
        "total": total_loss.item(),
    }
    return total_loss, components


def cosent_plus_head_loss(
    anchor_emb,
    pos_emb,
    neg_emb,
    anchor_score,
    pos_score,
    neg_score,
    anchor_value,
    pos_value,
    neg_value,
    beta=0.4,
    tau=0.05,
    eps=1e-8,
):
    """cosent_log_ratio + a downweighted auxiliary head regression with
    normalized targets (log1p(value)/HEAD_TARGET_SCALE, so head_loss starts
    ~O(0.1) instead of ~60 and cannot dominate the ranking signal).
    total = cosent + beta * head."""
    cosent, components = cosent_log_ratio_loss(
        anchor_emb, pos_emb, neg_emb,
        anchor_score, pos_score, neg_score,
        anchor_value, pos_value, neg_value,
        tau=tau, eps=eps,
    )

    t_a = torch.log1p(anchor_value) / HEAD_TARGET_SCALE
    t_p = torch.log1p(pos_value) / HEAD_TARGET_SCALE
    t_n = torch.log1p(neg_value) / HEAD_TARGET_SCALE
    head_loss = (
        F.mse_loss(anchor_score, t_a)
        + F.mse_loss(pos_score, t_p)
        + F.mse_loss(neg_score, t_n)
    ) / 3.0

    total_loss = cosent + beta * head_loss
    components.update({
        "head": head_loss.item(),
        "total": total_loss.item(),
    })
    return total_loss, components


def dynamic_margin_softplus_capped_loss(
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
    """dynamic_margin_softplus_loss with its two scale problems fixed:

    1. The dynamic margin is capped at 0.8. Uncapped, base_margin*(1+log_diff)
       exceeds 2 (the max possible cosine-distance gap) for log_diff > 9, and on
       the old 50x-fallback data the *median* triplet demanded a ~0.97 margin —
       the ReLU never deactivated, so gradient stayed on easy far negatives
       forever instead of near-ties.
    2. Head-regression targets are normalized by HEAD_TARGET_SCALE so head_loss
       starts ~O(0.1) instead of ~60 and early training isn't dominated by the
       auxiliary probe."""
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
    dyn_margin = (base_margin * (1.0 + log_diff.clamp(min=0))).clamp(max=0.8)
    triplet_loss = F.relu(pos_cos - neg_cos + dyn_margin).mean()

    def scale(x):
        return x / (1.0 + x)

    log_distance_loss = (
        F.mse_loss(pos_cos / 2.0, scale(log_pos))
        + F.mse_loss(neg_cos / 2.0, scale(log_neg))
        + F.mse_loss(pn_cos / 2.0, scale(log_pn))
    ) / 3.0

    t_a = log_a / HEAD_TARGET_SCALE
    t_p = log_p / HEAD_TARGET_SCALE
    t_n = log_n / HEAD_TARGET_SCALE
    head_loss = (
        F.mse_loss(anchor_score, t_a)
        + F.mse_loss(pos_score, t_p)
        + F.mse_loss(neg_score, t_n)
    ) / 3.0

    margin_rank = 0.1 / HEAD_TARGET_SCALE  # rank margin on the same normalized scale
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


def dynamic_margin_softplus_v2_loss(
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
    beta=0.4,
    tau=0.05,
    eps=1e-8,
):
    """Scale-safe v2 of the historical three-edge Softplus composite.

    The historical strengths are retained: anchor-positive/anchor-negative
    triplet ordering, calibration of all three triangle edges, magnitude
    decoding, and smooth direction ranking. The unsafe details are replaced:

    * all metric terms use absolute log-ratio distance;
    * the bounded dynamic margin is strongest for ambiguous near-gap rows and
      approaches ``base_margin`` for easy far negatives;
    * calibration maps to nonnegative target cosine and uses Smooth L1;
    * magnitude targets and the rank margin share a normalized scale;
    * temperature-scaled Softplus keeps rank-loss magnitude controlled.

    ``tau`` is the smooth-rank temperature. It is intentionally exposed through
    ``RunConfig.loss_tau`` so a later ablation can change it without code edits.
    """
    anchor = F.normalize(anchor_emb, dim=-1)
    pos = F.normalize(pos_emb, dim=-1)
    neg = F.normalize(neg_emb, dim=-1)

    pos_cos_dist = 1.0 - F.cosine_similarity(anchor, pos, dim=-1)
    neg_cos_dist = 1.0 - F.cosine_similarity(anchor, neg, dim=-1)
    pn_cos_dist = 1.0 - F.cosine_similarity(pos, neg, dim=-1)

    metric_log_a = torch.log(anchor_value + eps)
    metric_log_p = torch.log(pos_value + eps)
    metric_log_n = torch.log(neg_value + eps)
    pos_numeric_dist = torch.abs(metric_log_a - metric_log_p)
    neg_numeric_dist = torch.abs(metric_log_a - metric_log_n)
    pn_numeric_dist = torch.abs(metric_log_p - metric_log_n)

    # At an ambiguous gap of zero the margin is 1.5*m0. As a negative becomes
    # clearly farther than the positive, the margin smoothly approaches m0.
    numeric_gap = (neg_numeric_dist - pos_numeric_dist).clamp(min=0.0)
    dyn_margin = base_margin * (1.0 + 0.5 / (1.0 + numeric_gap))
    triplet_loss = F.relu(
        pos_cos_dist - neg_cos_dist + dyn_margin
    ).mean()

    # If normalized cosine distance equals this target, the implied cosine is
    # 1/(1+d): monotonic, positive, and less destructive to semantic geometry.
    def calibration_target(distance):
        return 0.5 * distance / (1.0 + distance)

    calibration_loss = (
        F.smooth_l1_loss(
            pos_cos_dist / 2.0, calibration_target(pos_numeric_dist)
        )
        + F.smooth_l1_loss(
            neg_cos_dist / 2.0, calibration_target(neg_numeric_dist)
        )
        + F.smooth_l1_loss(
            pn_cos_dist / 2.0, calibration_target(pn_numeric_dist)
        )
    ) / 3.0

    target_a = torch.log1p(anchor_value) / HEAD_TARGET_SCALE
    target_p = torch.log1p(pos_value) / HEAD_TARGET_SCALE
    target_n = torch.log1p(neg_value) / HEAD_TARGET_SCALE
    head_loss = (
        F.smooth_l1_loss(anchor_score, target_a)
        + F.smooth_l1_loss(pos_score, target_p)
        + F.smooth_l1_loss(neg_score, target_n)
    ) / 3.0

    rank_margin = 0.1 / HEAD_TARGET_SCALE
    safe_tau = max(float(tau), eps)
    ap_direction = torch.sign(target_a - target_p)
    an_direction = torch.sign(target_a - target_n)

    def smooth_rank(direction, score_gap):
        violation = (rank_margin - direction * score_gap) / safe_tau
        return (safe_tau * F.softplus(violation)).mean()

    rank_loss = (
        smooth_rank(ap_direction, anchor_score - pos_score)
        + smooth_rank(an_direction, anchor_score - neg_score)
    ) / 2.0

    metric_loss = triplet_loss + calibration_loss
    supervision_loss = beta * head_loss + (1.0 - beta) * rank_loss
    total_loss = alpha * metric_loss + (1.0 - alpha) * supervision_loss

    components = {
        "triplet": triplet_loss.item(),
        "calibration": calibration_loss.item(),
        "head": head_loss.item(),
        "rank": rank_loss.item(),
        "margin_mean": dyn_margin.mean().item(),
        "metric": metric_loss.item(),
        "supervision": supervision_loss.item(),
        "total": total_loss.item(),
    }
    return total_loss, components


LOSS_REGISTRY = {
    "dynamic_margin_sigmoid": dynamic_margin_sigmoid_loss,
    "dynamic_margin_sigmoid_balanced": dynamic_margin_sigmoid_balanced_loss,
    "dynamic_margin_softplus": dynamic_margin_softplus_loss,
    "dynamic_margin_softplus_infosec": dynamic_margin_softplus_infosec_loss,
    "dynamic_margin_softplus_capped": dynamic_margin_softplus_capped_loss,
    "dynamic_margin_softplus_v2": dynamic_margin_softplus_v2_loss,
    "cosent_log_ratio": cosent_log_ratio_loss,
    "cosent_plus_head": cosent_plus_head_loss,
}


def get_loss_fn(name: str):
    try:
        return LOSS_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown loss_fn {name!r}. Available: {list(LOSS_REGISTRY)}"
        ) from None
