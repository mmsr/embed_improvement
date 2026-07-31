"""Evaluation for numeracy embedding models: triplet metrics, similarity probes,
and ordering scores (Spearman vs log-distance). No model is loaded at import time;
every function takes (model, tokenizer) explicitly so the same session can compare
several checkpoints. Entry point: evaluate_adapter()."""
from __future__ import annotations

import json
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

RECALL_K = [1, 5, 10]
NUM_SAMPLED_NEGATIVES = 8


def load_eval_model(base_model: str, adapter_path: str | None = None, device: str | None = None):
    """Load base encoder, optionally with a LoRA adapter, in eval mode."""
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModel.from_pretrained(base_model)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return model.to(device).eval(), tokenizer


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)


@torch.no_grad()
def embed_texts(model, tokenizer, texts, batch_size=32, max_length=128, progress=True):
    """L2-normalized mean-pooled embeddings, (N, hidden) numpy array."""
    device = next(model.parameters()).device
    all_emb = []
    rng = range(0, len(texts), batch_size)
    for i in tqdm(rng, desc="Embedding", leave=False, disable=not progress):
        inputs = tokenizer(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        emb = mean_pooling(model(**inputs), inputs["attention_mask"])
        all_emb.append(F.normalize(emb, p=2, dim=1).cpu())
    return torch.cat(all_emb, dim=0).numpy()


############################################
# TRIPLET METRICS
############################################

def triplet_accuracy(A, P, N):
    return float(np.mean(np.sum(A * P, axis=1) > np.sum(A * N, axis=1)))


def cosine_gap(A, P, N):
    return float(np.mean(np.sum(A * P, axis=1) - np.sum(A * N, axis=1)))


def pairwise_mrr(A, P, N):
    pos = np.sum(A * P, axis=1)
    neg = np.sum(A * N, axis=1)
    return float(np.mean(1.0 / np.where(pos > neg, 1, 2)))


def _sampled_candidates(rng, i, n, P, N, num_extra_negs):
    neg_idx = rng.sample([j for j in range(n) if j != i], min(num_extra_negs, n - 1))
    return [P[i], N[i]] + [N[j] for j in neg_idx]


def recall_at_k_sampled(A, P, N, k, num_extra_negs=8, seed=42):
    rng = random.Random(seed)
    n = len(A)
    hits = []
    for i in range(n):
        candidates = _sampled_candidates(rng, i, n, P, N, num_extra_negs)
        topk = np.argsort(-np.dot(candidates, A[i]))[:k]
        hits.append(0 in topk)  # index 0 = positive
    return float(np.mean(hits))


def mrr_at_k_sampled(A, P, N, k, num_extra_negs=8, seed=42):
    rng = random.Random(seed)
    n = len(A)
    rr = []
    for i in range(n):
        candidates = _sampled_candidates(rng, i, n, P, N, num_extra_negs)
        order = np.argsort(-np.dot(candidates, A[i]))
        rank = int(np.where(order == 0)[0][0]) + 1
        rr.append(1.0 / rank if rank <= k else 0.0)
    return float(np.mean(rr))


def load_triplets(path):
    anchors, positives, negatives = [], [], []
    anchor_values, positive_values, negative_values = [], [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            anchors.append(row.get("anchor") or row["comment"])
            positives.append(row.get("positive_rewritten") or row["positive"])
            negatives.append(row.get("negative_rewritten") or row["negative"])
            anchor_values.append(float(row["number"]))
            positive_values.append(float(row["positive_number"]))
            negative_values.append(float(row["negative_number"]))
    return (
        anchors,
        positives,
        negatives,
        np.asarray(anchor_values),
        np.asarray(positive_values),
        np.asarray(negative_values),
    )


def evaluate_triplets(model, tokenizer, triplet_file, batch_size=32, max_length=128):
    """Full triplet metric dict for one JSONL file."""
    anchors, positives, negatives, anchor_values, _, negative_values = load_triplets(
        triplet_file
    )
    A = embed_texts(model, tokenizer, anchors, batch_size, max_length)
    P = embed_texts(model, tokenizer, positives, batch_size, max_length)
    N = embed_texts(model, tokenizer, negatives, batch_size, max_length)

    pos_sims = np.sum(A * P, axis=1)
    neg_sims = np.sum(A * N, axis=1)

    results = {
        "n_triplets": len(anchors),
        "triplet_accuracy": triplet_accuracy(A, P, N),
        "cosine_gap": cosine_gap(A, P, N),
        "pairwise_mrr": pairwise_mrr(A, P, N),
    }
    for k in RECALL_K:
        results[f"recall@{k}"] = recall_at_k_sampled(A, P, N, k, NUM_SAMPLED_NEGATIVES)
        results[f"mrr@{k}"] = mrr_at_k_sampled(A, P, N, k, NUM_SAMPLED_NEGATIVES)

    symmetric_ratio = np.maximum(anchor_values, negative_values) / np.minimum(
        anchor_values, negative_values
    )
    slices = {
        "very_near_<1.2x": symmetric_ratio < 1.2,
        "near_<1.5x": symmetric_ratio < 1.5,
        "mid_1.5-5x": (symmetric_ratio >= 1.5) & (symmetric_ratio < 5.0),
        "far_>=5x": symmetric_ratio >= 5.0,
    }
    for name, mask in slices.items():
        count = int(mask.sum())
        results[f"slice/{name}/n"] = count
        if count:
            results[f"slice/{name}/triplet_accuracy"] = float(
                np.mean(pos_sims[mask] > neg_sims[mask])
            )
            results[f"slice/{name}/cosine_gap"] = float(
                np.mean(pos_sims[mask] - neg_sims[mask])
            )
    return results


############################################
# ORDERING PROBES (test suites)
############################################

TEST_SUITES = {
    "decimal_magnitude": {
        "anchor": "0.1 USD",
        "candidates": ["0.099 USD", "0.101 USD", "0.09 USD", "0.11 USD",
                       "0.05 USD", "0.2 USD", "0.01 USD", "1 USD"],
        "description": "Sub-unit magnitude: can decimals be ordered by relative distance?",
    },
    "digit_bias": {
        "anchor": "321 ml",
        "candidates": ["320 ml", "329 ml", "312 ml", "301 ml", "231 ml",
                       "132 ml", "123 ml", "221 ml", "421 ml"],
        "description": "Left-to-right digit bias: are first-digit matches overweighted?",
    },
    "monotonic_decay": {
        "anchor": "500 ml",
        "candidates": ["499 ml", "501 ml", "490 ml", "510 ml", "450 ml", "550 ml",
                       "400 ml", "600 ml", "300 ml", "700 ml", "100 ml", "900 ml"],
        "description": "Near-tie ordering: does similarity decay with numeric distance?",
    },
    "place_value": {
        "anchor": "345 ml",
        "candidates": ["346 ml", "355 ml", "445 ml", "344 ml", "335 ml", "245 ml"],
        "description": "Is a +-1 change scored closer than +-10 or +-100?",
    },
    "prefix_trap": {
        "anchor": "199 ml",
        "candidates": ["200 ml", "198 ml", "190 ml", "299 ml", "109 ml", "119 ml", "999 ml"],
        "description": "Does the true nearest neighbor beat digit-sharing distractors?",
    },
    "permutation_trap": {
        "anchor": "247 ml",
        "candidates": ["274 ml", "427 ml", "724 ml", "742 ml", "472 ml",
                       "200 ml", "250 ml", "300 ml"],
        "description": "Are permuted digits confused with numeric proximity?",
    },
    "power_of_ten_boundary": {
        "anchor": "99 kg",
        "candidates": ["98 kg", "100 kg", "90 kg", "110 kg",
                       "9.9 kg", "990 kg", "0.99 kg", "9900 kg"],
        "description": "Magnitude boundary: does crossing 99 to 100 remain a near change?",
    },
    "thousand_magnitude": {
        "anchor": "1000 USD",
        "candidates": ["999 USD", "1001 USD", "900 USD", "1100 USD",
                       "500 USD", "2000 USD", "100 USD", "10000 USD"],
        "description": "Thousands: preserve relative ordering and reciprocal-ratio ties.",
    },
    "large_magnitude": {
        "anchor": "100000 shares",
        "candidates": ["99900 shares", "100100 shares", "90000 shares", "110000 shares",
                       "50000 shares", "200000 shares", "10000 shares", "1000000 shares"],
        "description": "Large values: does ordering remain stable across five to six digits?",
    },
}


def make_random_suites(n_suites=20, seed=42, units=("ml", "kg", "USD", "shares", "%")):
    """Randomized ordering suites across magnitudes/units, for robustness beyond
    the 5 hand-picked probes (which stay fixed for comparability with old results)."""
    rng = random.Random(seed)
    suites = {}
    for i in range(n_suites):
        unit = rng.choice(units)
        anchor = round(rng.uniform(1, 10) * 10 ** rng.randint(0, 5), 2)
        ratios = [1.002, 1.02, 1.1, 1.25, 1.5, 2.0, 5.0, 20.0]
        cands = []
        for r in ratios:
            r = r if rng.random() < 0.5 else 1.0 / r
            cands.append(f"{round(anchor * r, 2):g} {unit}")
        suites[f"random_{i}"] = {
            "anchor": f"{anchor:g} {unit}",
            "candidates": cands,
            "description": "randomized ordering probe",
        }
    return suites


def _spearman(x, y):
    def rank(a):
        """Average ranks for ties (the standard Spearman convention)."""
        values = np.asarray(a, float)
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return ranks
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _pairwise_ordering_accuracy(similarities, distances):
    """Fraction of unequal-distance candidate pairs in the correct order.

    True numeric ties are excluded rather than arbitrarily broken. Predicted
    similarity ties receive half credit.
    """
    correct = 0.0
    compared = 0
    for i in range(len(distances)):
        for j in range(i + 1, len(distances)):
            if distances[i] == distances[j]:
                continue
            compared += 1
            expected = distances[i] < distances[j]
            if similarities[i] == similarities[j]:
                correct += 0.5
            elif (similarities[i] > similarities[j]) == expected:
                correct += 1.0
    return float(correct / compared) if compared else float("nan")


def score_suite(model, tokenizer, anchor, candidates, max_length=128):
    """Ordering quality for one probe: Spearman rho between similarity rank and
    |log ratio| distance rank (1.0 = perfect numeric ordering), plus sim spread."""
    texts = [anchor] + list(candidates)
    embs = embed_texts(model, tokenizer, texts, max_length=max_length, progress=False)
    sims = embs[1:] @ embs[0]
    anchor_val = float(anchor.split()[0].replace(",", ""))
    log_dists = [
        abs(math.log(float(c.split()[0].replace(",", "")) / anchor_val))
        for c in candidates
    ]
    pairwise_accuracy = _pairwise_ordering_accuracy(sims, log_dists)
    return {
        "spearman": _spearman(-sims, log_dists),
        "pairwise_ordering": pairwise_accuracy,
        "inversion_rate": 1.0 - pairwise_accuracy,
        "sim_spread": float(sims.max() - sims.min()),
        "ranking": sorted(zip(candidates, sims.tolist()), key=lambda t: -t[1]),
    }


def ordering_scores(model, tokenizer, suites=None, max_length=128, verbose=False):
    """Run all suites; returns flat metrics: ordering@<suite> per suite plus
    ordering_mean and sim_spread_mean (the two headline numbers)."""
    suites = suites or TEST_SUITES
    out, rhos, pairwise_scores, inversion_rates, spreads = {}, [], [], [], []
    for name, s in suites.items():
        r = score_suite(model, tokenizer, s["anchor"], s["candidates"], max_length)
        out[f"ordering@{name}"] = round(r["spearman"], 4)
        out[f"pairwise_ordering@{name}"] = round(r["pairwise_ordering"], 4)
        out[f"inversion_rate@{name}"] = round(r["inversion_rate"], 4)
        rhos.append(r["spearman"])
        pairwise_scores.append(r["pairwise_ordering"])
        inversion_rates.append(r["inversion_rate"])
        spreads.append(r["sim_spread"])
        if verbose:
            print(f"\n{name}: rho={r['spearman']:.3f}  "
                f"pairwise={r['pairwise_ordering']:.3f}  "
                f"inversions={r['inversion_rate']:.3f}  spread={r['sim_spread']:.3f}"
                  f"  ({s['description']})")
            print(f"  Anchor: {s['anchor']}")
            for text, sim in r["ranking"]:
                print(f"  {text:>12} : {sim:.4f}")
    out["ordering_mean"] = round(float(np.nanmean(rhos)), 4)
    out["pairwise_ordering_mean"] = round(float(np.nanmean(pairwise_scores)), 4)
    out["inversion_rate_mean"] = round(float(np.nanmean(inversion_rates)), 4)
    out["sim_spread_mean"] = round(float(np.mean(spreads)), 4)
    return out


############################################
# TOP-LEVEL RUNNER
############################################

def evaluate_adapter(
    base_model: str,
    adapter_path: str | None,
    triplet_files: dict[str, str] | None = None,
    include_random_suites: bool = True,
    verbose: bool = True,
) -> dict:
    """Evaluate one checkpoint end to end. triplet_files maps a short label
    (e.g. 'test_same') to a JSONL path; labels prefix the metric names.
    Returns one flat dict ready for training.experiment.log_result()."""
    model, tokenizer = load_eval_model(base_model, adapter_path)
    metrics: dict = {}

    for label, path in (triplet_files or {}).items():
        if verbose:
            print(f"── triplets: {label} ──")
        res = evaluate_triplets(model, tokenizer, path)
        metrics.update({f"{label}/{k}": v for k, v in res.items()})
        if verbose:
            for k, v in res.items():
                print(f"  {k:<20s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    metrics.update(ordering_scores(model, tokenizer, verbose=verbose))
    if include_random_suites:
        rand = ordering_scores(model, tokenizer, suites=make_random_suites())
        metrics["ordering_mean_random"] = rand["ordering_mean"]
        metrics["pairwise_ordering_mean_random"] = rand["pairwise_ordering_mean"]
        metrics["inversion_rate_mean_random"] = rand["inversion_rate_mean"]
        metrics["sim_spread_mean_random"] = rand["sim_spread_mean"]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics
