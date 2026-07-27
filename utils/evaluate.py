import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

############################################
# CONFIG
############################################

MODEL_NAME = "modernbert-base"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLED_NEGATIVES = 8   # for Recall@K
RECALL_K = [1, 5, 10]

############################################
# POOLING
############################################

def mean_pooling(model_output, attention_mask):
    """
    Correct pooling for ModernBERT.
    """
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)

############################################
# LOAD MODEL
############################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

############################################
# EMBEDDING
############################################

@torch.no_grad()
def embed_texts(texts):
    """
    Returns L2-normalized sentence embeddings.
    """
    all_emb = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch_texts = texts[i:i+BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = model(**inputs)
        emb = mean_pooling(outputs, inputs["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        all_emb.append(emb.cpu())

    return torch.cat(all_emb, dim=0).numpy()

############################################
# METRICS (CORE)
############################################

def triplet_accuracy(A, P, N):
    """
    P(sim(a,p) > sim(a,n))
    """
    pos = np.sum(A * P, axis=1)
    neg = np.sum(A * N, axis=1)
    return float(np.mean(pos > neg))


def cosine_gap(A, P, N):
    """
    Mean(sim(a,p) - sim(a,n))
    """
    pos = np.sum(A * P, axis=1)
    neg = np.sum(A * N, axis=1)
    return float(np.mean(pos - neg))


def pairwise_mrr(A, P, N):
    """
    MRR over {positive, negative}
    Rank = 1 if pos > neg else 2
    """
    pos = np.sum(A * P, axis=1)
    neg = np.sum(A * N, axis=1)

    ranks = np.where(pos > neg, 1, 2)
    return float(np.mean(1.0 / ranks))

############################################
# MRR@K WITH SAMPLED NEGATIVES
############################################

def mrr_at_k_sampled(A, P, N, k, num_extra_negs=8):
    """
    Mean Reciprocal Rank at K with sampled negatives.

    For each anchor:
      candidates = {P[i], N[i], N[j1], ..., N[jm]}
      If P[i] ranks at position r <= K, contribute 1/r, else 0.
    """
    n = len(A)
    rr = []

    for i in range(n):
        neg_indices = random.sample(
            [j for j in range(n) if j != i],
            min(num_extra_negs, n - 1)
        )

        candidates = [P[i], N[i]] + [N[j] for j in neg_indices]
        sims = np.dot(candidates, A[i])

        order = np.argsort(-sims)
        rank = np.where(order == 0)[0][0] + 1  # index 0 = positive

        if rank <= k:
            rr.append(1.0 / rank)
        else:
            rr.append(0.0)

    return float(np.mean(rr))


############################################
# RECALL@K WITH SAMPLED NEGATIVES
############################################

def recall_at_k_sampled(A, P, N, k, num_extra_negs=8):
    """
    For each anchor:
      candidates = {P[i], N[i], N[j1], ..., N[jm]}
      checks if P[i] appears in top-K
    """
    n = len(A)
    hits = []

    for i in range(n):
        # sample extra negatives
        neg_indices = random.sample(
            [j for j in range(n) if j != i],
            min(num_extra_negs, n - 1)
        )

        candidates = [P[i], N[i]] + [N[j] for j in neg_indices]
        labels = [1] + [0] * (1 + len(neg_indices))

        sims = np.dot(candidates, A[i])
        topk = np.argsort(-sims)[:k]

        hits.append(any(labels[j] == 1 for j in topk))

    return float(np.mean(hits))

############################################
# LOAD DATA
############################################

def load_triplets(path):
    anchors, positives, negatives = [], [], []

    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            anchors.append(row["anchor"])
            positives.append(row["positive"])
            negatives.append(row["negative"])

    return anchors, positives, negatives

############################################
# EVALUATION
############################################

def evaluate(triplet_file):
    print("Loading triplets...")
    anchors, positives, negatives = load_triplets(triplet_file)
    print(f"Loaded {len(anchors)} triplets")

    print("\nComputing embeddings...")
    A = embed_texts(anchors)
    P = embed_texts(positives)
    N = embed_texts(negatives)

    print("\nComputing metrics...")
    results = {}

    results["triplet_accuracy"] = triplet_accuracy(A, P, N)
    results["cosine_gap"] = cosine_gap(A, P, N)
    results["pairwise_mrr"] = pairwise_mrr(A, P, N)

    for k in RECALL_K:
        results[f"recall@{k}"] = recall_at_k_sampled(
            A, P, N, k, NUM_SAMPLED_NEGATIVES
        )
        results[f"mrr@{k}"] = mrr_at_k_sampled(
            A, P, N, k, NUM_SAMPLED_NEGATIVES
        )

    print("\n===== RESULTS =====")
    for k, v in results.items():
        print(f"{k:20s}: {v:.4f}")

    return results

############################################
# ENTRY POINT
############################################

if __name__ == "__main__":
    # Example path
    evaluate("test_triplets.jsonl")
