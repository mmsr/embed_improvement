"""
Numeracy 600K → Training Triplet Dataset Builder
==================================================
Reads Numeracy_600K_comment.json, filters negative numbers,
samples records according to the ideal magnitude distribution,
generates positive and negative numbers satisfying all signal
quality and distance constraints, and writes a clean JSONL file.

Output fields per record (all source fields + generated numbers):
    id, UNIQUE_STORY_INDEX, offset, length, magnitude, comment, number,
    positive_number, negative_number

Usage:
    python build_dataset.py \
        --input   Numeracy_600K_comment.json \
        --output  train_triplets.jsonl \
        --total   100000 \
        [--seed   42]
"""

import re
import json
import math
import random
import logging
import argparse
from collections import defaultdict, Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

SEED = 42


# ─────────────────────────────────────────────────────────────
# TARGET MAGNITUDE DISTRIBUTION
# fraction of total dataset per magnitude bucket
# ─────────────────────────────────────────────────────────────
MAG_TARGETS = {
    -2: 0.04,   # decimal 0.01–0.09
    -1: 0.04,   # decimal 0.1–0.99
     0: 0.15,   # 1–9
     1: 0.20,   # 10–99
     2: 0.20,   # 100–999
     3: 0.15,   # 1000–9999
     4: 0.10,   # 10000–99999
     5: 0.07,   # 100000–999999
     6: 0.01,   # 1M+
}
# zero anchors: 5% of total — handled separately


# ─────────────────────────────────────────────────────────────
# GRADED TRIPLET GENERATION CONFIG
#
# The previous generator combined MIN_NEG_LOG_FACTOR=3.0 (measured on
# log1p of the LINEAR distance) with per-magnitude ratio caps of 10-20x.
# Those constraints are mutually unsatisfiable for most anchors, so 59.7%
# of negatives in train_triplets_04092026.jsonl fell through every strategy
# to the uncapped last-resort `max(anchor,pos)*50` — median neg ratio was
# exactly ln(50)=3.91 in log space, and near-tie ordering never appeared
# in training. This version samples the negative's |log ratio| directly
# from graded bands instead, so hardness is an explicit choice.
#
# Bands: (lo_ratio, hi_ratio, fraction of dataset). Sampled log-uniformly
# within the band; direction (x or /) is random.
# ─────────────────────────────────────────────────────────────
NEG_RATIO_BANDS = [
    (1.05,  1.5, 0.40),   # near  — the regime ordering evals probe (499 vs 501)
    (1.5,   5.0, 0.30),   # mid
    (5.0,  50.0, 0.30),   # far   — easy magnitude-level negatives
]
BAND_NAMES = ["near", "mid", "far"]

# Positive |ratio| range per band. Tighter positives accompany nearer
# negatives so that |log(neg/a)| >= NEG_POS_MARGIN * |log(pos/a)| always
# holds by construction (checked again after rounding).
POS_RATIO_BY_BAND = [
    (1.005, 1.03),
    (1.01,  1.15),
    (1.02,  1.30),
]
NEG_POS_MARGIN = 1.5
MAX_ATTEMPTS   = 20


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

NUMBER_RE = re.compile(r'-?\d[\d,]*(?:\.\d+)?')


def get_magnitude(x: float) -> int:
    if x <= 0:
        return -99
    return int(math.floor(math.log10(x)))


def format_number(x: float) -> str:
    """Format cleanly — integer if whole, else up to 4dp stripped of trailing zeros."""
    if x != 0 and abs(x) >= 1 and x == int(x):
        return str(int(x))
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s


def sentence_has_negative(comment: str, number: float) -> bool:
    """
    Return True if the anchor number appears with a leading minus in the sentence.
    Used to filter out sign-stripped records.
    """
    num_str = format_number(number)
    # look for -<number> pattern in sentence
    pattern = re.compile(r'-\s*' + re.escape(num_str))
    return bool(pattern.search(comment))


# ─────────────────────────────────────────────────────────────
# GRADED PAIR GENERATION
# ─────────────────────────────────────────────────────────────

def sample_log_uniform(lo: float, hi: float) -> float:
    return math.exp(random.uniform(math.log(lo), math.log(hi)))


def round_like(anchor: float, x: float) -> float:
    """Round generated numbers to text-friendly precision (2dp for values
    >= 1, 4dp for decimals)."""
    return round(x, 2 if anchor >= 1 else 4)


def log_ratio_dist(a: float, b: float) -> float:
    return abs(math.log(b / a))


def loss_space_dist(a: float, b: float) -> float:
    """Distance in the space the training losses use: |log1p(a) - log1p(b)|."""
    return abs(math.log1p(a) - math.log1p(b))


def generate_pair(anchor: float) -> tuple[float, float, int] | None:
    """
    Sample (positive, negative, band_index) for an anchor.

    Band is drawn from NEG_RATIO_BANDS fractions; the negative's |log ratio|
    is log-uniform within the band and the positive's ratio comes from the
    band-matched POS_RATIO_BY_BAND range, so the triplet is valid by
    construction. Both numbers are re-validated after rounding, in both
    log-ratio space and the loss's log1p-value space. Returns None if no
    valid pair survives rounding (tiny decimals) after MAX_ATTEMPTS.
    """
    band = random.choices(
        range(len(NEG_RATIO_BANDS)), weights=[b[2] for b in NEG_RATIO_BANDS]
    )[0]
    pos_lo, pos_hi = POS_RATIO_BY_BAND[band]
    neg_lo, neg_hi, _ = NEG_RATIO_BANDS[band]

    for _ in range(MAX_ATTEMPTS):
        pos_ratio = sample_log_uniform(pos_lo, pos_hi)
        neg_ratio = sample_log_uniform(neg_lo, neg_hi)

        pos = anchor * pos_ratio if random.random() < 0.5 else anchor / pos_ratio
        neg = anchor * neg_ratio if random.random() < 0.5 else anchor / neg_ratio
        pos, neg = round_like(anchor, pos), round_like(anchor, neg)

        if pos <= 0 or neg <= 0:
            continue
        # rounding must not collapse the numbers into identical text
        if len({format_number(anchor), format_number(pos), format_number(neg)}) < 3:
            continue
        # margin in log-ratio space, re-checked on rounded values
        if log_ratio_dist(anchor, neg) < NEG_POS_MARGIN * log_ratio_dist(anchor, pos):
            continue
        # validity in the space the loss actually measures
        if loss_space_dist(anchor, neg) <= loss_space_dist(anchor, pos):
            continue
        return pos, neg, band

    return None


def build_triplet(rec: dict) -> dict | None:
    """
    Generate positive_number/negative_number for one source record.
    Returns None on failure — failed records are dropped and counted,
    not written with nulls (the old null rows were never fixed and ended
    up silently skipped by every downstream stage).
    """
    anchor = float(rec["number"])
    if get_magnitude(anchor) == -99:
        return None

    pair = generate_pair(anchor)
    if pair is None:
        return None

    pos, neg, band = pair
    out = dict(rec)
    out["positive_number"] = pos
    out["negative_number"] = neg
    out["neg_band"] = BAND_NAMES[band]
    return out


# ─────────────────────────────────────────────────────────────
# SAMPLING WITH MAGNITUDE TARGETS
# ─────────────────────────────────────────────────────────────

def sample_by_magnitude(records: list[dict], total: int) -> list[dict]:
    """
    Sample records from the pool to hit MAG_TARGETS distribution.
    For each magnitude bucket, compute how many records we need
    and sample (with replacement if pool is too small).
    """
    # group source records by magnitude
    pools = defaultdict(list)
    for rec in records:
        mag = get_magnitude(float(rec["number"]))
        pools[mag].append(rec)

    log.info("\n  Source pool by magnitude:")
    for mag in sorted(pools):
        upper = f"< {10**(mag+1):>10,}" if mag >= 0 else "  decimal/zero "
        log.info(f"    mag={mag:>3} {upper} : {len(pools[mag]):>7,} records")

    sampled = []
    for mag, frac in MAG_TARGETS.items():
        needed = int(total * frac)
        pool   = pools.get(mag, [])

        if not pool:
            log.warning(f"  mag={mag}: no source records available — skipping")
            continue

        if len(pool) >= needed:
            chosen = random.sample(pool, needed)
        else:
            # sample with replacement if pool is smaller than needed
            log.warning(f"  mag={mag}: only {len(pool):,} records available, "
                        f"need {needed:,} — sampling with replacement")
            chosen = random.choices(pool, k=needed)

        log.info(f"  mag={mag}: sampled {len(chosen):,} / target {needed:,}")
        sampled.extend(chosen)

    return sampled


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str, total: int, seed: int):
    random.seed(seed)

    # ── Load ──
    log.info(f"Loading {input_path} ...")
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    log.info(f"  Loaded {len(raw):,} records")

    # ── Filter negative numbers ──
    log.info("\nFiltering negative number records ...")
    clean = []
    neg_filtered   = 0
    zero_filtered  = 0
    invalid_filtered = 0

    for rec in raw:
        num = rec.get("number")
        if num is None:
            invalid_filtered += 1
            continue
        try:
            num_f = float(num)
        except (ValueError, TypeError):
            invalid_filtered += 1
            continue

        if num_f < 0:
            neg_filtered += 1
            continue

        if num_f == 0:
            zero_filtered += 1
            continue

        # check if sentence contains a leading minus before the number
        if sentence_has_negative(str(rec.get("comment", "")), num_f):
            neg_filtered += 1
            continue

        clean.append(rec)

    log.info(f"  Negative numbers filtered : {neg_filtered:,}")
    log.info(f"  Zero anchors filtered     : {zero_filtered:,}")
    log.info(f"  Invalid records filtered  : {invalid_filtered:,}")
    log.info(f"  Clean records remaining   : {len(clean):,}")

    # ── Sample by magnitude ──
    log.info(f"\nSampling {total:,} records by magnitude targets ...")
    sampled = sample_by_magnitude(clean, total)
    random.shuffle(sampled)
    log.info(f"  Total sampled: {len(sampled):,}")

    # ── Generate triplets ──
    log.info("\nGenerating positive and negative numbers ...")
    results = []
    dropped = 0

    for i, rec in enumerate(sampled):
        triplet = build_triplet(rec)
        if triplet is None:
            dropped += 1
        else:
            results.append(triplet)

        if (i + 1) % 10_000 == 0:
            log.info(f"  Processed {i+1:,} / {len(sampled):,} "
                     f"— kept={len(results):,}  dropped={dropped:,}")

    log.info(f"\n  Generation complete:")
    log.info(f"  Triplets kept   : {len(results):,} ({100*len(results)/len(sampled):.1f}%)")
    log.info(f"  Dropped (failed): {dropped:,} ({100*dropped/len(sampled):.1f}%)")

    # ── Hardness report: neg |log ratio| distribution ──
    log.info("\n── Negative Hardness Report (|log(neg/anchor)|) ──")
    neg_lr = sorted(
        log_ratio_dist(float(r["number"]), float(r["negative_number"]))
        for r in results
    )
    n = len(neg_lr)
    if n:
        for q in (10, 25, 50, 75, 90):
            v = neg_lr[min(n - 1, int(n * q / 100))]
            log.info(f"  p{q:<3}: {v:6.3f}  (ratio {math.exp(v):6.2f}x)")
        band_counts = Counter(r["neg_band"] for r in results)
        log.info("  Band distribution:")
        for name in BAND_NAMES:
            c = band_counts.get(name, 0)
            log.info(f"    {name:5s}: {c:6,} ({100*c/n:5.1f}%)")
        near = sum(1 for v in neg_lr if v < 0.182)  # within 1.2x
        log.info(f"  Negatives within 1.2x of anchor: {near:,} ({100*near/n:.1f}%) "
                 f"(was 1.4% in train_triplets_04092026)")

    # ── Magnitude distribution of output ──
    log.info("\n── Output Magnitude Distribution ──")
    mag_counts = Counter(get_magnitude(float(r["number"])) for r in results)
    total_out = len(results)
    for mag in sorted(mag_counts):
        count = mag_counts[mag]
        pct   = 100 * count / total_out
        upper = f"< {10**(mag+1):>10,}" if mag >= 0 else "  decimal/zero "
        bar   = "█" * int(pct / 2)
        log.info(f"  mag={mag:>3} {upper} : {count:6,} ({pct:5.1f}%)  {bar}")

    # ── Write ──
    random.shuffle(results)
    log.info(f"\nWriting {len(results):,} records to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Done.")

    # ── Sample output ──
    log.info("\n── Sample output records ──")
    for rec in results[:3]:
        log.info(f"  number={rec['number']}  "
                 f"positive_number={rec['positive_number']}  "
                 f"negative_number={rec['negative_number']}")
        log.info(f"  comment: {rec['comment'][:80]}")
        log.info("")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,
                        help="Path to Numeracy_600K_comment.json")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file path")
    parser.add_argument("--total",  type=int, default=100_000,
                        help="Total records to generate (default: 100000)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    run(args.input, args.output, args.total, args.seed)
