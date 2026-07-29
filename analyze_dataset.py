"""
Triplet Dataset Analysis
=========================
Analyzes training data quality across:
  1. Anchor magnitude distribution
  2. Learning signal quality (log-distance ratios)
  3. Data skew and clustering
  4. Degenerate / invalid triplets
  5. Decimal tokenization impact analysis
  6. Duplicate detection
  7. Overall health score

Usage:
    python analyze_dataset.py --input train.jsonl

Share the printed output for analysis.
"""

import json
import math
import argparse
import logging
from collections import Counter, defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

TOKENIZER_NAME = "answerdotai/ModernBERT-base"   # change if using different model


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def safe_float(x) -> float | None:
    try:
        return abs(float(x))
    except (TypeError, ValueError):
        return None


def get_magnitude(x: float) -> int:
    if x <= 0:
        return -1
    return int(math.floor(math.log10(x)))


def log1p_dist(a: float, b: float) -> float:
    return math.log1p(abs(a - b))


def count_decimal_tokens(value: float) -> dict:
    """
    Analyze how a number tokenizes at the decimal boundary.
    Returns token structure info without needing the actual tokenizer.

    For ModernBERT (BPE-based):
      - Integer part and decimal part are usually separate tokens
      - "13.5"  → ["13", ".", "5"]       — 3 tokens
      - "1.49"  → ["1", ".", "49"]       — 3 tokens
      - "1000"  → ["1000"]               — 1 token (clean integer)
      - "13"    → ["13"]                 — 1 token

    We estimate token boundary impact by checking:
      - has decimal point
      - integer part length
      - decimal part length
      - whether integer parts match between anchor/pos/neg (same prefix)
    """
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    has_decimal = "." in s
    parts = s.split(".")
    int_part = parts[0]
    dec_part = parts[1] if has_decimal else ""
    return {
        "string": s,
        "has_decimal": has_decimal,
        "int_part": int_part,
        "dec_part": dec_part,
        "int_len": len(int_part),
        "dec_len": len(dec_part),
        "estimated_tokens": 1 + (2 if has_decimal else 0),  # rough: int + "." + dec
    }


def same_integer_part(a: float, b: float) -> bool:
    """True if a and b share the same integer part — tokenizer gives no help."""
    return int(a) == int(b)


def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def subsection(title: str):
    print(f"\n  ── {title} ──")


# ─────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────

def analyze_magnitude(anchors: list[float]):
    section("1. ANCHOR MAGNITUDE DISTRIBUTION")

    total = len(anchors)
    mag_counts = Counter(get_magnitude(a) for a in anchors)
    zero_count = sum(1 for a in anchors if a == 0)

    print(f"\n  Total records : {total:,}")
    print(f"  Zero anchors  : {zero_count:,} ({100*zero_count/total:.1f}%)")
    print()
    print(f"  {'mag':>5}  {'range':>18}  {'count':>7}  {'pct':>6}  bar")
    print(f"  {'-'*5}  {'-'*18}  {'-'*7}  {'-'*6}  {'-'*20}")

    for mag in sorted(mag_counts):
        count = mag_counts[mag]
        pct   = 100 * count / total
        if mag < 0:
            label = f"{'zero / < 0.1':>18}"
        else:
            label = f"{'< ' + f'{10**(mag+1):,}':>18}"
        bar = "█" * min(int(pct / 2), 30)
        print(f"  {mag:>5}  {label}  {count:>7,}  {pct:>5.1f}%  {bar}")

    # Health check
    high_mag = sum(mag_counts.get(m, 0) for m in range(2, 7))
    high_pct = 100 * high_mag / total
    print(f"\n  Records with anchor >= 100 (mag>=2): {high_mag:,} ({high_pct:.1f}%)")
    if high_pct < 15:
        print("  ⚠️  SEVERE SKEW — less than 15% of data has anchor >= 100")
        print("     Model will not learn ordering for large numbers")
    elif high_pct < 30:
        print("  ⚠️  MODERATE SKEW — consider oversampling high magnitude records")
    else:
        print("  ✅  Magnitude distribution is reasonably balanced")


def analyze_signal(anchors, pos_nums, neg_nums):
    section("2. LEARNING SIGNAL QUALITY")

    log_pos_dists = [log1p_dist(a, p) for a, p in zip(anchors, pos_nums)]
    log_neg_dists = [log1p_dist(a, n) for a, n in zip(anchors, neg_nums)]
    ratios        = [
        n / p if p > 1e-10 else 0.0
        for p, n in zip(log_pos_dists, log_neg_dists)
    ]

    subsection("Log-distance statistics")
    print(f"  log_pos_dist — mean={np.mean(log_pos_dists):.4f}  "
          f"median={np.median(log_pos_dists):.4f}  "
          f"min={np.min(log_pos_dists):.4f}  "
          f"max={np.max(log_pos_dists):.4f}")
    print(f"  log_neg_dist — mean={np.mean(log_neg_dists):.4f}  "
          f"median={np.median(log_neg_dists):.4f}  "
          f"min={np.min(log_neg_dists):.4f}  "
          f"max={np.max(log_neg_dists):.4f}")
    print(f"  ratio neg/pos — mean={np.mean(ratios):.2f}  "
          f"median={np.median(ratios):.2f}  "
          f"min={np.min(ratios):.2f}  "
          f"max={np.max(ratios):.2f}")

    subsection("Ratio distribution (neg_log_dist / pos_log_dist)")
    total = len(ratios)
    buckets = [
        (0,    1.5,  "< 1.5   almost no signal   ❌"),
        (1.5,  2.0,  "1.5–2   weak               ⚠️"),
        (2.0,  3.0,  "2–3     moderate            ✅"),
        (3.0,  5.0,  "3–5     good               ✅"),
        (5.0,  10.0, "5–10    strong             ✅"),
        (10.0, 1e9,  "> 10    very strong        ✅"),
    ]
    for lo, hi, label in buckets:
        count = sum(1 for r in ratios if lo <= r < hi)
        pct   = 100 * count / total
        print(f"  {label:40s}: {count:6,} ({pct:5.1f}%)")

    weak = sum(1 for r in ratios if r < 2.0)
    weak_pct = 100 * weak / total
    print(f"\n  Weak/no signal (ratio < 2.0): {weak:,} ({weak_pct:.1f}%)")
    if weak_pct > 50:
        print("  ❌  CRITICAL — majority of triplets provide no useful gradient")
    elif weak_pct > 25:
        print("  ⚠️  HIGH — consider filtering ratio < 1.5 and regenerating negatives")
    else:
        print("  ✅  Signal quality is acceptable")

    subsection("Tiny positive distances (below learning resolution)")
    for thresh in [0.01, 0.05, 0.10, 0.20]:
        count = sum(1 for d in log_pos_dists if d < thresh)
        pct   = 100 * count / total
        flag  = "❌" if pct > 10 else ("⚠️" if pct > 5 else "✅")
        print(f"  log_pos_dist < {thresh:.2f}: {count:6,} ({pct:5.1f}%) {flag}")

    return log_pos_dists, log_neg_dists, ratios


def analyze_invalids(anchors, pos_nums, neg_nums, log_pos_dists, log_neg_dists):
    section("3. INVALID / DEGENERATE TRIPLETS")

    total = len(anchors)

    # Log-space invalid (pos farther than neg)
    log_invalid = sum(
        1 for lp, ln in zip(log_pos_dists, log_neg_dists) if lp >= ln
    )
    # Absolute invalid
    abs_invalid = sum(
        1 for a, p, n in zip(anchors, pos_nums, neg_nums)
        if abs(a - p) >= abs(a - n)
    )
    # anchor == pos
    anchor_eq_pos = sum(
        1 for a, p in zip(anchors, pos_nums) if abs(a - p) < 1e-9
    )
    # anchor == neg
    anchor_eq_neg = sum(
        1 for a, n in zip(anchors, neg_nums) if abs(a - n) < 1e-9
    )
    # pos == neg
    pos_eq_neg = sum(
        1 for p, n in zip(pos_nums, neg_nums) if abs(p - n) < 1e-9
    )
    # zero anchor with non-trivial pos/neg
    zero_anchor = sum(1 for a in anchors if a == 0)

    print(f"\n  Log-space invalid (pos farther than neg) : {log_invalid:,} ({100*log_invalid/total:.1f}%)")
    print(f"  Absolute invalid (pos farther than neg)  : {abs_invalid:,} ({100*abs_invalid/total:.1f}%)")
    print(f"  anchor == pos (zero distance)            : {anchor_eq_pos:,} ({100*anchor_eq_pos/total:.1f}%)")
    print(f"  anchor == neg                            : {anchor_eq_neg:,} ({100*anchor_eq_neg/total:.1f}%)")
    print(f"  pos == neg                               : {pos_eq_neg:,} ({100*pos_eq_neg/total:.1f}%)")
    print(f"  zero anchors                             : {zero_anchor:,} ({100*zero_anchor/total:.1f}%)")

    total_bad = log_invalid + anchor_eq_pos + pos_eq_neg
    if total_bad > 0:
        print(f"\n  ❌  {total_bad:,} records should be filtered before training")
    else:
        print(f"\n  ✅  No degenerate triplets found")


def analyze_duplicates(anchors, pos_nums, neg_nums):
    section("4. DUPLICATE ANALYSIS")

    total = len(anchors)
    triple_counter = Counter(
        (round(a, 4), round(p, 4), round(n, 4))
        for a, p, n in zip(anchors, pos_nums, neg_nums)
    )
    anchor_counter = Counter(round(a, 4) for a in anchors)

    dupes         = {k: v for k, v in triple_counter.items() if v > 1}
    excess        = sum(v - 1 for v in dupes.values())
    dup_rate      = 100 * excess / total if total > 0 else 0

    print(f"\n  Unique numeric triples     : {len(triple_counter):,}")
    print(f"  Triples with duplicates    : {len(dupes):,}")
    print(f"  Excess records (removable) : {excess:,} ({dup_rate:.1f}%)")
    print(f"  Unique anchor values       : {len(anchor_counter):,}")

    subsection("Anchor clustering (anchors with most triplets)")
    for anchor, count in sorted(anchor_counter.items(), key=lambda x: -x[1])[:10]:
        pct = 100 * count / total
        print(f"  anchor={anchor:<12} → {count:5,} triplets ({pct:.1f}%)")

    subsection("Top duplicated numeric triples")
    if dupes:
        for triple, count in sorted(dupes.items(), key=lambda x: -x[1])[:10]:
            print(f"  {triple} → {count}×")
    else:
        print("  No duplicates found ✅")

    if dup_rate > 20:
        print(f"\n  ❌  HIGH duplication — {dup_rate:.1f}% of records are redundant")
    elif dup_rate > 10:
        print(f"\n  ⚠️  MODERATE duplication — consider capping at 3-5 per triple")
    else:
        print(f"\n  ✅  Duplication rate is acceptable")


def analyze_decimal_tokenization(anchors, pos_nums, neg_nums):
    section("5. DECIMAL TOKENIZATION IMPACT")

    total = len(anchors)

    # Count decimal numbers
    anchor_decimals = sum(1 for a in anchors if a != int(a))
    pos_decimals    = sum(1 for p in pos_nums if p != int(p))
    neg_decimals    = sum(1 for n in neg_nums if n != int(n))

    print(f"\n  Decimal anchors   : {anchor_decimals:,} ({100*anchor_decimals/total:.1f}%)")
    print(f"  Decimal positives : {pos_decimals:,} ({100*pos_decimals/total:.1f}%)")
    print(f"  Decimal negatives : {neg_decimals:,} ({100*neg_decimals/total:.1f}%)")

    subsection("Same integer part (tokenizer gives no proximity signal)")
    # Cases where anchor and pos share the same integer part
    # → tokenizer can't help, model must rely purely on learned embeddings
    ap_same_int = sum(
        1 for a, p in zip(anchors, pos_nums)
        if same_integer_part(a, p)
    )
    an_same_int = sum(
        1 for a, n in zip(anchors, neg_nums)
        if same_integer_part(a, n)
    )
    all_same_int = sum(
        1 for a, p, n in zip(anchors, pos_nums, neg_nums)
        if same_integer_part(a, p) and same_integer_part(a, n)
    )

    print(f"  anchor & pos share integer part        : {ap_same_int:,} ({100*ap_same_int/total:.1f}%)")
    print(f"  anchor & neg share integer part        : {an_same_int:,} ({100*an_same_int/total:.1f}%)")
    print(f"  all three share same integer part      : {all_same_int:,} ({100*all_same_int/total:.1f}%)")

    if all_same_int / total > 0.3:
        print("  ❌  HIGH — model must distinguish purely by decimal tokens for 30%+ of data")
        print("     These are the hardest cases and likely failing in eval")

    subsection("Estimated token count distribution for anchors")
    token_counts = Counter()
    for a in anchors:
        info = count_decimal_tokens(a)
        token_counts[info["estimated_tokens"]] += 1

    for tc, count in sorted(token_counts.items()):
        pct = 100 * count / total
        label = {1: "integer (e.g. 13, 500)", 3: "decimal (e.g. 13.5, 0.04)"}.get(tc, f"{tc} tokens")
        print(f"  ~{tc} tokens [{label:30s}]: {count:6,} ({pct:5.1f}%)")

    subsection("Hardest tokenization cases — same int, different decimal")
    hard_cases = []
    for a, p, n in zip(anchors, pos_nums, neg_nums):
        if same_integer_part(a, p) and same_integer_part(a, n) and a != p and a != n:
            pos_dec_diff = abs((a - int(a)) - (p - int(p)))
            neg_dec_diff = abs((a - int(a)) - (n - int(n)))
            hard_cases.append({
                "anchor": a, "pos": p, "neg": n,
                "pos_dec_diff": pos_dec_diff,
                "neg_dec_diff": neg_dec_diff,
                "ratio": neg_dec_diff / (pos_dec_diff + 1e-10),
            })

    print(f"  Total hard cases (same int, diff decimal): {len(hard_cases):,} ({100*len(hard_cases)/total:.1f}%)")
    if hard_cases:
        low_ratio = sum(1 for h in hard_cases if h["ratio"] < 2.0)
        print(f"  Of these, ratio < 2.0 (very hard):        {low_ratio:,} ({100*low_ratio/len(hard_cases):.1f}%)")
        print("  Sample hard cases:")
        for h in sorted(hard_cases, key=lambda x: x["ratio"])[:5]:
            print(f"    anchor={h['anchor']:.4f}  pos={h['pos']:.4f}  neg={h['neg']:.4f}  "
                  f"ratio={h['ratio']:.2f}")

    subsection("Decimal precision distribution")
    def decimal_places(x: float) -> int:
        s = f"{x:.10f}".rstrip("0")
        if "." in s:
            return len(s.split(".")[1])
        return 0

    dp_counter = Counter(decimal_places(a) for a in anchors)
    for dp, count in sorted(dp_counter.items()):
        pct = 100 * count / total
        print(f"  {dp} decimal places: {count:6,} ({pct:5.1f}%)")


def analyze_composition(anchors, pos_nums, neg_nums, ratios, log_pos_dists):
    section("6. RECOMMENDED DATASET COMPOSITION vs ACTUAL")

    total = len(anchors)

    print("""
  Ideal composition for learning numeric ordering:
  ┌─────────────────────────────┬──────────┬─────────────────────────────┐
  │ Category                    │ Target % │ Rationale                   │
  ├─────────────────────────────┼──────────┼─────────────────────────────┤
  │ Magnitude 0  (1–9)          │   15%    │ Basic ordering              │
  │ Magnitude 1  (10–99)        │   20%    │ Core range                  │
  │ Magnitude 2  (100–999)      │   20%    │ Mid range                   │
  │ Magnitude 3  (1000–9999)    │   15%    │ Large numbers               │
  │ Magnitude 4+ (10000+)       │   10%    │ Very large                  │
  │ Decimal < 1  (0.01–0.99)    │   10%    │ Fractional ordering         │
  │ Zero anchors                │    5%    │ Edge case only              │
  │ Cross-magnitude triplets    │   5%+    │ Breaks token pattern bias   │
  └─────────────────────────────┴──────────┴─────────────────────────────┘

  Signal quality targets:
    ratio < 1.5  : < 10%   (currently these are wasted compute)
    ratio 2–5    : > 40%   (core learning signal)
    ratio > 5    : > 20%   (hard cases, high value)
    log_pos_dist : > 0.05  (below this tokenizer gives no help)
""")

    subsection("Actual vs target")
    mag_counts = Counter(get_magnitude(a) for a in anchors)
    zero_count = sum(1 for a in anchors if a == 0)

    actual_targets = {
        "mag 0 (1–9)":      mag_counts.get(0, 0),
        "mag 1 (10–99)":    mag_counts.get(1, 0),
        "mag 2 (100–999)":  mag_counts.get(2, 0),
        "mag 3 (1k–9k)":    mag_counts.get(3, 0),
        "mag 4+ (10k+)":    sum(mag_counts.get(m, 0) for m in range(4, 10)),
        "decimal < 1":      sum(mag_counts.get(m, 0) for m in range(-4, 0) if m != -1)
                            + sum(1 for a in anchors if 0 < a < 1),
        "zero anchors":     zero_count,
    }
    targets = {
        "mag 0 (1–9)":     15,
        "mag 1 (10–99)":   20,
        "mag 2 (100–999)": 20,
        "mag 3 (1k–9k)":   15,
        "mag 4+ (10k+)":   10,
        "decimal < 1":     10,
        "zero anchors":     5,
    }

    for label, count in actual_targets.items():
        actual_pct = 100 * count / total
        target_pct = targets.get(label, 0)
        gap        = actual_pct - target_pct
        flag       = "✅" if abs(gap) < 5 else ("⚠️" if abs(gap) < 15 else "❌")
        print(f"  {label:<20} actual={actual_pct:5.1f}%  target={target_pct:5.1f}%  gap={gap:+.1f}%  {flag}")

    signal_ok  = sum(1 for r in ratios if r >= 2.0)
    signal_pct = 100 * signal_ok / total
    target_sig = 70
    gap        = signal_pct - target_sig
    flag       = "✅" if gap >= 0 else ("⚠️" if gap > -20 else "❌")
    print(f"  {'ratio >= 2.0':<20} actual={signal_pct:5.1f}%  target={target_sig:5.1f}%  gap={gap:+.1f}%  {flag}")

    tiny_pos   = sum(1 for d in log_pos_dists if d < 0.05)
    tiny_pct   = 100 * tiny_pos / total
    target_tiny = 10
    gap         = tiny_pct - target_tiny
    flag        = "✅" if gap <= 0 else ("⚠️" if gap < 10 else "❌")
    print(f"  {'log_pos < 0.05':<20} actual={tiny_pct:5.1f}%  target=<{target_tiny:4.1f}%  gap={gap:+.1f}%  {flag}")


def overall_health(anchors, ratios, log_pos_dists, log_neg_dists):
    section("7. OVERALL HEALTH SCORE")

    total     = len(anchors)
    scores    = {}

    # Magnitude balance
    mag_counts = Counter(get_magnitude(a) for a in anchors)
    high_mag   = sum(mag_counts.get(m, 0) for m in range(2, 10))
    scores["magnitude_balance"] = min(100, int(100 * high_mag / total / 0.40))

    # Signal quality
    good_signal = sum(1 for r in ratios if r >= 2.0)
    scores["signal_quality"] = min(100, int(100 * good_signal / total / 0.70))

    # Low noise (few tiny pos distances)
    tiny = sum(1 for d in log_pos_dists if d < 0.05)
    scores["low_noise"] = max(0, 100 - int(100 * tiny / total / 0.10))

    # No invalids
    invalid = sum(1 for lp, ln in zip(log_pos_dists, log_neg_dists) if lp >= ln)
    scores["no_invalids"] = 100 if invalid == 0 else max(0, 100 - int(100 * invalid / total / 0.01))

    # Zero anchor proportion
    zero_count = sum(1 for a in anchors if a == 0)
    zero_pct   = zero_count / total
    scores["zero_balance"] = 100 if zero_pct <= 0.05 else max(0, int(100 * (1 - zero_pct / 0.90)))

    overall = int(np.mean(list(scores.values())))

    print()
    for metric, score in scores.items():
        bar  = "█" * (score // 5)
        flag = "✅" if score >= 70 else ("⚠️" if score >= 40 else "❌")
        print(f"  {metric:<25} {score:>3}/100  {bar:<20} {flag}")

    print(f"\n  {'OVERALL HEALTH':.<25} {overall:>3}/100  ", end="")
    if overall >= 70:
        print("✅  Dataset is healthy for training")
    elif overall >= 40:
        print("⚠️  Dataset needs improvement before training")
    else:
        print("❌  Dataset is not suitable for training — significant fixes needed")

    return overall


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run(input_path: str):
    print(f"\n{'#'*65}")
    print(f"  TRIPLET DATASET ANALYSIS")
    print(f"  File: {input_path}")
    print(f"{'#'*65}")

    # Load
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"\n  Loaded {len(records):,} records")

    # Parse numbers
    anchors, pos_nums, neg_nums = [], [], []
    parse_errors = 0
    for r in records:
        a = safe_float(r.get("number"))
        p = safe_float(r.get("positive_number"))
        n = safe_float(r.get("negative_number"))
        if a is None or p is None or n is None:
            parse_errors += 1
            continue
        anchors.append(a)
        pos_nums.append(p)
        neg_nums.append(n)

    if parse_errors:
        print(f"  ⚠️  Parse errors (skipped): {parse_errors:,}")

    # Run analyses
    analyze_magnitude(anchors)
    log_pos_dists, log_neg_dists, ratios = analyze_signal(anchors, pos_nums, neg_nums)
    analyze_invalids(anchors, pos_nums, neg_nums, log_pos_dists, log_neg_dists)
    analyze_duplicates(anchors, pos_nums, neg_nums)
    analyze_decimal_tokenization(anchors, pos_nums, neg_nums)
    analyze_composition(anchors, pos_nums, neg_nums, ratios, log_pos_dists)
    overall_health(anchors, ratios, log_pos_dists, log_neg_dists)

    print(f"\n{'#'*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSONL dataset")
    args = parser.parse_args()
    run(args.input)
