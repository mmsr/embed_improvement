"""
Magnitude Relationship Analysis
=================================
Analyzes how anchor, positive, and negative numbers relate
in terms of magnitude — revealing training bias and test coverage gaps.

Key questions answered:
  1. Do pos/neg stay in the same magnitude bucket as anchor?
  2. How far apart are pos and neg from anchor numerically?
  3. Does intra-magnitude signal have enough ratio to train on?
  4. Are test-range anchors covered in training data?
  5. What is the per-magnitude breakdown of pos/neg placement?

Usage:
    python analyze_magnitude_relationships.py --input train.jsonl
"""

import json
import math
import argparse
import logging
from collections import Counter, defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")


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
        return -99
    return int(math.floor(math.log10(x)))


def log1p_dist(a: float, b: float) -> float:
    return math.log1p(abs(a - b))


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def subsection(title):
    print(f"\n  ── {title} ──")


# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

def run(input_path: str):
    print(f"\n{'#'*65}")
    print(f"  MAGNITUDE RELATIONSHIP ANALYSIS")
    print(f"  File: {input_path}")
    print(f"{'#'*65}")

    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    anchors, pos_nums, neg_nums = [], [], []
    for r in records:
        a = safe_float(r.get("number"))
        p = safe_float(r.get("positive_number"))
        n = safe_float(r.get("negative_number"))
        if a is not None and p is not None and n is not None:
            anchors.append(a)
            pos_nums.append(p)
            neg_nums.append(n)

    total = len(anchors)
    print(f"\n  Loaded {total:,} valid records")

    # ──────────────────────────────────────────
    # 1. Same vs different magnitude
    # ──────────────────────────────────────────
    section("1. DO POS/NEG STAY IN SAME MAGNITUDE AS ANCHOR?")

    ap_same  = sum(1 for a,p in zip(anchors,pos_nums) if a>0 and p>0 and get_magnitude(a)==get_magnitude(p))
    an_same  = sum(1 for a,n in zip(anchors,neg_nums) if a>0 and n>0 and get_magnitude(a)==get_magnitude(n))
    all_same = sum(1 for a,p,n in zip(anchors,pos_nums,neg_nums)
                   if a>0 and p>0 and n>0
                   and get_magnitude(a)==get_magnitude(p)==get_magnitude(n))
    neg_cross = total - an_same
    pos_cross = total - ap_same

    print(f"\n  anchor & pos same magnitude      : {ap_same:,}  ({100*ap_same/total:.1f}%)")
    print(f"  anchor & neg same magnitude      : {an_same:,}  ({100*an_same/total:.1f}%)")
    print(f"  all three same magnitude         : {all_same:,}  ({100*all_same/total:.1f}%)")
    print(f"  neg crosses magnitude boundary   : {neg_cross:,}  ({100*neg_cross/total:.1f}%)")
    print(f"  pos crosses magnitude boundary   : {pos_cross:,}  ({100*pos_cross/total:.1f}%)")

    print()
    if 100*all_same/total > 60:
        print("  ⚠️  > 60% of triplets stay within same magnitude")
        print("     Model may not generalise cross-magnitude ordering")
    elif 100*neg_cross/total < 30:
        print("  ⚠️  Negatives rarely cross magnitude — consider harder negatives")
    else:
        print("  ✅  Good mix of same and cross-magnitude triplets")

    # ──────────────────────────────────────────
    # 2. Magnitude delta distribution
    # ──────────────────────────────────────────
    section("2. MAGNITUDE DELTA: ANCHOR → NEG")
    print("  (how many magnitude steps away is the negative from the anchor?)")

    mag_deltas = [
        get_magnitude(n) - get_magnitude(a)
        for a, n in zip(anchors, neg_nums)
        if a > 0 and n > 0
    ]
    delta_counter = Counter(mag_deltas)

    for delta in sorted(delta_counter):
        count = delta_counter[delta]
        pct   = 100 * count / total
        if delta == 0:
            label = "same magnitude as anchor"
        elif delta > 0:
            label = f"{delta} magnitude(s) ABOVE anchor"
        else:
            label = f"{abs(delta)} magnitude(s) BELOW anchor"
        bar = "█" * int(pct / 2)
        flag = "✅" if delta != 0 else "⚠️"
        print(f"  delta={delta:>+3}  {label:<35}: {count:6,} ({pct:5.1f}%)  {bar}  {flag}")

    # ──────────────────────────────────────────
    # 3. Numeric value ratio
    # ──────────────────────────────────────────
    section("3. NUMERIC VALUE RATIO (pos/anchor and neg/anchor)")
    print("  A ratio of 1.0 means they are identical.")
    print("  A ratio of 2.0 means one is double the other.")
    print("  A ratio of 10.0 means one is 10× the other.")

    pos_ratios = [max(a,p)/min(a,p) for a,p in zip(anchors,pos_nums) if a>0 and p>0]
    neg_ratios = [max(a,n)/min(a,n) for a,n in zip(anchors,neg_nums) if a>0 and n>0]

    print(f"\n  pos/anchor ratio — "
          f"mean={np.mean(pos_ratios):.2f}  "
          f"median={np.median(pos_ratios):.2f}  "
          f"p10={np.percentile(pos_ratios,10):.2f}  "
          f"p90={np.percentile(pos_ratios,90):.2f}  "
          f"max={np.max(pos_ratios):.1f}")
    print(f"  neg/anchor ratio — "
          f"mean={np.mean(neg_ratios):.2f}  "
          f"median={np.median(neg_ratios):.2f}  "
          f"p10={np.percentile(neg_ratios,10):.2f}  "
          f"p90={np.percentile(neg_ratios,90):.2f}  "
          f"max={np.max(neg_ratios):.1f}")

    subsection("Pos ratio buckets")
    for lo, hi, label in [(1,1.05,"< 1.05 (trivially close)"),
                           (1.05,1.15,"1.05–1.15 (very close)"),
                           (1.15,1.5,"1.15–1.5 (close)"),
                           (1.5,2,"1.5–2.0 (moderate)"),
                           (2,5,"2–5 (far)"),
                           (5,1e9,"> 5 (very far)")]:
        count = sum(1 for r in pos_ratios if lo <= r < hi)
        print(f"  {label:<30}: {count:6,} ({100*count/total:5.1f}%)")

    subsection("Neg ratio buckets")
    for lo, hi, label in [(1,1.5,"< 1.5 (barely farther than pos)"),
                           (1.5,2,"1.5–2.0"),
                           (2,5,"2–5"),
                           (5,10,"5–10"),
                           (10,1e9,"> 10 (very far)")]:
        count = sum(1 for r in neg_ratios if lo <= r < hi)
        print(f"  {label:<35}: {count:6,} ({100*count/total:5.1f}%)")

    # ──────────────────────────────────────────
    # 4. Per-magnitude breakdown
    # ──────────────────────────────────────────
    section("4. PER-MAGNITUDE BREAKDOWN")
    print(f"\n  {'mag':<6} {'count':>7} {'ap_same%':>9} {'an_same%':>9} "
          f"{'neg_cross%':>11} {'med_pos_r':>10} {'med_neg_r':>10} {'med_ratio':>10}")
    print(f"  {'-'*6} {'-'*7} {'-'*9} {'-'*9} {'-'*11} {'-'*10} {'-'*10} {'-'*10}")

    groups = defaultdict(lambda: {
        "ap_same": 0, "an_same": 0, "neg_cross": 0,
        "pos_ratios": [], "neg_ratios": [], "signal_ratios": [], "count": 0
    })

    for a, p, n in zip(anchors, pos_nums, neg_nums):
        if a <= 0:
            continue
        mag = get_magnitude(a)
        g   = groups[mag]
        g["count"] += 1
        if p > 0:
            if get_magnitude(a) == get_magnitude(p):
                g["ap_same"] += 1
            g["pos_ratios"].append(max(a,p)/min(a,p))
        if n > 0:
            if get_magnitude(a) == get_magnitude(n):
                g["an_same"] += 1
            else:
                g["neg_cross"] += 1
            g["neg_ratios"].append(max(a,n)/min(a,n))
        if p > 0 and n > 0:
            lp = log1p_dist(a, p)
            ln = log1p_dist(a, n)
            if lp > 1e-10:
                g["signal_ratios"].append(ln / lp)

    for mag in sorted(groups):
        g   = groups[mag]
        cnt = g["count"]
        if cnt == 0:
            continue
        ap_s  = 100 * g["ap_same"]   / cnt
        an_s  = 100 * g["an_same"]   / cnt
        nc    = 100 * g["neg_cross"] / cnt
        med_p = np.median(g["pos_ratios"])     if g["pos_ratios"]     else 0
        med_n = np.median(g["neg_ratios"])     if g["neg_ratios"]     else 0
        med_r = np.median(g["signal_ratios"])  if g["signal_ratios"]  else 0
        flag  = "✅" if med_r >= 2.0 else ("⚠️" if med_r >= 1.5 else "❌")
        label = f"< {10**(mag+1):>8,}" if mag >= 0 else "  < 0.1  "
        print(f"  {mag:<6} {cnt:>7,} {ap_s:>8.1f}% {an_s:>8.1f}% "
              f"{nc:>10.1f}% {med_p:>10.2f} {med_n:>10.2f} {med_r:>9.2f}  {flag}")

    print(f"\n  med_ratio = median(log_neg_dist / log_pos_dist)")
    print(f"  ✅ >= 2.0 (good signal)  ⚠️ 1.5–2.0 (weak)  ❌ < 1.5 (no signal)")

    # ──────────────────────────────────────────
    # 5. Intra-magnitude triplets
    # ──────────────────────────────────────────
    section("5. INTRA-MAGNITUDE TRIPLETS (hardest for model)")
    print("  All three numbers in same magnitude bucket.")
    print("  Token signal is ambiguous — model must rely purely on learned embeddings.")

    intra = [
        (a, p, n) for a, p, n in zip(anchors, pos_nums, neg_nums)
        if a>0 and p>0 and n>0
        and get_magnitude(a)==get_magnitude(p)==get_magnitude(n)
    ]

    if intra:
        intra_ratios = [
            log1p_dist(a,n)/(log1p_dist(a,p)+1e-10)
            for a,p,n in intra
        ]
        weak   = sum(1 for r in intra_ratios if r < 1.5)
        mod    = sum(1 for r in intra_ratios if 1.5 <= r < 2.0)
        good   = sum(1 for r in intra_ratios if r >= 2.0)
        n_intra = len(intra)

        print(f"\n  Total intra-magnitude triplets   : {n_intra:,} ({100*n_intra/total:.1f}%)")
        print(f"  ratio < 1.5  (no signal)         : {weak:,} ({100*weak/n_intra:.1f}%)")
        print(f"  ratio 1.5–2  (weak signal)        : {mod:,} ({100*mod/n_intra:.1f}%)")
        print(f"  ratio >= 2   (usable signal)      : {good:,} ({100*good/n_intra:.1f}%)")
        print(f"  Median signal ratio               : {np.median(intra_ratios):.2f}")

        if np.median(intra_ratios) < 2.0:
            print("\n  ❌  Intra-magnitude triplets have weak median signal.")
            print("     Model will struggle with fine-grained ordering within")
            print("     same order of magnitude — exactly your failing test cases.")
        else:
            print("\n  ✅  Intra-magnitude signal is acceptable")

        # Per-magnitude intra breakdown
        subsection("Intra-magnitude signal by bucket")
        intra_by_mag = defaultdict(list)
        for a, p, n in intra:
            lp = log1p_dist(a, p)
            ln = log1p_dist(a, n)
            if lp > 1e-10:
                intra_by_mag[get_magnitude(a)].append(ln/lp)

        print(f"  {'mag':<6} {'count':>7} {'weak%':>7} {'med_ratio':>10} status")
        for mag in sorted(intra_by_mag):
            ratios_m = intra_by_mag[mag]
            weak_m   = sum(1 for r in ratios_m if r < 1.5)
            med_r    = np.median(ratios_m)
            flag     = "✅" if med_r >= 2.0 else ("⚠️" if med_r >= 1.5 else "❌")
            print(f"  {mag:<6} {len(ratios_m):>7,} {100*weak_m/len(ratios_m):>6.1f}% "
                  f"{med_r:>10.2f}  {flag}")

    # ──────────────────────────────────────────
    # 6. Test probe coverage
    # ──────────────────────────────────────────
    section("6. TRAINING COVERAGE FOR YOUR TEST PROBE VALUES")
    print("  Checking triplets where anchor is within 20% of each test value")

    test_values = [13, 100, 321, 345, 500, 1000]
    for tv in test_values:
        nearby = [
            (a, p, n) for a, p, n in zip(anchors, pos_nums, neg_nums)
            if a > 0 and abs(a - tv) / tv < 0.20
        ]
        if not nearby:
            print(f"\n  anchor≈{tv:<6}: ❌  NO training triplets within 20% of this value")
            continue

        nearby_ratios = [
            log1p_dist(a,n)/(log1p_dist(a,p)+1e-10)
            for a,p,n in nearby
        ]
        intra_nearby = [
            (a,p,n) for a,p,n in nearby
            if get_magnitude(a)==get_magnitude(p)==get_magnitude(n)
        ]
        cross_nearby = len(nearby) - len(intra_nearby)
        med_r = np.median(nearby_ratios)
        flag  = "✅" if med_r >= 2.0 else ("⚠️" if med_r >= 1.5 else "❌")

        print(f"\n  anchor≈{tv:<6}: {len(nearby):>5,} triplets  "
              f"med_ratio={med_r:.2f}  "
              f"intra={len(intra_nearby):,}  cross={cross_nearby:,}  {flag}")

        # Sample some triplets
        for a, p, n in nearby[:3]:
            r = log1p_dist(a,n)/(log1p_dist(a,p)+1e-10)
            print(f"    anchor={a:<10.3f} pos={p:<10.3f} neg={n:<10.3f} ratio={r:.2f}")

    # ──────────────────────────────────────────
    # 7. Summary diagnosis
    # ──────────────────────────────────────────
    section("7. DIAGNOSIS SUMMARY")

    all_signal = [
        log1p_dist(a,n)/(log1p_dist(a,p)+1e-10)
        for a,p,n in zip(anchors,pos_nums,neg_nums)
        if log1p_dist(a,p) > 1e-10
    ]

    good_signal_pct = 100*sum(1 for r in all_signal if r >= 2.0)/total
    intra_pct       = 100*len(intra)/total if intra else 0
    cross_neg_pct   = 100*neg_cross/total

    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │ Metric                          │ Value   │ Target  │   │
  ├─────────────────────────────────┼─────────┼─────────┼───┤
  │ Good signal (ratio >= 2.0)      │ {good_signal_pct:>5.1f}%  │  > 70%  │ {'✅' if good_signal_pct>=70 else '❌'}  │
  │ Intra-magnitude triplets        │ {intra_pct:>5.1f}%  │  < 40%  │ {'✅' if intra_pct<=40 else '⚠️'}  │
  │ Neg crosses magnitude           │ {cross_neg_pct:>5.1f}%  │  > 30%  │ {'✅' if cross_neg_pct>=30 else '❌'}  │
  │ Intra med signal ratio          │ {np.median(intra_ratios) if intra else 0:>5.2f}   │  > 2.0  │ {'✅' if intra and np.median(intra_ratios)>=2.0 else '❌'}  │
  └─────────────────────────────────┴─────────┴─────────┴───┘
""")

    print(f"{'#'*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    run(args.input)
