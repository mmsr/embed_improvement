"""
Group-aware train/val/test split for the graded sentence dataset
=================================================================
Fixes three leakage risks in a naive split:

1. sample_by_magnitude oversamples scarce magnitude buckets with replacement,
   so the same source record (same comment) can appear up to ~34 times with
   different generated pos/neg numbers. All copies of a comment are kept in
   the same split (grouped split), never straddling train/test.
2. ~3.3% of records share their anchor comment with the OLD test/val files
   (test_*_extracted.jsonl). Those are excluded from every new split so the
   old test sets remain uncontaminated for old-vs-new model comparisons.
3. Writes a val/test drawn from the SAME graded distribution as train, so
   validation is sensitive to near-tie ordering (the old val.jsonl negatives
   are 50x away and can't measure it).

Usage:
    python split_dataset.py \
        --input data/train_sentences_graded.jsonl \
        --exclude data/test_same_extracted.jsonl data/test_generic_extracted.jsonl \
                  data/test_sub_extracted.jsonl data/val_extracted.jsonl \
        --out-prefix data/graded \
        [--val-ratio 0.10] [--test-ratio 0.10] [--seed 42]

Outputs: <out-prefix>_train.jsonl, <out-prefix>_val.jsonl, <out-prefix>_test.jsonl
"""

import argparse
import json
import logging
import random
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_exclusion_comments(paths: list[str]) -> set[str]:
    excluded = set()
    for path in paths:
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                text = r.get("comment") or r.get("anchor")
                if text:
                    excluded.add(text)
                    n += 1
        log.info(f"  exclusion source {path}: {n:,} comments")
    return excluded


def run(input_path: str, exclude_paths: list[str], out_prefix: str,
        val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)

    log.info("Loading exclusion comments (old test/val anchors) ...")
    excluded_comments = load_exclusion_comments(exclude_paths)
    log.info(f"  total unique excluded comments: {len(excluded_comments):,}")

    log.info(f"\nLoading {input_path} ...")
    groups: dict[str, list[dict]] = defaultdict(list)
    total = dropped_contaminated = 0
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            if rec["comment"] in excluded_comments:
                dropped_contaminated += 1
                continue
            groups[rec["comment"]].append(rec)

    log.info(f"  records loaded          : {total:,}")
    log.info(f"  dropped (in old test/val): {dropped_contaminated:,} "
             f"({100 * dropped_contaminated / total:.1f}%)")
    log.info(f"  unique comment groups   : {len(groups):,}")
    multi = sum(1 for g in groups.values() if len(g) > 1)
    log.info(f"  groups with >1 record   : {multi:,} (oversampled — kept whole in one split)")

    # ── grouped split ──
    keys = sorted(groups)          # deterministic base order
    rng.shuffle(keys)
    n_groups = len(keys)
    n_test = int(n_groups * test_ratio)
    n_val = int(n_groups * val_ratio)

    split_keys = {
        "test": keys[:n_test],
        "val": keys[n_test:n_test + n_val],
        "train": keys[n_test + n_val:],
    }

    for split, skeys in split_keys.items():
        records = [rec for k in skeys for rec in groups[k]]
        rng.shuffle(records)
        out_path = f"{out_prefix}_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        bands = Counter(r.get("neg_band", "?") for r in records)
        band_str = "  ".join(f"{b}={c:,}" for b, c in sorted(bands.items()))
        log.info(f"  {split:5s}: {len(records):6,} records "
                 f"({len(skeys):,} groups) → {out_path}   bands: {band_str}")

    # ── verify no comment straddles splits ──
    seen: dict[str, str] = {}
    leaks = 0
    for split, skeys in split_keys.items():
        for k in skeys:
            if k in seen and seen[k] != split:
                leaks += 1
            seen[k] = split
    log.info(f"\n  cross-split comment leaks: {leaks} (must be 0)")
    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="old test/val JSONL files whose anchor comments must not appear in any new split")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.input, args.exclude, args.out_prefix, args.val_ratio, args.test_ratio, args.seed)
