"""
Build Positive/Negative Sentences from Triplet JSONL
======================================================
Reads train_triplets_04092026.jsonl, skips records with missing
positive_number or negative_number, copies comment into positive
and negative fields, then substitutes the anchor number with
positive_number and negative_number respectively.

Usage:
    python build_sentences.py \
        --input  data/train_triplets_04092026.jsonl \
        --output data/train_sentences.jsonl
"""

import re
import json
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

NUMBER_RE = re.compile(r'\d[\d,]*(?:\.\d+)?')


def format_number(x: float) -> str:
    """Format cleanly — integer if whole, else strip trailing zeros."""
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def replace_at_offset(sentence: str, offset: int, length: int,
                      old_value: float, new_value: float) -> str | None:
    """
    Replace the exact anchor span using the source dataset's offset/length
    fields. Verifies the span actually parses to old_value first; returns
    None if it doesn't (caller falls back to string matching).

    This fixes the wrong-occurrence bug in the string-matching path: e.g. in
    "…ENDED JUNE 24.2, 2014 … TO $24.2 MILLION" the first "24.2" is a date
    fragment, and first-match replacement mutated the date instead of the
    dollar amount (visible throughout train_extracted_improved*.jsonl).
    """
    # The source's length field often excludes trailing digits ("5.5" inside
    # "5.50"), which would leave a stray digit glued to the replacement
    # ("4.86" + "0" -> "4.860", or "1737" + "0" -> "17370"). Extend the span
    # to the end of the full number token before replacing.
    end = offset + length
    while end < len(sentence):
        ch = sentence[end]
        if ch.isdigit():
            end += 1
        elif ch in ".," and end + 1 < len(sentence) and sentence[end + 1].isdigit():
            end += 2
        else:
            break

    span = sentence[offset:end]
    try:
        if abs(float(span.replace(",", "")) - old_value) > 1e-9:
            return None
    except ValueError:
        return None
    return sentence[:offset] + format_number(new_value) + sentence[end:]


def replace_number_in_sentence(sentence: str, old_value: float, new_value: float) -> str:
    """
    String-matching fallback when offset/length is missing or stale.
    Replaces the first occurrence of old_value (may hit the wrong occurrence
    if the number appears twice — prefer replace_at_offset).
    """
    old_str    = format_number(old_value)
    new_str    = format_number(new_value)

    # try exact string match first (most reliable)
    if old_str in sentence:
        return sentence.replace(old_str, new_str, 1)

    # try matching with comma formatting (e.g. "1,234" for 1234)
    old_comma = f"{int(old_value):,}" if old_value == int(old_value) else old_str
    if old_comma in sentence:
        return sentence.replace(old_comma, new_str, 1)

    # fallback: replace the first number token in the sentence
    return NUMBER_RE.sub(new_str, sentence, count=1)


def build_sentence(rec: dict, comment: str, anchor_num: float, new_num: float) -> tuple[str, bool]:
    """Substitute anchor_num -> new_num, preferring the offset span.
    Returns (sentence, used_offset)."""
    offset, length = rec.get("offset"), rec.get("length")
    if offset is not None and length is not None:
        replaced = replace_at_offset(comment, int(offset), int(length), anchor_num, new_num)
        if replaced is not None:
            return replaced, True
    return replace_number_in_sentence(comment, anchor_num, new_num), False


def run(input_path: str, output_path: str):
    log.info(f"Reading {input_path}")

    total       = 0
    skipped     = 0
    written     = 0
    offset_used = 0   # substitutions done via the exact offset span
    fallback    = 0   # substitutions that fell back to string matching

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            rec = json.loads(line)

            # skip if pos or neg number missing
            pos_num = rec.get("positive_number")
            neg_num = rec.get("negative_number")
            if pos_num is None or neg_num is None:
                skipped += 1
                continue

            anchor_num = float(rec["number"])
            pos_num    = float(pos_num)
            neg_num    = float(neg_num)
            comment    = str(rec.get("comment", ""))

            positive, pos_ok = build_sentence(rec, comment, anchor_num, pos_num)
            negative, neg_ok = build_sentence(rec, comment, anchor_num, neg_num)

            if pos_ok and neg_ok:
                offset_used += 1
            else:
                # offset span didn't verify (anchor embedded in a longer number,
                # e.g. "15.92" inside "RMB15.9219") — string fallback would
                # corrupt the text, so drop the record instead
                fallback += 1
                continue

            out = dict(rec)
            out["positive"] = positive
            out["negative"] = negative

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    log.info(f"  Total records   : {total:,}")
    log.info(f"  Skipped (None)  : {skipped:,}")
    log.info(f"  Written         : {written:,}")
    log.info(f"  Offset-exact    : {offset_used:,}")
    log.info(f"  Dropped         : {fallback:,} (offset span unverifiable — anchor inside a longer number)")

    # sample output
    log.info("\n── Sample output ──")
    with open(output_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            rec = json.loads(line)
            log.info(f"  anchor  : {rec['number']}")
            log.info(f"  comment : {rec['comment']}")
            log.info(f"  positive: {rec['positive']}  [{rec['positive_number']}]")
            log.info(f"  negative: {rec['negative']}  [{rec['negative_number']}]")
            log.info("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input, args.output)
