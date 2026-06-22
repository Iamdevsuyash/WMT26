"""
prepare_data.py
---------------
Your data is already in JSONL format:
    {"src": "...", "tgt": "...", "src_lang": "eng_Latn", "tgt_lang": "kha_Latn"}

This script:
  1. Reads your existing JSONL files for Khasi and Nyishi
  2. Normalises legacy lang codes to NLLB-200-compatible FLORES-200 codes
  3. Adds the reverse direction (tgt→src) for each pair
  4. Combines everything and writes train/val splits

Edit JSONL_FILES below to match your actual files.

NLLB-200 lang code format
--------------------------
NLLB-200 uses FLORES-200 codes: ISO 639-3 (3-letter) + "_" + script tag.
These are the same BCP-47-style codes as IndicTrans2, so your existing
JSONL files should need minimal changes.

  eng_Latn  — English   (in NLLB vocab)
  kha_Latn  — Khasi     (in NLLB vocab — it's a FLORES-200 language)
  nif_Latn  — Nyishi    (NOT in NLLB vocab — train.py adds it as a new token)
"""

import json
import random
from pathlib import Path

# ── EDIT THESE ────────────────────────────────────────────────────────────────
JSONL_FILES = [
    "en_kha.jsonl",      # Khasi   (eng_Latn ↔ kha_Latn)
    "en_njz.jsonl",      # Nyishi  (eng_Latn ↔ nif_Latn)
]

VAL_RATIO = 0.05    # 5 % held out for validation
SEED      = 42
OUT_DIR   = Path("data/processed")
# ──────────────────────────────────────────────────────────────────────────────

# ── Lang-code normalisation ───────────────────────────────────────────────────
# Maps whatever codes your JSONL files contain → FLORES-200 / NLLB codes.
#
# Changes from the IndicTrans2 version:
#   • Logic is the same — FLORES-200 codes are identical to IT2 codes.
#   • Only the supported-set comment/report changes: kha_Latn is confirmed
#     in NLLB-200's vocab; nif_Latn is not.
# ──────────────────────────────────────────────────────────────────────────────
LANG_CODE_MAP: dict[str, str] = {
    "en_Latn":  "eng_Latn",   # short → 3-letter
    "eng_Latn": "eng_Latn",   # already correct
    "kha_Latn": "kha_Latn",   # Khasi — correct, no change
    "njz_Latn": "nif_Latn",   # Naxi code was wrong; Nyishi = nif
    "nif_Latn": "nif_Latn",   # already correct
}

# NLLB-200 natively supports these FLORES-200 codes (subset relevant here).
# kha_Latn IS in NLLB vocab (unlike IndicTrans2 where it may not be).
# nif_Latn is NOT — train.py registers it as a new special token.
NLLB_SUPPORTED: set[str] = {
    "eng_Latn", "kha_Latn",
    # Add others here if you expand to more languages later.
    # Full list: https://github.com/facebookresearch/flores/tree/main/flores200
}


def normalise_lang(code: str) -> str:
    """Return the canonical FLORES-200 lang code, or raise clearly if unknown."""
    canon = LANG_CODE_MAP.get(code)
    if canon is None:
        raise ValueError(
            f"Unknown lang code: {code!r}.\n"
            f"Add it to LANG_CODE_MAP in prepare_data.py."
        )
    return canon


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping malformed line {lineno} in {path}: {e}")
    return records


def normalise_records(records: list[dict], source_file: str) -> list[dict]:
    """Apply LANG_CODE_MAP to every record; warn and skip bad ones."""
    out = []
    skipped = 0
    for r in records:
        try:
            r["src_lang"] = normalise_lang(r["src_lang"])
            r["tgt_lang"] = normalise_lang(r["tgt_lang"])
        except ValueError as e:
            print(f"  WARNING: skipping record ({e})")
            skipped += 1
            continue
        if not r.get("src", "").strip() or not r.get("tgt", "").strip():
            skipped += 1
            continue
        out.append(r)
    if skipped:
        print(f"  Skipped {skipped} records in {source_file}")
    return out


def add_reverse(records: list[dict]) -> list[dict]:
    """For every A→B pair, append the B→A direction."""
    augmented = []
    for r in records:
        augmented.append(r)
        augmented.append({
            "src":      r["tgt"],
            "tgt":      r["src"],
            "src_lang": r["tgt_lang"],
            "tgt_lang": r["src_lang"],
        })
    return augmented


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    all_records: list[dict] = []
    all_lang_codes: set[str] = set()

    for fpath in JSONL_FILES:
        print(f"\nReading: {fpath}")
        recs = load_jsonl(fpath)
        print(f"  Raw records      : {len(recs)}")

        recs = normalise_records(recs, fpath)
        print(f"  After normalise  : {len(recs)}")

        langs = set((r["src_lang"], r["tgt_lang"]) for r in recs)
        print(f"  Lang pairs       : {langs}")
        for code in (c for pair in langs for c in pair):
            all_lang_codes.add(code)

        recs = add_reverse(recs)
        print(f"  After reverse aug: {len(recs)}")
        all_records.extend(recs)

    random.seed(SEED)
    random.shuffle(all_records)

    n_val = max(1, int(len(all_records) * VAL_RATIO))
    val   = all_records[:n_val]
    train = all_records[n_val:]

    write_jsonl(train, OUT_DIR / "combined_train.jsonl")
    write_jsonl(val,   OUT_DIR / "combined_val.jsonl")

    print(f"\n{'='*55}")
    print(f"  Train : {len(train):,}")
    print(f"  Val   : {len(val):,}")
    print(f"  Saved → {OUT_DIR}/")

    # Report which codes need to be registered as new tokens in train.py
    new_codes = all_lang_codes - NLLB_SUPPORTED
    if new_codes:
        print(f"\n  ⚠  Lang codes NOT in NLLB-200 vocab (train.py will add them):")
        for c in sorted(new_codes):
            print(f"      {c}")
    else:
        print("\n  ✓  All lang codes are natively supported by NLLB-200.")

    print("\nSample records (first 3):")
    for r in all_records[:3]:
        print(" ", r)


if __name__ == "__main__":
    main()