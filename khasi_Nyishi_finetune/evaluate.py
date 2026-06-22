"""
evaluate.py
-----------
Compute BLEU and chrF scores on the held-out validation set for each
language pair individually, so you can see per-language performance.

Usage:
    python evaluate.py --direction en_indic
    python evaluate.py --direction indic_en
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from translate import load_model, translate_batch

LANG_PAIRS = {
    "en_indic": [
        {"src_lang": "eng_Latn", "tgt_lang": "kha_Latn", "label": "EN→KHA"},
        {"src_lang": "eng_Latn", "tgt_lang": "nyi_Latn", "label": "EN→NYI"},
    ],
    "indic_en": [
        {"src_lang": "kha_Latn", "tgt_lang": "eng_Latn", "label": "KHA→EN"},
        {"src_lang": "nyi_Latn", "tgt_lang": "eng_Latn", "label": "NYI→EN"},
    ],
}

VAL_FILE   = "data/processed/combined_val.jsonl"
BATCH_SIZE = 16


def load_val_pairs(direction: str) -> dict[str, list]:
    """Group validation records by (src_lang, tgt_lang)."""
    pairs: dict[str, dict] = {}
    for cfg in LANG_PAIRS[direction]:
        key = f"{cfg['src_lang']}-{cfg['tgt_lang']}"
        pairs[key] = {"label": cfg["label"], "src": [], "ref": []}

    with open(VAL_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = f"{rec['src_lang']}-{rec['tgt_lang']}"
            if key in pairs:
                pairs[key]["src"].append(rec["src"])
                pairs[key]["ref"].append(rec["tgt"])

    return pairs


def evaluate(direction: str):
    print(f"Loading model for direction: {direction} …")
    model, tokenizer = load_model(direction)

    bleu_metric = BLEU(effective_order=True)
    chrf_metric = CHRF()

    pairs = load_val_pairs(direction)

    print(f"\n{'Lang Pair':<12} {'BLEU':>8} {'chrF':>8} {'Samples':>8}")
    print("-" * 42)

    for key, data in pairs.items():
        src_list = data["src"]
        ref_list = data["ref"]
        tgt_lang = key.split("-")[1]

        if not src_list:
            print(f"{data['label']:<12} {'N/A':>8} {'N/A':>8} {'0':>8}")
            continue

        hyps = []
        for i in range(0, len(src_list), BATCH_SIZE):
            batch = src_list[i : i + BATCH_SIZE]
            hyps.extend(translate_batch(batch, tgt_lang, model, tokenizer))

        bleu = bleu_metric.corpus_score(hyps, [ref_list])
        chrf = chrf_metric.corpus_score(hyps, [ref_list])

        print(f"{data['label']:<12} {bleu.score:>8.2f} {chrf.score:>8.2f} {len(src_list):>8}")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", required=True, choices=["en_indic", "indic_en", "both"])
    args = parser.parse_args()

    if args.direction == "both":
        evaluate("en_indic")
        evaluate("indic_en")
    else:
        evaluate(args.direction)


if __name__ == "__main__":
    main()
