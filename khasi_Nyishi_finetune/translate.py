"""
translate.py
------------
Run inference with the fine-tuned LoRA checkpoints.

Examples:
    # Single sentence
    python translate.py \
        --direction en_indic \
        --tgt_lang kha_Latn \
        --text "The river is very long."

    # Batch from a file (one sentence per line)
    python translate.py \
        --direction indic_en \
        --src_lang kha_Latn \
        --input_file sentences.txt \
        --output_file translations.txt
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── configuration ─────────────────────────────────────────────────────────────
MODEL_IDS = {
    "en_indic": "ai4bharat/indictrans2-en-indic-1B",
    "indic_en": "ai4bharat/indictrans2-indic-en-1B",
}

LORA_DIRS = {
    "en_indic": "checkpoints/en_indic/final",
    "indic_en": "checkpoints/indic_en/final",
}

LANG_TOKEN_FMT = ">>{lang}<<"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────


def load_model(direction: str):
    base_id  = MODEL_IDS[direction]
    lora_dir = LORA_DIRS[direction]

    tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(DEVICE)

    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()
    return model, tokenizer


def translate_batch(
    texts: list[str],
    tgt_lang: str,
    model,
    tokenizer,
    beam_size: int = 5,
    max_new_tokens: int = 256,
) -> list[str]:
    prefixed = [f"{LANG_TOKEN_FMT.format(lang=tgt_lang)} {t}" for t in texts]

    inputs = tokenizer(
        prefixed,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=beam_size,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction",   required=True, choices=["en_indic", "indic_en"])
    parser.add_argument("--tgt_lang",    required=True, help="e.g. kha_Latn or nyi_Latn or eng_Latn")
    parser.add_argument("--text",        default=None,  help="Single sentence to translate")
    parser.add_argument("--input_file",  default=None,  help="File with one sentence per line")
    parser.add_argument("--output_file", default=None,  help="Where to write translations")
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--beam_size",   type=int, default=5)
    args = parser.parse_args()

    print(f"Loading model ({args.direction}) …")
    model, tokenizer = load_model(args.direction)

    if args.text:
        result = translate_batch([args.text], args.tgt_lang, model, tokenizer, args.beam_size)
        print(f"\nTranslation: {result[0]}")
        return

    if args.input_file:
        sentences = Path(args.input_file).read_text(encoding="utf-8").splitlines()
        sentences = [s.strip() for s in sentences if s.strip()]

        translations = []
        for i in range(0, len(sentences), args.batch_size):
            batch = sentences[i : i + args.batch_size]
            translations.extend(translate_batch(batch, args.tgt_lang, model, tokenizer, args.beam_size))
            print(f"  Translated {min(i + args.batch_size, len(sentences))}/{len(sentences)}")

        if args.output_file:
            Path(args.output_file).write_text("\n".join(translations), encoding="utf-8")
            print(f"Saved {len(translations)} translations to {args.output_file}")
        else:
            for src, tgt in zip(sentences, translations):
                print(f"SRC: {src}")
                print(f"TGT: {tgt}\n")


if __name__ == "__main__":
    main()
