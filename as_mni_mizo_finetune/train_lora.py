"""
train.py
--------
Fine-tunes facebook/nllb-200-distilled-1.3B with a single multilingual LoRA
covering Assamese, Meitei (Manipuri), and Lushai (Mizo) (both directions)
in one checkpoint per direction.

Usage:
    python train.py --direction en_indic   # EN → Assamese/Meitei/Lushai
    python train.py --direction indic_en   # Assamese/Meitei/Lushai → EN
    python train.py --direction both       # run both sequentially

Before running:
    1. Run prepare_data.py to generate data/processed/combined_*.jsonl
    2. pip install -r requirements.txt

NLLB-200 vocab status for these languages:
  - eng_Latn  (English)                  → in vocab
  - asm_Beng  (Assamese)                 → in vocab
  - mni_Beng  (Meitei, Bengali script)   → in vocab
  - mni_Mtei  (Meitei, Meitei script)    → NOT in vocab; added via add_tokens()
  - lus_Latn  (Lushai/Mizo)              → NOT in vocab; added via add_tokens()

Why NllbTokenizer (slow, use_fast=False):
  NllbTokenizerFast._switch_to_target_mode() looks up tgt_lang in an internal
  lang_code_to_id dict populated at load time from the original vocab only.
  Tokens added later via add_tokens() are absent from that dict, so mni_Mtei
  and lus_Latn → None → int(None) → TypeError crash.  The slow tokenizer
  avoids this entirely; we tokenise targets by temporarily setting
  src_lang = tgt_lang.
"""

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizer,               # slow tokenizer — required for new lang tokens
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from dataset import Seq2SeqCollator, TranslationDataset

# ── GPU detection ─────────────────────────────────────────────────────────────
_bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
_DTYPE   = torch.bfloat16 if _bf16_ok else torch.float16

print(f"[GPU] CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GPU] Device         : {torch.cuda.get_device_name(0)}")
    print(f"[GPU] VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"[GPU] Using dtype    : {_DTYPE}")

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID = "facebook/nllb-200-distilled-1.3B"

DATA = {
    "train": "data/processed/combined_train.jsonl",
    "val":   "data/processed/combined_val.jsonl",
}

OUTPUT_BASE = Path("checkpoints")

# ── Language codes ─────────────────────────────────────────────────────────────
# NLLB-200 uses FLORES-200 language codes (ISO 639-3 + script tag).
#
#   eng_Latn  — English                   (already in NLLB vocab)
#   asm_Beng  — Assamese                  (already in NLLB vocab)
#   mni_Beng  — Meitei, Bengali script    (already in NLLB vocab)
#   mni_Mtei  — Meitei, Meitei script     (NOT in NLLB vocab — added below)
#   lus_Latn  — Lushai/Mizo              (NOT in NLLB vocab — added below)
#
# At inference time NLLB selects the output language via forced_bos_token_id,
# so the lang code token must be in the tokenizer vocab.
# ──────────────────────────────────────────────────────────────────────────────
LANG_CODES = {
    "english":       "eng_Latn",
    "assamese":      "asm_Beng",
    "meitei_beng":   "mni_Beng",
    "meitei_mtei":   "mni_Mtei",
    "lushai":        "lus_Latn",
}

VALID_LANGS: set[str] = set(LANG_CODES.values())

# ── LoRA config ────────────────────────────────────────────────────────────────
# NLLB-200 is built on top of the mBART architecture (transformer enc-dec).
# Attention projection names: q_proj, k_proj, v_proj, out_proj.
# r=8 (down from 16) to keep memory comfortable with 3 languages on 8 GB VRAM.
# Bump back to r=16 if you have headroom after monitoring peak VRAM.
LORA_CFG = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    bias="none",
)

# ── Training hyperparameters ───────────────────────────────────────────────────
# With ~9500 train samples and batch size 4 + grad_accum 8 (effective 32):
#   steps/epoch ≈ 9500 / 32 ≈ 297
#   total steps for 3 epochs ≈ 891
# eval/save every 250 steps → ~3 checkpoints per epoch; reasonable.

TRAIN_KWARGS = dict(
    num_train_epochs=3,                   # more epochs help with small data
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,        # effective batch = 32
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,                     # slightly longer warmup for small data
    bf16=_bf16_ok,
    fp16=not _bf16_ok,
    tf32=True,
    save_strategy="steps",
    save_steps=250,
    eval_strategy="steps",
    eval_steps=250,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=False,
    logging_steps=25,
    # Keep at 0: forked workers won't see tokenizer state for newly added tokens.
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    report_to="none",
    save_total_limit=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

MAX_SRC_LEN = 256
MAX_TGT_LEN = 256


# ── Helpers ────────────────────────────────────────────────────────────────────

def extend_tokenizer_for_new_langs(tokenizer, lang_codes: set[str]) -> list[str]:
    """
    Add any lang-code tokens missing from the NLLB tokenizer.

    asm_Beng and mni_Beng are natively in vocab.
    mni_Mtei (Meitei/Meitei script) and lus_Latn (Lushai/Mizo) are not —
    they get added here.

    Returns the list of tokens actually added.
    """
    vocab = tokenizer.get_vocab()
    to_add = [code for code in sorted(lang_codes) if code not in vocab]

    if to_add:
        tokenizer.add_tokens(to_add, special_tokens=True)
        print(f"  [tokenizer] Added {len(to_add)} new lang token(s): {to_add}")
    else:
        print("  [tokenizer] All lang codes already in vocab — no changes needed.")

    return to_add


def resize_model_embeddings(model, tokenizer) -> None:
    """
    Resize the model's token embeddings to match the (possibly extended)
    tokenizer vocabulary. New rows are mean-initialised by HuggingFace.
    """
    old_size = model.config.vocab_size
    new_size = len(tokenizer)
    if new_size != old_size:
        model.resize_token_embeddings(new_size)
        print(f"  [model] Embeddings resized: {old_size} → {new_size} tokens")
    else:
        print(f"  [model] Embedding size unchanged ({old_size})")


def set_src_lang(tokenizer, direction: str) -> None:
    """
    Set tokenizer.src_lang for the training direction.

    NllbTokenizer prepends the source-language token automatically when
    src_lang is set. For a multilingual dataset with mixed source langs,
    the dataset __getitem__ overrides this per-sample.
    """
    if direction == "en_indic":
        tokenizer.src_lang = LANG_CODES["english"]
    else:
        # indic_en: src is one of the Indic langs — default to Assamese;
        # per-sample override is handled in TranslationDataset.__getitem__.
        tokenizer.src_lang = LANG_CODES["assamese"]


def count_trainable(model) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct       = 100 * trainable / total
    return f"{trainable:,} / {total:,}  ({pct:.2f} %)"


def inspect_module_names(model) -> None:
    """Run once to verify LoRA target_modules match your checkpoint."""
    print("\nAttention-related module names:")
    for name, _ in model.named_modules():
        if any(k in name for k in ["proj", "attn", "self", "query", "key", "value"]):
            print(f"  {name}")


# ── Main training loop ─────────────────────────────────────────────────────────

def train_direction(direction: str) -> None:
    out_dir = OUTPUT_BASE / direction
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Direction : {direction}")
    print(f"  Languages : Assamese (asm_Beng), Meitei/Bengali (mni_Beng), Meitei/Meitei (mni_Mtei), Lushai (lus_Latn)")
    print(f"  Base model: {MODEL_ID}")
    print(f"  Output    : {out_dir}")
    print(f"{'='*60}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # use_fast=False is REQUIRED: NllbTokenizerFast crashes on newly added
    # lang codes (mni_Mtei, lus_Latn) because _switch_to_target_mode() looks
    # them up in a dict built only from the original vocab.
    tokenizer = NllbTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    # asm_Beng and mni_Beng are already in vocab; mni_Mtei and lus_Latn are added here.
    added_tokens = extend_tokenizer_for_new_langs(tokenizer, VALID_LANGS)

    # Set default src_lang (NllbTokenizer uses this to prepend the lang token).
    set_src_lang(tokenizer, direction)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=_DTYPE,
        device_map="cuda",
    )

    # Standard HF resize — works cleanly with NLLB.
    resize_model_embeddings(model, tokenizer)

    # Uncomment once to verify LoRA target_modules names:
    # inspect_module_names(model)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LORA_CFG)

    print(f"\n  Trainable params: {count_trainable(model)}")
    model.print_trainable_parameters()

    # ── Datasets ───────────────────────────────────────────────────────────────
    print("\nLoading datasets …")
    train_ds = TranslationDataset(
        DATA["train"], tokenizer, MAX_SRC_LEN, MAX_TGT_LEN, valid_langs=VALID_LANGS
    )
    val_ds = TranslationDataset(
        DATA["val"], tokenizer, MAX_SRC_LEN, MAX_TGT_LEN, valid_langs=VALID_LANGS
    )
    collator = Seq2SeqCollator(tokenizer)

    # ── Trainer ────────────────────────────────────────────────────────────────
    args = Seq2SeqTrainingArguments(output_dir=str(out_dir), **TRAIN_KWARGS)

    trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=collator,    
    )

    print("\nStarting training …")
    # trainer.train()
    # To resume from a checkpoint, pass e.g.:
    trainer.train(resume_from_checkpoint="checkpoints/en_indic/checkpoint-5500")

    # ── Save adapter + extended tokenizer ─────────────────────────────────────
    final = out_dir / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"\n  Saved to {final}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direction",
        choices=["en_indic", "indic_en", "both"],
        default="both",
    )
    args = parser.parse_args()

    if args.direction == "both":
        train_direction("en_indic")
        train_direction("indic_en")
    else:
        train_direction(args.direction)


if __name__ == "__main__":
    main()